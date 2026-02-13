import torch
import torch.nn as nn
import numpy as np
import copy
from typing import List, Dict, Any, Optional

from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel
from p2pfl.learning.aggregators.aggregator import Aggregator, NoModelsToAggregateError
from p2pfl.management.logger import logger
# Đảm bảo đường dẫn này đúng với project của bạn
from p2pfl.learning.frameworks.pytorch.lightning_model import LightningModel 

class DSGT(Aggregator):
    """
    Implements the architecturally-compliant "Lagged" version of the 
    Distributed Stochastic Gradient Tracking (DSGT) algorithm.
    This version uses the innovation term g_k - g_{k-1}.
    """
    requires_gradient_only: bool = True
    REQUIRED_INFO_KEYS = ["delta", "degrees"]

    def __init__(self, alpha: float = 0.01, **kwargs):
        """
        Initializes the Lagged DSGT Aggregator.
        Args:
            alpha (float): The learning rate (step size).
        """
        super().__init__(learning_rate=alpha, **kwargs)
        
        self.alpha = alpha
        
        # --- Internal State Management ---
        # y_tracker stores the gradient tracker state (y_k)
        self.y_tracker: List[np.ndarray] = []
        # prev_grads stores the gradient from the previous iteration (g_{k-1})
        self.prev_grads: List[np.ndarray] = []
        
        self.is_initialized = False

    def aggregate(self, models: List[P2PFLModel]) -> P2PFLModel:
        """
        Performs one round of the Lagged DSGT algorithm.
        """
        if not models:
            raise NoModelsToAggregateError(f"({self.addr}) No models to aggregate")

        # --- 1. Setup: Map models by address for robust access ---
        model_map = {m.get_contributors()[0]: m for m in models}
        self_model = model_map.get(self.addr)
        if self_model is None:
            raise NoModelsToAggregateError("Self model not found in the aggregation list for DSGT.")

        # --- 2. Get Current Gradient (g_k) from the callback's delta ---
        # delta = -lr * g_k  =>  g_k = -delta / lr
        self_info = self._get_and_validate_model_info(self_model)
        current_grads = [-d / self.alpha for d in self_info["delta"]]

        # --- 3. One-Time Initialization (Round 0) ---
        if not self.is_initialized:
            self.y_tracker = [g.copy() for g in current_grads]
            self.prev_grads = [g.copy() for g in current_grads]
            self.is_initialized = True
            logger.info(self.addr, f"Lagged DSGT initialized. Alpha: {self.alpha}")

        # --- 4. Calculate Metropolis-Hastings Weights (Dynamically) ---
        my_degree = int(self_info["degrees"])
        neighbor_weights = {}
        valid_neighbors = {}
        
        for addr, model in model_map.items():
            if addr == self.addr:
                continue
            try:
                neighbor_info = self._get_and_validate_model_info(model)
                neighbor_degree = int(neighbor_info["degrees"])
                neighbor_weights[addr] = 1.0 / (1.0 + max(my_degree, neighbor_degree))
                valid_neighbors[addr] = model
            except (ValueError, KeyError):
                logger.debug(self.addr, f"Skipping neighbor {addr} in DSGT aggregation: missing metadata.")
        
        self_weight = 1.0 - sum(neighbor_weights.values())
        weights = {**neighbor_weights, self.addr: self_weight}

        # --- 5. Step 1: Solution Update (x_{k+1}) ---
        # x_{k+1} = Σ w_ij * (x_j - α * y_j)
        new_x_params = [np.zeros_like(p) for p in self_model.get_parameters()]

        # Use valid_neighbors + self
        for addr in list(valid_neighbors.keys()) + [self.addr]:
            model = model_map[addr]
            w_ij = weights.get(addr, 0.0)
            if w_ij == 0.0:
                continue
            
            x_j = model.get_parameters()
            # On round 0, neighbors might not have a y_tracker yet
            y_j = model.additional_info.get('dsgt_y', self.y_tracker)
            
            for i, (p_x, p_y) in enumerate(zip(x_j, y_j)):
                term = p_x - self.alpha * p_y
                new_x_params[i] += w_ij * term
        
        # --- 6. Step 2: Tracker Update (y_{k+1}) ---
        # y_{k+1} = Σ w_ij * y_j + (g_k - g_{k-1})
        consensus_y = [np.zeros_like(p) for p in self.y_tracker]
        for addr in list(valid_neighbors.keys()) + [self.addr]:
            model = model_map[addr]
            w_ij = weights.get(addr, 0.0)
            if w_ij == 0.0:
                continue

            y_j = model.additional_info.get('dsgt_y', self.y_tracker)
            for i, p_y in enumerate(y_j):
                consensus_y[i] += w_ij * p_y

        # Lagged Innovation = g_k - g_{k-1}
        for i in range(len(self.y_tracker)):
            grad_innovation = current_grads[i] - self.prev_grads[i]
            self.y_tracker[i] = consensus_y[i] + grad_innovation

        # --- 7. State Update for Next Round ---
        self.prev_grads = [g.copy() for g in current_grads]

        # --- 8. Build and return the resulting model for the next round ---
        # The main result is the new parameters. The tracker (y) is piggybacked.
        return self_model.build_copy(
            params=new_x_params,
            additional_info={'dsgt_y': self.y_tracker}
        )

    def _get_and_validate_model_info(self, model: P2PFLModel) -> dict[str, Any]:
        """Helper to retrieve and validate required info from the model."""
        try:
            info = model.get_info("gradient_delta_calculator")
        except KeyError:
            info = model.get_info()
        
        for key in self.REQUIRED_INFO_KEYS:
            if key not in info:
                raise ValueError(f"Model missing '{key}' information required for DSGT.")
        return info

    def get_required_callbacks(self) -> list[str]:
        return ["gradient_delta_calculator"]