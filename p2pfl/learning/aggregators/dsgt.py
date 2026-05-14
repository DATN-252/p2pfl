#
# This file is part of the federated_learning_p2p (p2pfl) distribution
# (see https://github.com/pguijas/p2pfl).
#

import torch
import numpy as np
from typing import List, Dict, Any, Optional

from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel
from p2pfl.learning.aggregators.aggregator import Aggregator, NoModelsToAggregateError
from p2pfl.management.logger import logger
from p2pfl.settings import Settings

class DSGTAggregator(Aggregator):
    """
    Implements Decentralized Stochastic Gradient Tracking with Heavy-ball momentum 
    over Time-Varying directed networks (DSGTm-TV).
    Ref: https://arxiv.org/html/2409.17189v1
    """
    requires_gradient_only: bool = False # Allow local training to produce non-zero delta
    # We need 'delta' to extract the gradient and 'degrees' for push-pull weights
    REQUIRED_INFO_KEYS = ["delta", "degrees"]

    def __init__(self, alpha: float = 0.01, beta: float = 0.0, **kwargs):
        """
        Initializes the DSGTm-TV Aggregator.
        Args:
            alpha (float): Stepsize (uncoordinated per node).
            beta (float): Heavy-ball momentum parameter (uncoordinated per node).
        """
        super().__init__(learning_rate=alpha, **kwargs)
        
        self.alpha = alpha
        self.beta = beta
        
        # --- Internal State Management ---
        self.y_tracker: Optional[List[np.ndarray]] = None      # y_k
        self.consensus_y: Optional[List[np.ndarray]] = None    # sum(B_ij * y_k^j)
        self.prev_x: Optional[List[np.ndarray]] = None          # x_{k-1}
        self.prev_g: Optional[List[np.ndarray]] = None          # g_{k-1}
        
        self.is_initialized = False

    def preprocess_local_model(self, model: P2PFLModel) -> P2PFLModel:
        """
        Calculates the gradient tracker update (y_k) and prepares the weighted 
        tracker (dsgt_y_msg) for Push-Pull communication.
        """
        # 1. Extract Current Gradient (g_k) from the callback's delta
        # delta = weights_start - weights_end ≈ alpha * g_k  =>  g_k ≈ delta / alpha
        info = self._get_and_validate_model_info(model)
        self_delta = [np.array(d) for d in info["delta"]]
        g_k = [d / self.alpha for d in self_delta]
        
        # 2. Initialization (Round 0)
        if not self.is_initialized:
            self.y_tracker = [g.copy() for g in g_k]
            self.prev_g = [g.copy() for g in g_k]
            
            # Recover x_k^i (before local update) for momentum term in next round
            x_updated = model.get_parameters()
            x_k_i = [p + d for p, d in zip(x_updated, self_delta)]
            self.prev_x = [p.copy() for p in x_k_i]
            
            self.is_initialized = True
            logger.info(self.addr, f"DSGTm-TV initialized. Alpha: {self.alpha}, Beta: {self.beta}")
        else:
            # 3. Tracker Update: y_{k} = consensus_y + g_k - g_{k-1}
            if self.consensus_y is not None:
                for i in range(len(self.y_tracker)):
                    self.y_tracker[i] = self.consensus_y[i] + (g_k[i] - self.prev_g[i])
            self.prev_g = [g.copy() for g in g_k]

        # 4. Prepare Pushed Message (Column-stochastic weighting)
        # out_degree_plus_1 is the number of neighbors we are pushing to + ourselves
        out_degree_plus_1 = len(self.train_set) if self.train_set else 1
        
        dsgt_y_msg = [y / out_degree_plus_1 for y in self.y_tracker]
        
        # Attach the weighted tracker to the model's additional info
        model.additional_info['dsgt_y_msg'] = dsgt_y_msg
        return model

    def aggregate(self, models: List[P2PFLModel]) -> P2PFLModel:
        """
        Performs Consensus on model parameters (A_k) and gradient trackers (B_k).
        Then performs the Heavy-ball momentum model update.
        """
        if not models:
            raise NoModelsToAggregateError(f"({self.addr}) No models to aggregate")

        # --- 1. Identify Self Model and Extract Parameters ---
        model_map = {m.get_contributors()[0]: m for m in models}
        self_model = model_map.get(self.addr)
        if self_model is None:
            raise NoModelsToAggregateError("Self model not found in DSGT aggregation.")

        # x_updated is the model AFTER local training (x_k - alpha*g_k)
        x_updated = self_model.get_parameters()
        
        # --- 2. Calculate Consensus for X (Row-stochastic A_k) ---
        num_models = len(models)
        consensus_x = [np.zeros_like(p) for p in x_updated]
        for m in models:
            # Recover x_k^j by adding back the delta to the updated model
            # x_k = x_updated + delta
            m_info = self._get_and_validate_model_info(m)
            m_delta = [np.array(d) for d in m_info["delta"]]
            m_params_updated = m.get_parameters()
            
            for i, (p_upd, d) in enumerate(zip(m_params_updated, m_delta)):
                x_k_j = p_upd + d
                consensus_x[i] += x_k_j / num_models

        # --- 3. Calculate Consensus for Y (Column-stochastic B_k) ---
        self.consensus_y = [np.zeros_like(p) for p in self.y_tracker]
        for m in models:
            y_msg_j = m.additional_info.get('dsgt_y_msg')
            if y_msg_j is not None:
                for i, p_y in enumerate(y_msg_j):
                    self.consensus_y[i] += p_y

        # --- 4. Model Update (x_{k+1}) with Momentum ---
        # Current local model before update (x_k^i)
        self_info = self._get_and_validate_model_info(self_model)
        self_delta = [np.array(d) for d in self_info["delta"]]
        x_k_i = [p + d for p, d in zip(x_updated, self_delta)]

        # x_{k+1} = consensus_x - alpha * y_tracker + beta * (x_k - x_{prev})
        new_x_params = []
        for i in range(len(x_k_i)):
            update = consensus_x[i] - self.alpha * self.y_tracker[i] + self.beta * (x_k_i[i] - self.prev_x[i])
            new_x_params.append(update)

        # --- 5. State Update for Next Round ---
        self.prev_x = [p.copy() for p in x_k_i]

        # Return the resulting model for the next round (x_{k+1})
        return self_model.build_copy(params=new_x_params)

    def _get_and_validate_model_info(self, model: P2PFLModel) -> dict[str, Any]:
        """Helper to retrieve and validate required info from the model."""
        try:
            info = model.get_info("gradient_delta_calculator")
        except KeyError:
            info = model.get_info()
        
        for key in self.REQUIRED_INFO_KEYS:
            if key not in info:
                # Provide useful debug info
                raise ValueError(f"Model missing '{key}' information required for DSGT. info keys: {list(info.keys())}")
        return info

    def get_required_callbacks(self) -> list[str]:
        return ["gradient_delta_calculator"]
