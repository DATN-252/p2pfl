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

class DSGTAggregator(Aggregator):
    """
    Implements a self-contained Distributed Stochastic Gradient Tracking (DSGT)
    aggregator for the P2PFL framework.
    """

    def __init__(
        self,
        alpha: float = 0.01,
        mixing_weights: Dict[str, float] = None,
        local_nn_model: Optional[nn.Module] = None,
        **kwargs
    ):
        """
        Initializes the DSGT Aggregator.
        """
        super().__init__(learning_rate=alpha, **kwargs)
        
        self.alpha = alpha
        self.mixing_weights = mixing_weights if mixing_weights is not None else {}
        
        # --- Internal State Management ---
        self.local_nn_model = local_nn_model
        self.y_tracker: List[torch.Tensor] = []
        self.prev_grads: List[torch.Tensor] = []
        
        self.device = 'cpu'
        self.is_initialized = False

        if self.local_nn_model is not None:
            try:
                self.device = next(self.local_nn_model.parameters()).device
            except Exception:
                pass

    @torch.no_grad()
    def aggregate(self, models: List[P2PFLModel]) -> P2PFLModel:
        """
        Performs one full, self-contained round of the DSGT algorithm.
        """
        # --- 1. One-Time Lazy Initialization (Round 0) ---
        if not self.is_initialized:
            return self._lazy_initialize(models)

        # --- 2. Get Current Gradient (g_k) ---
        current_grads = self._get_current_gradients()

        # --- 3. Step 1: Model Update (Consensus on X) ---
        w_ii = self.mixing_weights.get(self.addr, 0.0)
        new_x_params = []
        for i, p in enumerate(self.local_nn_model.parameters()):
            # term = x_i - alpha * y_i
            term = p.data.clone() - self.alpha * self.y_tracker[i]
            new_x_params.append(term * w_ii)

        for neighbor_model in models:
            neighbor_contributors = neighbor_model.get_contributors()
            if self.addr in neighbor_contributors:
                continue 

            neighbor_addr = neighbor_contributors[0]
            w_ij = self.mixing_weights.get(neighbor_addr, 0.0)
            
            if w_ij > 0:
                self._accumulate_neighbor_model(new_x_params, neighbor_model, w_ij)
        
        # Apply update to local model weights immediately
        for i, p in enumerate(self.local_nn_model.parameters()):
            p.data.copy_(new_x_params[i])

        # --- 4. Step 2: Tracker Update (Consensus on Y + Innovation) ---
        consensus_y = []
        for i, y_param in enumerate(self.y_tracker):
            consensus_y.append(y_param.clone() * w_ii)

        for neighbor_model in models:
            neighbor_contributors = neighbor_model.get_contributors()
            if self.addr in neighbor_contributors:
                continue

            neighbor_addr = neighbor_contributors[0]
            w_ij = self.mixing_weights.get(neighbor_addr, 0.0)
            
            if w_ij > 0:
                self._accumulate_neighbor_tracker(consensus_y, neighbor_model, w_ij)

        # Innovation = g_k - g_{k-1}
        for i in range(len(self.y_tracker)):
            grad_innovation = current_grads[i] - self.prev_grads[i]
            self.y_tracker[i].copy_(consensus_y[i] + grad_innovation)

        # --- 5. State Update for Next Round ---
        self.prev_grads = [g.clone() for g in current_grads]

        # Return result
        return self._build_return_model()

    def _lazy_initialize(self, models: List[P2PFLModel]) -> P2PFLModel:
        """Performs initialization of y_tracker and prev_grads at Round 0."""
        if self.local_nn_model is None:
            local_p2pfl_model = None
            for m in models:
                if self.addr in m.get_contributors():
                    local_p2pfl_model = m
                    break
            
            if local_p2pfl_model is None:
                if not models:
                     raise NoModelsToAggregateError("Cannot initialize DSGT: model list is empty.")
                local_p2pfl_model = models[0] 
                logger.warning(self.addr, "Could not find self in model list; assuming models[0] is self.")

            self.local_nn_model = local_p2pfl_model.get_model()

        try:
            self.device = next(self.local_nn_model.parameters()).device
        except StopIteration:
            self.device = 'cpu'
        
        # Initialize State: y_0 = g_0, prev_grads = g_0
        initial_grads = self._get_current_gradients()
        
        self.y_tracker = [g.clone() for g in initial_grads]
        self.prev_grads = [g.clone() for g in initial_grads]
        
        self.is_initialized = True
        logger.info(self.addr, f"DSGT initialized. Device: {self.device}. Alpha: {self.alpha}")
        
        return self._build_return_model()

    def _get_current_gradients(self) -> List[torch.Tensor]:
        """Safely extracts gradients from the local model."""
        grads = []
        for p in self.local_nn_model.parameters():
            if p.grad is None:
                grads.append(torch.zeros_like(p.data).to(self.device))
            else:
                grads.append(p.grad.data.clone().to(self.device))
        return grads

    def _accumulate_neighbor_model(self, new_x_params, neighbor_model, w_ij):
        """Accumulates neighbor's X contribution: w_ij * (x_j - alpha * y_j)"""
        neighbor_x_np = neighbor_model.get_parameters()
        neighbor_y_np = neighbor_model.additional_info.get('dsgt_y', [])

        for i, x_param_new in enumerate(new_x_params):
            x_j = torch.from_numpy(neighbor_x_np[i]).to(self.device)
            
            if neighbor_y_np and len(neighbor_y_np) > i:
                y_j = torch.from_numpy(neighbor_y_np[i]).to(self.device)
            else:
                y_j = torch.zeros_like(x_j)
            
            term = x_j - self.alpha * y_j
            x_param_new.add_(term, alpha=w_ij)

    def _accumulate_neighbor_tracker(self, consensus_y, neighbor_model, w_ij):
        """Accumulates neighbor's Y contribution: w_ij * y_j"""
        neighbor_y_np = neighbor_model.additional_info.get('dsgt_y', [])
        
        for i, y_consensus_param in enumerate(consensus_y):
            if neighbor_y_np and len(neighbor_y_np) > i:
                y_j = torch.from_numpy(neighbor_y_np[i]).to(self.device)
                y_consensus_param.add_(y_j, alpha=w_ij)

    def _build_return_model(self) -> P2PFLModel:
        """Constructs the P2PFLModel containing updated weights and piggybacked Y."""
        updated_y_np = [y.cpu().numpy() for y in self.y_tracker]
        updated_g_np = [g.cpu().numpy() for g in self.prev_grads]

        template = LightningModel(model=self.local_nn_model) 
        
        # --- FIX: Tạo model copy trước, KHÔNG truyền gradients_estimate vào build_copy ---
        new_p2pfl_model = template.build_copy(
            params=[p.data.cpu().numpy() for p in self.local_nn_model.parameters()],
            additional_info={'dsgt_y': updated_y_np}, # Vẫn gửi y qua additional_info
        )

        # --- FIX: Gán thủ công sau khi object đã được tạo ---
        # Điều này tránh lỗi TypeError vì __init__ không nhận tham số này
        new_p2pfl_model.gradients_estimate = updated_g_np

        new_p2pfl_model.set_contribution(contributors=[self.addr], num_samples=0)
        return new_p2pfl_model