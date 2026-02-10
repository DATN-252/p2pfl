import torch
import torch.nn as nn
import numpy as np
import copy
from typing import List, Dict, Any
from collections import OrderedDict

from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel
from p2pfl.learning.aggregators.aggregator import Aggregator, NoModelsToAggregateError
from p2pfl.management.logger import logger
from p2pfl.learning.frameworks.pytorch.lightning_model import LightningModel

class DSGTAggregator(Aggregator):
    """
    Implements a self-contained Distributed Stochastic Gradient Tracking (DSGT)
    aggregator for the P2PFL framework, using a "lazy initialization" pattern.

    This implementation is a drop-in replacement for other aggregators, requiring
    no changes to the core `Node` or `launch_from_yaml.py` script.

    **Key Pattern: Lazy Initialization**
    - The `__init__` signature is generic and compatible with the framework.
    - The first time `aggregate` is called, it identifies the local node's model
      from the list of received models, extracts the underlying `nn.Module`,
      and initializes its internal state (`y_0 = g_0`).

    **Algorithm Variant (Delayed Gradient Innovation):**
    - This implementation uses a delayed gradient innovation term, a practical
      necessity to remain self-contained within a single `aggregate` call.
      The tracker update is: `y_new = consensus(y) + g_current - g_previous_round`

    **Communication Strategy (Piggybacking):**
    - The aggregator piggybacks its tracker variable `y` onto the outgoing
      `P2PFLModel` using `additional_info['dsgt_y']`.
    """
    def __init__(
        self,
        alpha: float = 0.01,
        mixing_weights: Dict[str, float] = None,
        **kwargs
    ):
        """
        Initializes the DSGT Aggregator with a generic signature.

        Args:
            alpha (float): The learning rate for the DSGT update.
            mixing_weights (Dict[str, float]): A dictionary mapping node addresses
                (including self) to their mixing weights (w_ij).
        """
        super().__init__(learning_rate=alpha, **kwargs)
        
        self.alpha = alpha
        self.mixing_weights = mixing_weights if mixing_weights is not None else {}
        self.device = 'cpu' # Will be updated on first run

        # --- Internal State Management (Lazy Initialized) ---
        self.local_nn_model: nn.Module = None
        self.y_tracker: List[torch.Tensor] = []
        self.prev_grads: List[torch.Tensor] = []
        self.is_initialized = False

    @torch.no_grad()
    def aggregate(self, models: List[P2PFLModel]) -> P2PFLModel:
        """
        Performs one full, self-contained round of the DSGT algorithm.
        Handles one-time lazy initialization on the first call.
        """
        # --- One-Time Lazy Initialization (k=0) ---
        if not self.is_initialized:
            return self._lazy_initialize(models)

        # --- Get Current Gradient (g_k) ---
        current_grads = self._get_current_gradients()

        # --- Step 1: Model Update (x_new) ---
        # x_new = sum(w_ij * (x_j - alpha * y_j))
        w_ii = self.mixing_weights.get(self.addr, 0.0)
        new_x_params = []
        for i, p in enumerate(self.local_nn_model.parameters()):
            term = p.data.clone() - self.alpha * self.y_tracker[i]
            new_x_params.append(term * w_ii)

        for neighbor_model in models:
            if self.addr in neighbor_model.get_contributors():
                continue # Skip self

            neighbor_addr = neighbor_model.get_contributors()[0]
            w_ij = self.mixing_weights.get(neighbor_addr, 0.0)
            if w_ij > 0:
                self._accumulate_neighbor_model(new_x_params, neighbor_model, w_ij)
        
        for i, p in enumerate(self.local_nn_model.parameters()):
            p.data.copy_(new_x_params[i])

        # --- Step 2: Tracker Update (y_new) with Delayed Innovation ---
        # y_new = sum(w_ij * y_j) + (g_current - g_previous_round)
        consensus_y = []
        for i, y_param in enumerate(self.y_tracker):
            consensus_y.append(y_param.clone() * w_ii)

        for neighbor_model in models:
            if self.addr in neighbor_model.get_contributors():
                continue

            neighbor_addr = neighbor_model.get_contributors()[0]
            w_ij = self.mixing_weights.get(neighbor_addr, 0.0)
            if w_ij > 0:
                self._accumulate_neighbor_tracker(consensus_y, neighbor_model, w_ij)

        for i in range(len(self.y_tracker)):
            grad_innovation = current_grads[i] - self.prev_grads[i]
            self.y_tracker[i].copy_(consensus_y[i] + grad_innovation)

        # --- State Update for Next Round ---
        self.prev_grads = [g.clone() for g in current_grads]

        return self._build_return_model()

    def _lazy_initialize(self, models: List[P2PFLModel]) -> P2PFLModel:
        """Performs one-time initialization of the aggregator's state."""
        # Find the local model from the list to get the nn.Module instance
        local_p2pfl_model = None
        for m in models:
            if self.addr in m.get_contributors():
                local_p2pfl_model = m
                break
        
        if local_p2pfl_model is None:
            # Fallback if self model is not in the list (should not happen in p2pfl)
            if not models:
                 raise NoModelsToAggregateError("Cannot initialize DSGT: model list is empty.")
            local_p2pfl_model = models[0] # Assume first model is self as last resort
            logger.warning(self.addr, "Could not find self in model list; assuming models[0] is self.")

        self.local_nn_model = local_p2pfl_model.get_model()
        self.device = next(self.local_nn_model.parameters()).device
        
        # Get g_0 and initialize state
        initial_grads = self._get_current_gradients()
        self.y_tracker = [g.clone() for g in initial_grads]
        self.prev_grads = [g.clone() for g in initial_grads]
        
        self.is_initialized = True
        logger.info(self.addr, f"DSGT aggregator lazy-initialized for node '{self.addr}'.")
        
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
        """Helper to add a neighbor's model contribution."""
        neighbor_x_np = neighbor_model.get_parameters()
        neighbor_y_np = neighbor_model.additional_info.get('dsgt_y', [])

        for i, x_param_new in enumerate(new_x_params):
            x_j = torch.from_numpy(neighbor_x_np[i]).to(self.device)
            y_j = torch.zeros_like(x_j)
            if neighbor_y_np and len(neighbor_y_np) > i:
                 y_j = torch.from_numpy(neighbor_y_np[i]).to(self.device)
            
            term = x_j - self.alpha * y_j
            x_param_new.add_(term, alpha=w_ij)

    def _accumulate_neighbor_tracker(self, consensus_y, neighbor_model, w_ij):
        """Helper to add a neighbor's tracker contribution."""
        neighbor_y_np = neighbor_model.additional_info.get('dsgt_y', [])
        for i, y_consensus_param in enumerate(consensus_y):
            if neighbor_y_np and len(neighbor_y_np) > i:
                y_j = torch.from_numpy(neighbor_y_np[i]).to(self.device)
                y_consensus_param.add_(y_j, alpha=w_ij)

    def _build_return_model(self) -> P2PFLModel:
        """Helper to construct the P2PFLModel to be returned."""
        updated_y_np = [y.cpu().numpy() for y in self.y_tracker]
        updated_g_np = [g.cpu().numpy() for g in self.prev_grads]

        # Use the base class's build_copy method for consistency.
        # It requires a template model; we use one of the received models.
        # The parameters will be set from our updated local_nn_model.
        template = LightningModel(model=self.local_nn_model) # Create a fresh wrapper
        
        new_p2pfl_model = template.build_copy(
            params=[p.data.cpu().numpy() for p in self.local_nn_model.parameters()],
            additional_info={'dsgt_y': updated_y_np},
            gradients_estimate=updated_g_np
        )
        new_p2pfl_model.set_contribution(contributors=[self.addr], num_samples=0)
        return new_p2pfl_model
