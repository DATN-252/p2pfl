import numpy as np
import math
from collections import defaultdict
from typing import Any, List, Dict
from p2pfl.learning.aggregators.aggregator import Aggregator, NoModelsToAggregateError
from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel
from p2pfl.management.logger import logger

class DFedAdp(Aggregator):
    SUPPORTS_PARTIAL_AGGREGATION: bool = False
    requires_gradient_only: bool = True
    REQUIRED_INFO_KEYS = ["delta", "degrees"] 
    

    def __init__(self, disable_partial_aggregation: bool = False, learning_rate: float = 0.001, log_dfedadp_params: bool = False, decay_rate: float = 0.95, min_learning_rate: float = 0.0001,alpha : float = 1.0) -> None:
        super().__init__(disable_partial_aggregation=disable_partial_aggregation)
        self.global_model_params: List[np.ndarray] = []
        # Map contributor_id -> smoothed_angle history
        self.node_correlation: Dict[str, float] = defaultdict(lambda: 0.0)
        # Learning rate
        self.learning_rate = learning_rate
        self.min_learning_rate = min_learning_rate
        self.decay_rate = decay_rate
        # Store previous local gradient for Gradient Tracking
        self.prev_local_gradient: List[np.ndarray] = []
        # Control logging of dfedadp parameters
        self.log_dfedadp_params = log_dfedadp_params
        # ALPHA for glompetz function
        self.ALPHA = alpha

    def aggregate(self, models: List[P2PFLModel]) -> P2PFLModel:
        # Validate input
        if len(models) == 0:
            raise NoModelsToAggregateError(f"({self.addr}) No models to aggregate")

        # --- Setup: Map models by address for robust access ---
        model_map = {m.get_contributors()[0]: m for m in models}
        self_model = model_map.get(self.addr)
        if self_model is None:
            raise NoModelsToAggregateError("Self model not found in the aggregation list for DFedAdp.")
        
        total_samples = sum(m.get_num_samples() for m in model_map.values())
        contributors = list(model_map.keys())
        current_round = self.each_trained_round.get(self.addr, 0)

        # --- Initial Round (Round 0) ---
        if not self.global_model_params:
            self.global_model_params = [p.copy() for p in self_model.get_parameters()]
            info = self._get_and_validate_model_info(self_model)
            delta = info["delta"]
            self.prev_local_gradient = [-d / self.learning_rate for d in delta]
            self_model.gradients_estimate = self.prev_local_gradient

        # --- 3. Calculate Metropolis-Hastings Weights (Robustly) ---
        my_degree = int(self._get_and_validate_model_info(self_model)["degrees"])
        
        neighbor_weights = {}
        for addr, model in model_map.items():
            if addr == self.addr:
                continue
            neighbor_degree = int(self._get_and_validate_model_info(model)["degrees"])
            neighbor_weights[addr] = 1.0 / (1.0 + max(my_degree, neighbor_degree))
        
        self_weight = 1.0 - sum(neighbor_weights.values())
        
        metro_weights = neighbor_weights
        metro_weights[self.addr] = self_weight

        # --- 4. Compute Current Local Gradient ---
        self_info = self._get_and_validate_model_info(self_model)
        self_delta = self_info["delta"]
        curr_local_gradient = [-d / self.learning_rate for d in self_delta]

        # --- 5. Gradient Tracking: Estimate Global Gradient ---
        weighted_neighbor_tracking = [np.zeros_like(p) for p in self.global_model_params]
        for addr, m in model_map.items():
            if hasattr(m, 'gradients_estimate') and m.gradients_estimate:
                g_j_prev = m.gradients_estimate
            else: # Fallback for first round
                d = self._get_and_validate_model_info(m)["delta"]
                g_j_prev = [-x / self.learning_rate for x in d]
            
            w_ij = metro_weights.get(addr, 0.0)
            weighted_neighbor_tracking = [acc + w_ij * g for acc, g in zip(weighted_neighbor_tracking, g_j_prev)]

        if not self.prev_local_gradient:
            self.prev_local_gradient = [np.zeros_like(p) for p in curr_local_gradient]
        tracking_gradient = [wn + curr - prev for wn, curr, prev in zip(weighted_neighbor_tracking, curr_local_gradient, self.prev_local_gradient)]
        self.prev_local_gradient = [g.copy() for g in curr_local_gradient]

        # --- 6. Calculate FedAdp Scores ---
        fedadp_scores = {}
        g_vec = np.concatenate([p.ravel() for p in tracking_gradient])
        g_norm = np.linalg.norm(g_vec)

        for addr, m in model_map.items():
            m_delta = self._get_and_validate_model_info(m)["delta"]
            neigh_local_grad = [-d / self.learning_rate for d in m_delta]
            l_vec = np.concatenate([p.ravel() for p in neigh_local_grad])
            l_norm = np.linalg.norm(l_vec)

            cos_sim = 1.0 if g_norm == 0 or l_norm == 0 else np.clip(np.dot(g_vec, l_vec) / (g_norm * l_norm), -1.0, 1.0)
            angle = float(np.arccos(cos_sim))

            prev_angle = self.node_correlation.get(addr, 0.0)
            smoothed_angle = angle if current_round <= 1 or prev_angle == 0.0 else ((current_round - 1)/current_round)*prev_angle + (1/current_round)*angle
            self.node_correlation[addr] = smoothed_angle
            
            f_val = self._gompertz_function(smoothed_angle)
            fedadp_scores[addr] = m.get_num_samples() * math.exp(f_val)

        # --- 7. Calculate Adaptive Mixing Matrix ---
        total_score = sum(fedadp_scores.values())
        psi = {addr: s / total_score if total_score > 0 else 1.0/len(model_map) for addr, s in fedadp_scores.items()}
        
        unnormalized_mix = {addr: psi[addr] * metro_weights[addr] for addr in model_map}
        sum_mix = sum(unnormalized_mix.values())
        final_mixing_weights = {addr: u / sum_mix if sum_mix > 0 else 1.0/len(model_map) for addr, u in unnormalized_mix.items()}
        
        # --- 8. Aggregation Step (Consensus) & 9. Final Update ---
        w_half = [np.zeros_like(p, dtype=np.float64) for p in self.global_model_params]
        for addr, m in model_map.items():
            w = final_mixing_weights.get(addr, 0.0)
            for i, layer in enumerate(m.get_parameters()):
                w_half[i] += layer * w

        clip_threshold = 5.0
        tracking_gradient = [np.clip(tg, -clip_threshold, clip_threshold) for tg in tracking_gradient]
        self.global_model_params = [wh - self.learning_rate * tg for wh, tg in zip(w_half, tracking_gradient)]

        # --- Build and return result ---
        result_model = self_model.build_copy(params=self.global_model_params, num_samples=total_samples, contributors=contributors)
        result_model.gradients_estimate = tracking_gradient
        self.learning_rate = max(self.learning_rate*self.decay_rate, self.min_learning_rate)
        return result_model

    def _gompertz_function(self, angle: float):
        # Non-linear Gompertz mapping function
        return self.ALPHA * (1 - math.exp(-math.exp(-self.ALPHA * (angle - 1))))
    
    def _get_and_validate_model_info(self, model: P2PFLModel) -> dict[str, Any]:
        try:
            info = model.get_info("gradient_delta_calculator")
        except KeyError:
            info = model.get_info()
        
        if "delta" not in info:
            raise ValueError(f"Model missing 'delta' information required for DFedAdp.")
        return info

    def get_required_callbacks(self) -> list[str]:
        return ["gradient_delta_calculator"]