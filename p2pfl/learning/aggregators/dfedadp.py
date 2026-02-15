import numpy as np
import math
from collections import defaultdict
from typing import Any, List, Dict
from p2pfl.learning.aggregators.aggregator import Aggregator, NoModelsToAggregateError
from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel
from p2pfl.management.logger import logger

class DFedAdp(Aggregator):
    SUPPORTS_PARTIAL_AGGREGATION: bool = False
    requires_gradient_only: bool = False
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
        # Store previous local pseudo-gradient for Gradient Tracking
        self.prev_local_gradient: List[np.ndarray] = []
        # Control logging of dfedadp parameters
        self.log_dfedadp_params = log_dfedadp_params
        # ALPHA for glompetz function
        self.ALPHA = alpha

    def aggregate(self, models: List[P2PFLModel]) -> P2PFLModel:
        # Validate input
        if len(models) == 0:
            raise NoModelsToAggregateError(f"({self.addr}) No models to aggregate")

        # --- Setup ---
        # Normalize addresses for robust mapping
        norm_self_addr = self.normalize_addr(self.addr)
        model_map = {}
        for m in models:
            contributors = m.get_contributors()
            if contributors:
                norm_contributor = self.normalize_addr(contributors[0])
                model_map[norm_contributor] = m

        self_model = model_map.get(norm_self_addr)
        if self_model is None:
            # Provide more diagnostic info
            available_nodes = list(model_map.keys())
            raise NoModelsToAggregateError(f"Self model ({norm_self_addr}) not found in the aggregation list for DFedAdp. Available: {available_nodes}")
        
        total_samples = sum(m.get_num_samples() for m in model_map.values())
        contributors = list(model_map.keys())
        current_round = self.each_trained_round.get(self.addr, 0)

        def get_delta(m):
            m_info = self._get_and_validate_model_info(m)
            return [np.array(d) for d in m_info.get("delta", [np.zeros_like(p) for p in m.get_parameters()])]

        self_delta = get_delta(self_model)

        # --- Initial Round (Round 0) ---
        if not self.global_model_params:
            self.global_model_params = [p.copy() for p in self_model.get_parameters()]
            self.prev_local_gradient = [d.copy() for d in self_delta]
            self_model.gradients_estimate = [d.copy() for d in self_delta]

        # --- 3. Calculate Metropolis-Hastings Weights ---
        my_info = self._get_and_validate_model_info(self_model)
        my_degree = int(my_info.get("degrees", len(model_map) - 1))
        
        neighbor_weights = {}
        for addr, model in model_map.items():
            if addr == self.addr:
                continue
            neighbor_info = self._get_and_validate_model_info(model)
            neighbor_degree = int(neighbor_info.get("degrees", len(model_map) - 1))
            neighbor_weights[addr] = 1.0 / (1.0 + max(my_degree, neighbor_degree))
        
        metro_weights = neighbor_weights
        metro_weights[self.addr] = 1.0 - sum(neighbor_weights.values())

        # --- 4. Update Tracking Variable V (tracks Average Delta) ---
        weighted_v_consensus = [np.zeros_like(p) for p in self.global_model_params]
        for addr, m in model_map.items():
            w_ij = metro_weights.get(addr, 0.0)
            v_j_prev = m.gradients_estimate if hasattr(m, 'gradients_estimate') and m.gradients_estimate else self_delta
            weighted_v_consensus = [acc + w_ij * np.array(v) for acc, v in zip(weighted_v_consensus, v_j_prev)]

        tracking_delta = [wv + curr - prev for wv, curr, prev in zip(weighted_v_consensus, self_delta, self.prev_local_gradient)]
        self.prev_local_gradient = [d.copy() for d in self_delta]

        # --- 5. Calculate Adaptive FedAdp Scores (using Tracking Delta) ---
        fedadp_scores = {}
        g_vec = np.concatenate([p.ravel() for p in tracking_delta])
        g_norm = np.linalg.norm(g_vec)

        for addr, m in model_map.items():
            m_delta = get_delta(m)
            l_vec = np.concatenate([p.ravel() for p in m_delta])
            l_norm = np.linalg.norm(l_vec)

            cos_sim = 1.0 if g_norm == 0 or l_norm == 0 else np.clip(np.dot(g_vec, l_vec) / (g_norm * l_norm), -1.0, 1.0)
            angle = float(np.arccos(cos_sim))

            prev_angle = self.node_correlation.get(addr, 0.0)
            smoothed_angle = angle if current_round <= 1 or prev_angle == 0.0 else ((current_round - 1)/current_round)*prev_angle + (1/current_round)*angle
            self.node_correlation[addr] = smoothed_angle
            
            f_val = self._gompertz_function(smoothed_angle)
            fedadp_scores[addr] = m.get_num_samples() * math.exp(f_val)

        # --- 6. Final Update: Weight Consensus using Adaptive Scores ---
        total_score = sum(fedadp_scores.values())
        psi = {addr: s / total_score if total_score > 0 else 1.0/len(model_map) for addr, s in fedadp_scores.items()}
        
        unnormalized_mix = {addr: psi[addr] * metro_weights[addr] for addr in model_map}
        sum_mix = sum(unnormalized_mix.values())
        final_mixing_weights = {addr: u / sum_mix if sum_mix > 0 else 1.0/len(model_map) for addr, u in unnormalized_mix.items()}
        
        new_weights = [np.zeros_like(p, dtype=np.float64) for p in self.global_model_params]
        for addr, m in model_map.items():
            w = final_mixing_weights.get(addr, 0.0)
            for i, layer in enumerate(m.get_parameters()):
                new_weights[i] += layer * w

        self.global_model_params = new_weights

        # --- Build and return result ---
        result_model = self_model.build_copy(params=self.global_model_params, num_samples=total_samples, contributors=contributors)
        result_model.gradients_estimate = tracking_delta
        return result_model

    def _gompertz_function(self, angle: float):
        # Non-linear Gompertz mapping function
        return self.ALPHA * (1 - math.exp(-math.exp(-self.ALPHA * (angle - 1))))
    
    def _get_and_validate_model_info(self, model: P2PFLModel) -> dict[str, Any]:
        try:
            info = model.get_info("gradient_delta_calculator")
        except KeyError:
            info = model.get_info()
        
        if info is None:
            return {}
        return info

    def get_required_callbacks(self) -> list[str]:
        return ["gradient_delta_calculator"]