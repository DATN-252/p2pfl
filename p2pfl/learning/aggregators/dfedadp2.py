import numpy as np
from collections import defaultdict
from typing import Any, List, Dict
from p2pfl.learning.aggregators.aggregator import Aggregator, NoModelsToAggregateError
from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel
from p2pfl.management.logger import logger
import math

class DFedAdp_2(Aggregator):
    SUPPORTS_PARTIAL_AGGREGATION: bool = False
    requires_gradient_only: bool = False
    REQUIRED_INFO_KEYS = ["delta", "degrees"] 

    def __init__(self, disable_partial_aggregation: bool = False, base_learning_rate: float = 0.001, 
                 min_learning_rate: float = 0.0001, B: float = 2.0, beta: float = 1.0) -> None:
        super().__init__(disable_partial_aggregation=disable_partial_aggregation)
        self.global_model_params: List[np.ndarray] = []
        
        # Tracks smoothed cosine similarity instead of angles
        self.node_correlation: Dict[str, float] = {}
        
        # State-space dynamic bounds
        self.base_learning_rate = base_learning_rate
        self.current_learning_rate = base_learning_rate
        self.min_learning_rate = min_learning_rate
        self.B = B # Decreased default B for better neighborhood inclusion
        self.beta = beta
        
        self.prev_local_gradient: List[np.ndarray] = []

    def aggregate(self, models: List[P2PFLModel]) -> P2PFLModel:
        if not models:
            raise NoModelsToAggregateError(f"({self.addr}) No models to aggregate")

        # --- 1. Address Normalization & Mapping ---
        norm_self_addr = self.normalize_addr(self.addr)
        model_map = {self.normalize_addr(m.get_contributors()[0]): m for m in models if m.get_contributors()}
        
        self_model = model_map.get(norm_self_addr)
        if not self_model:
            raise NoModelsToAggregateError(f"Self model ({norm_self_addr}) missing. Available: {list(model_map.keys())}")
        
        total_samples = sum(m.get_num_samples() for m in model_map.values())
        current_round = self.each_trained_round.get(self.addr, 0)

        def get_delta(m):
            m_info = self._get_and_validate_model_info(m)
            delta_raw = m_info.get("delta")
            if delta_raw is None:
                return [np.zeros_like(p) for p in m.get_parameters()]
            return [np.array(d) for d in delta_raw]

        self_delta = get_delta(self_model)

        # --- Initial Round Setup ---
        if not self.global_model_params:
            self.global_model_params = [p.copy() for p in self_model.get_parameters()]
            self.prev_local_gradient = [d.copy() for d in self_delta]

        # --- 2. Metropolis-Hastings Weights ---
        my_degree = int(self._get_and_validate_model_info(self_model).get("degrees", len(model_map) - 1))
        neighbor_weights = {
            addr: 1.0 / (1.0 + max(my_degree, int(self._get_and_validate_model_info(m).get("degrees", len(model_map) - 1))))
            for addr, m in model_map.items() if addr != norm_self_addr
        }
        metro_weights = {**neighbor_weights, norm_self_addr: 1.0 - sum(neighbor_weights.values())}

        # --- 3. Global Gradient Tracking (Consensus Vector) ---
        weighted_v_consensus = [np.zeros_like(p) for p in self.global_model_params]
        
        # Shape safety check
        if len(self_delta) != len(weighted_v_consensus) or any(s.shape != g.shape for s, g in zip(self_delta, weighted_v_consensus)):
            self_delta = [np.zeros_like(p) for p in self.global_model_params]
            self.prev_local_gradient = [np.zeros_like(p) for p in self.global_model_params]

        for addr, m in model_map.items():
            w_ij = metro_weights.get(addr, 0.0)
            v_j_prev = self._get_and_validate_model_info(m).get("tracking_v")
            
            if not v_j_prev or len(v_j_prev) != len(weighted_v_consensus) or any(np.array(v).shape != g.shape for v, g in zip(v_j_prev, weighted_v_consensus)):
                v_j_prev = [np.zeros_like(p) for p in self.global_model_params]
            
            weighted_v_consensus = [acc + w_ij * np.array(v) for acc, v in zip(weighted_v_consensus, v_j_prev)]

        tracking_delta = [wv + curr - prev for wv, curr, prev in zip(weighted_v_consensus, self_delta, self.prev_local_gradient)]
        self.prev_local_gradient = [d.copy() for d in self_delta]

        # --- 4. Adaptive Analytic Weighting & Zero-Trust Filter ---
        fedadp_scores = {}
        g_vec = np.concatenate([p.ravel() for p in tracking_delta])
        g_norm = np.linalg.norm(g_vec)
        
        # Soften the safety threshold impact
        safety_threshold = (self.B * self.beta * self.current_learning_rate) / 2.0
        avg_cos_sim = 0.0

        for addr, m in model_map.items():
            l_vec = np.concatenate([p.ravel() for p in get_delta(m)])
            l_norm = np.linalg.norm(l_vec)

            # Cosine similarity calculation
            cos_sim = 1.0 if g_norm == 0 or l_norm == 0 else np.clip(np.dot(g_vec, l_vec) / (g_norm * l_norm), -1.0, 1.0)
            
            prev_cos = self.node_correlation.get(addr, cos_sim)
            smoothed_cos = cos_sim if current_round <= 1 else 0.9 * prev_cos + 0.1 * cos_sim
            self.node_correlation[addr] = smoothed_cos
            avg_cos_sim += smoothed_cos
            
            # Use Exponential mapping for stability (similar to original but with cos_sim)
            # This ensures no node gets exactly 0 weight, preserving network connectivity
            score = math.exp(self.beta * (smoothed_cos - safety_threshold))
            fedadp_scores[addr] = m.get_num_samples() * score

        # --- 5. Dynamic Learning Rate Control ---
        avg_cos_sim /= len(model_map)
        # Improved LR scaling: allow LR to stay closer to base_lr (min scale 0.5)
        # This prevents the "plateau" by ensuring steps are large enough
        lr_scale = max(0.5, min(1.2, (2.0 * max(0.1, avg_cos_sim)) / (self.B * self.beta + 1e-6)))
        target_lr = self.base_learning_rate * lr_scale

        # Faster momentum (0.6) to allow LR to recover when consensus improves
        self.current_learning_rate = 0.6 * self.current_learning_rate + 0.4 * target_lr
        self.current_learning_rate = max(self.min_learning_rate, self.current_learning_rate)

        # --- 6. Final Model Mixing ---
        total_score = sum(fedadp_scores.values())
        psi = {addr: s / total_score if total_score > 0 else 1.0/len(model_map) for addr, s in fedadp_scores.items()}

        # Adaptive Mixing Schedule:
        # Round 0-3: 0.2 (Warm-up, focus on MH)
        # Round 4-30: 0.5 (Hybrid)
        # Round > 30: 0.8 (Aggressive Adaptive - This is where we beat the original)
        if current_round <= 3:
            adaptive_weight = 0.2
        elif current_round <= 30:
            adaptive_weight = 0.5
        else:
            adaptive_weight = 0.8

        consensus_weight = 1.0 - adaptive_weight

        final_mixing_weights = {}
        for addr in model_map:
            final_mixing_weights[addr] = adaptive_weight * psi[addr] + consensus_weight * metro_weights[addr]

        
        # Re-normalize to ensure sum is 1.0
        total_final_w = sum(final_mixing_weights.values())
        final_mixing_weights = {addr: w / total_final_w for addr, w in final_mixing_weights.items()}
        
        new_weights = [np.zeros_like(p, dtype=np.float64) for p in self.global_model_params]
        for addr, m in model_map.items():
            w = final_mixing_weights.get(addr, 0.0)
            for i, layer in enumerate(m.get_parameters()):
                new_weights[i] += layer * w

        self.global_model_params = new_weights

        # --- Build and Return ---
        result_model = self_model.build_copy(params=self.global_model_params, num_samples=total_samples, contributors=list(model_map.keys()))
        result_model.add_info("gradient_delta_calculator", {
            "delta": self_delta,
            "tracking_v": [v.copy() for v in tracking_delta],
            "dynamic_lr": float(self.current_learning_rate)
        })
        return result_model

    def _get_and_validate_model_info(self, model: P2PFLModel) -> dict[str, Any]:
        try:
            info = model.get_info("gradient_delta_calculator")
        except KeyError:
            info = model.get_info()
        return info if info else {}

    def get_required_callbacks(self) -> list[str]:
        return ["gradient_delta_calculator"]