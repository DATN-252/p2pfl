#
# Centralized FedAdp Aggregator.
#
import math
import numpy as np
from collections import defaultdict
from typing import Any, List, Dict

from p2pfl.learning.aggregators.aggregator import Aggregator, NoModelsToAggregateError
from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel
from p2pfl.management.logger import logger

class FedAdp(Aggregator):
    """
    Centralized Federated Adaptive Weighting (FedAdp) Aggregator.
    Adapts client weights based on the contribution of their updates to the global direction.
    """
    
    def __init__(
        self, 
        alpha: float = 1.0, 
        **kwargs
    ) -> None:
        super().__init__(disable_partial_aggregation=True)
        self.global_model_params: List[np.ndarray] = []
        self.node_correlation: Dict[str, float] = defaultdict(lambda: 0.0)
        self.ALPHA = alpha

    def aggregate(self, models: List[P2PFLModel]) -> P2PFLModel:
        if len(models) == 0:
            raise NoModelsToAggregateError(f"({self.addr}) No models to aggregate")

        # 1. Setup and basic FedAvg reference
        total_samples = sum(m.get_num_samples() for m in models)
        current_round = self.each_trained_round.get(self.addr, 0)
        
        # Get current global parameters (reference from previous round)
        # If not exists, use the first model as base
        if not self.global_model_params:
            self.global_model_params = [p.copy() for p in models[0].get_parameters()]

        # 2. Calculate Local Deltas (Updates)
        local_updates = []
        for m in models:
            delta = [p_global - p_local for p_global, p_local in zip(self.global_model_params, m.get_parameters())]
            local_updates.append(delta)

        # 3. Calculate Global Reference Update (Simple average of updates)
        # This acts as the "Global Gradient" direction
        global_delta = [np.zeros_like(p) for p in self.global_model_params]
        for i, m in enumerate(models):
            weight = m.get_num_samples() / total_samples
            for j, layer_delta in enumerate(local_updates[i]):
                global_delta[j] += weight * layer_delta

        # Flatten global delta for similarity calculation
        g_vec = np.concatenate([p.ravel() for p in global_delta])
        g_norm = np.linalg.norm(g_vec)

        # 4. Calculate Adaptive Scores
        fedadp_scores = {}
        for i, m in enumerate(models):
            addr = m.get_contributors()[0] if m.get_contributors() else f"node_{i}"
            
            # Flatten local update
            l_vec = np.concatenate([p.ravel() for p in local_updates[i]])
            l_norm = np.linalg.norm(l_vec)

            # Cosine Similarity
            cos_sim = 1.0 if g_norm == 0 or l_norm == 0 else np.clip(np.dot(g_vec, l_vec) / (g_norm * l_norm), -1.0, 1.0)
            angle = float(np.arccos(cos_sim))

            # Smoothing angle across rounds
            prev_angle = self.node_correlation.get(addr, 0.0)
            smoothed_angle = angle if current_round <= 1 or prev_angle == 0.0 else 0.9 * prev_angle + 0.1 * angle
            self.node_correlation[addr] = smoothed_angle
            
            # Gompertz amplification
            f_val = self._gompertz_function(smoothed_angle)
            fedadp_scores[addr] = m.get_num_samples() * math.exp(f_val)

        # 5. Final Aggregation using Adaptive Scores
        total_score = sum(fedadp_scores.values())
        final_weights = {addr: s / total_score if total_score > 0 else 1.0/len(models) for addr, s in fedadp_scores.items()}
        
        new_params = [np.zeros_like(p, dtype=np.float64) for p in self.global_model_params]
        contributors = []
        
        for i, m in enumerate(models):
            addr = m.get_contributors()[0] if m.get_contributors() else f"node_{i}"
            w = final_weights.get(addr, 0.0)
            contributors.extend(m.get_contributors())
            
            for j, layer in enumerate(m.get_parameters()):
                new_params[j] += layer * w

        self.global_model_params = [p.astype(np.float32) for p in new_params]

        # 6. Return Result
        return models[0].build_copy(
            params=self.global_model_params, 
            num_samples=total_samples, 
            contributors=list(set(contributors))
        )

    def _gompertz_function(self, angle: float) -> float:
        # Standard Gompertz function: f(x) = a * exp(-b * exp(-cx))
        # Here simplified/adapted for FedAdp logic
        return self.ALPHA * (1 - math.exp(-math.exp(-self.ALPHA * (angle - 1))))
