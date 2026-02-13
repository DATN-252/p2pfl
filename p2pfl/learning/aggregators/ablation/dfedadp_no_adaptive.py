"""
Ablation of DFedAdp: No Adaptive Weighting.

This file implements an ablation of the DFedAdp aggregator where the adaptive
weighting (FedAdp) is disabled. The aggregation will only use the
Metropolis-Hastings weights.
"""
import numpy as np
from p2pfl.learning.aggregators.dfedadp import DFedAdp
from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel
from p2pfl.management.logger import logger
from p2pfl.learning.aggregators.aggregator import NoModelsToAggregateError


class DFedAdp_no_adaptive(DFedAdp):

    def aggregate(self, models: list[P2PFLModel]) -> P2PFLModel:
        """
        Aggregate the models without adaptive weighting.

        Overrides the DFedAdp aggregate method to use only Metropolis-Hastings weights.
        """
        if not models:
            raise NoModelsToAggregateError(f"({self.addr}) No models to aggregate")

        # Basic setup from the parent class
        total_samples = sum([m.get_num_samples() for m in models])
        contributors = [c for m in models for c in m.get_contributors()]

        current_round = self.each_trained_round.get(self.addr, 0)
        self_model = models[0]

        if not self.global_model_params:
            self.global_model_params = [p.copy() for p in self_model.get_parameters()]

        # Calculate Metropolis-Hastings Weights
        degrees = [int(self._get_and_validate_model_info(m)["degrees"]) for m in models]
        weight_metro = [0.0] * len(degrees)
        my_degree = degrees[0]
        for i in range(1, len(degrees)):
            weight_metro[i] = 1.0 / (1 + max(my_degree, degrees[i]))
        weight_metro[0] = 1.0 - sum(weight_metro[1:])

        if self.log_dfedadp_params:
            logger.info(self.addr, f"DFedAdp (No Adaptive) Round {current_round}: Metro weights = {weight_metro}")
        
        # *** ABLATION: Use only Metropolis-Hastings weights ***
        final_mixing_weights = weight_metro

        # Aggregation Step (Consensus)
        w_half = [np.zeros_like(p, dtype=np.float64) for p in self.global_model_params]
        for idx, m in enumerate(models):
            w = final_mixing_weights[idx]
            for i, layer in enumerate(m.get_parameters()):
                w_half[i] += layer * w
        
        # The rest of the logic (gradient tracking and update) remains the same.
        self_delta = self._get_and_validate_model_info(self_model)["delta"]
        curr_local_gradient = [-d / self.learning_rate for d in self_delta]

        weighted_neighbor_tracking = [np.zeros_like(p) for p in self.global_model_params]
        for idx, m in enumerate(models):
            g_j_prev = getattr(m, 'gradients_estimate', [-x / self.learning_rate for x in self._get_and_validate_model_info(m)["delta"]])
            w_ij = weight_metro[idx]
            weighted_neighbor_tracking = [acc + w_ij * g for acc, g in zip(weighted_neighbor_tracking, g_j_prev)]

        if not self.prev_local_gradient:
            self.prev_local_gradient = [np.zeros_like(p) for p in curr_local_gradient]

        tracking_gradient = [wn + curr - prev for wn, curr, prev in zip(weighted_neighbor_tracking, curr_local_gradient, self.prev_local_gradient)]
        self.prev_local_gradient = [g.copy() for g in curr_local_gradient]
        
        clip_threshold = 5.0
        tracking_gradient = [np.clip(tg, -clip_threshold, clip_threshold) for tg in tracking_gradient]

        self.global_model_params = [wh - self.learning_rate * tg for wh, tg in zip(w_half, tracking_gradient)]
        
        result_model = models[0].build_copy(
            params=self.global_model_params,
            num_samples=total_samples,
            contributors=contributors
        )
        result_model.gradients_estimate = tracking_gradient

        self.learning_rate = max(self.learning_rate * self.decay_rate, self.min_learning_rate)
        
        return result_model
