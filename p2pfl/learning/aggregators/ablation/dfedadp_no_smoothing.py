"""
Ablation of DFedAdp: No Angle Smoothing.

This file implements an ablation of the DFedAdp aggregator where the
angle smoothing is disabled. The raw angle is used for the score calculation.
"""
import numpy as np
import math
from p2pfl.learning.aggregators.dfedadp import DFedAdp
from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel
from p2pfl.management.logger import logger
from p2pfl.learning.aggregators.aggregator import NoModelsToAggregateError

class DFedAdp_no_smoothing(DFedAdp):

    def aggregate(self, models: list[P2PFLModel]) -> P2PFLModel:
        """
        Aggregate the models without angle smoothing.

        Overrides the DFedAdp aggregate method to use the raw angle for score
        calculation, ignoring the historical node_correlation.
        """
        if not models:
            raise NoModelsToAggregateError(f"({self.addr}) No models to aggregate")

        total_samples = sum([m.get_num_samples() for m in models])
        contributors = [c for m in models for c in m.get_contributors()]

        current_round = self.each_trained_round.get(self.addr, 0)
        self_model = models[0]

        if not self.global_model_params:
            self.global_model_params = [p.copy() for p in self_model.get_parameters()]

        degrees = [int(self._get_and_validate_model_info(m)["degrees"]) for m in models]
        weight_metro = [0.0] * len(degrees)
        my_degree = degrees[0]
        for i in range(1, len(degrees)):
            weight_metro[i] = 1.0 / (1 + max(my_degree, degrees[i]))
        weight_metro[0] = 1.0 - sum(weight_metro[1:])

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

        # Calculate FedAdp Scores (Contribution Measurement)
        fedadp_scores = []
        g_vec = np.concatenate([p.ravel() for p in tracking_gradient])
        g_norm = np.linalg.norm(g_vec)

        for idx, m in enumerate(models):
            m_delta = self._get_and_validate_model_info(m)["delta"]
            neigh_local_grad = [-d / self.learning_rate for d in m_delta]
            l_vec = np.concatenate([p.ravel() for p in neigh_local_grad])
            l_norm = np.linalg.norm(l_vec)

            if g_norm == 0 or l_norm == 0:
                cos_sim = 1.0
            else:
                cos_sim = np.dot(g_vec, l_vec) / (g_norm * l_norm)
                cos_sim = np.clip(cos_sim, -1.0, 1.0)
            angle = float(np.arccos(cos_sim))

            # *** ABLATION: Use raw angle, no smoothing ***
            f_val = self._gompertz_function(angle)
            score = m.get_num_samples() * math.exp(f_val)
            fedadp_scores.append(score)

            if self.log_dfedadp_params:
                logger.info(self.addr, f"DFedAdp (No Smoothing) Round {current_round}, Node {m.get_contributors()[0]}: Angle={angle:.4f}, Score={score:.4f}")

        total_score = sum(fedadp_scores)
        psi = [s / total_score if total_score > 0 else 1.0/len(models) for s in fedadp_scores]
        
        unnormalized_mix = [p * w for p, w in zip(psi, weight_metro)]
        sum_mix = sum(unnormalized_mix)
        final_mixing_weights = [u / sum_mix if sum_mix > 0 else 1.0/len(models) for u in unnormalized_mix]
        
        w_half = [np.zeros_like(p, dtype=np.float64) for p in self.global_model_params]
        for idx, m in enumerate(models):
            w = final_mixing_weights[idx]
            for i, layer in enumerate(m.get_parameters()):
                w_half[i] += layer * w
        
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
