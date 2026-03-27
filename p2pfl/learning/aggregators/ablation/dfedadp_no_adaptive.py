"""
Ablation of DFedAdp: No Adaptive Weighting.

This file implements an ablation of the DFedAdp aggregator where the adaptive
weighting (FedAdp) is disabled. The aggregation will only use the
Metropolis-Hastings weights.
"""
from typing import Dict
from p2pfl.learning.aggregators.dfedadp import DFedAdp
from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel


class DFedAdp_no_adaptive(DFedAdp):
    """
    Ablation of DFedAdp: No Adaptive Weighting.
    """
    def _get_final_mixing_weights(self, model_map: Dict[str, P2PFLModel], fedadp_scores: Dict[str, float], metro_weights: Dict[str, float]) -> Dict[str, float]:
        """
        Override to return only metro_weights, effectively disabling the adaptive FedAdp scores.
        """
        return metro_weights
