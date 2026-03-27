"""
Ablation of DFedAdp: No Angle-based Score.

This file implements an ablation of the DFedAdp aggregator where the
angle-based score is replaced by a score based only on the number of samples.
"""
from typing import Dict, List
import numpy as np
from p2pfl.learning.aggregators.dfedadp import DFedAdp
from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel


class DFedAdp_no_angle(DFedAdp):
    """
    Ablation of DFedAdp: No Angle-based Score.
    """
    def _get_fedadp_scores(self, model_map: Dict[str, P2PFLModel], tracking_delta: List[np.ndarray], current_round: int) -> Dict[str, float]:
        """
        Replace angle-based scores with scores based only on the number of samples.
        """
        return {addr: m.get_num_samples() for addr, m in model_map.items()}
