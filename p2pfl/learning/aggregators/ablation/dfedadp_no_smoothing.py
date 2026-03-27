"""
Ablation of DFedAdp: No Angle Smoothing.

This file implements an ablation of the DFedAdp aggregator where the
angle smoothing is disabled. The raw angle is used for the score calculation.
"""
from typing import Dict, List
import numpy as np
import math
from p2pfl.learning.aggregators.dfedadp import DFedAdp
from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel


class DFedAdp_no_smoothing(DFedAdp):
    """
    Ablation of DFedAdp: No Angle Smoothing.
    """
    def _get_fedadp_scores(self, model_map: Dict[str, P2PFLModel], tracking_delta: List[np.ndarray], current_round: int) -> Dict[str, float]:
        """
        Calculate FedAdp scores using raw angles without smoothing.
        """
        fedadp_scores = {}
        g_vec = np.concatenate([p.ravel() for p in tracking_delta])
        g_norm = np.linalg.norm(g_vec)

        for addr, m in model_map.items():
            m_delta = self._get_delta(m)
            l_vec = np.concatenate([p.ravel() for p in m_delta])
            l_norm = np.linalg.norm(l_vec)

            cos_sim = 1.0 if g_norm == 0 or l_norm == 0 else np.clip(np.dot(g_vec, l_vec) / (g_norm * l_norm), -1.0, 1.0)
            angle = float(np.arccos(cos_sim))

            f_val = self._gompertz_function(angle)
            fedadp_scores[addr] = m.get_num_samples() * math.exp(f_val)
        return fedadp_scores
