"""
Ablation of DFedAdp: No Gompertz Function.

This file implements an ablation of the DFedAdp aggregator where the Gompertz
function is replaced by a linear function f_val = -angle.
"""
from p2pfl.learning.aggregators.dfedadp import DFedAdp


class DFedAdp_no_gompertz(DFedAdp):
    """
    Ablation of DFedAdp: No Gompertz Function.
    """
    def _gompertz_function(self, angle: float) -> float:
        """
        Overrides the Gompertz function with a linear function.
        """
        return -angle
