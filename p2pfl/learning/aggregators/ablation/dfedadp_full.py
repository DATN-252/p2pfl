"""
Full DFedAdp Aggregator (Baseline).

This file serves as a baseline for the ablation studies. It uses the DFedAdp
aggregator as-is, without any modifications.
"""
from p2pfl.learning.aggregators.dfedadp import DFedAdp

# This is the baseline implementation, which is just a re-export of DFedAdp.
DFedAdp_full = DFedAdp
