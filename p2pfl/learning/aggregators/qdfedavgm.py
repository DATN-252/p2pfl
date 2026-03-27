#
# This file is part of the federated_learning_p2p (p2pfl) distribution
# (see https://github.com/pguijas/p2pfl).
# Copyright (c) 2024 Pedro Guijas Bravo.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#

"""Quantized Decentralized Federated Averaging with Momentum (Q-DFedAvgM) Aggregator."""

import numpy as np
from typing import List, Dict, Any, Optional

from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel
from p2pfl.learning.aggregators.aggregator import Aggregator, NoModelsToAggregateError
from p2pfl.management.logger import logger
from p2pfl.utils.quantization import StochasticQuantizer

class QDFedAvgMAggregator(Aggregator):
    """
    Implements Q-DFedAvgM (Algorithm 2) from arXiv:2104.11375v1.
    Equation 7: x^{t+1}(i) = x^t(i) + Σ w_il * q^t(l)
    where q^t(l) = Q(y^{t,K}(l) - x^t(l))
    """

    def __init__(self, bits: int = 8, **kwargs):
        """
        Initialize the Q-DFedAvgM Aggregator.
        Args:
            bits (int): Number of bits for quantization.
        """
        super().__init__(**kwargs)
        self.bits = bits
        self.x_state: List[np.ndarray] = []  # Current global state x^t
        self.is_initialized = False

    def preprocess_local_model(self, model: P2PFLModel) -> P2PFLModel:
        """
        Quantizes the local model difference before broadcasting.
        Calculates q^t(i) = Q(y^{t,K}(i) - x^t(i)).
        """
        params = model.get_parameters()
        
        # Initialization at Round 0
        if not self.is_initialized:
            self.x_state = [p.copy() for p in params]
            self.is_initialized = True
            # For round 0, diff is usually 0 if initialized together, 
            # but we send the initial state quantized for consistency if needed.
            # However, Algorithm 2 assumes nodes start with the same x^0.
            
        # q^t(i) = Q(y^{t,K}(i) - x^t(i))
        q_diff = []
        for p_y, p_x in zip(params, self.x_state):
            diff = p_y - p_x
            q_diff.append(StochasticQuantizer.quantize(diff, self.bits))
        
        # Build a model containing the quantized difference
        # We piggyback the 'q_diff' in the parameters for the P2P communication
        return model.build_copy(params=q_diff)

    def aggregate(self, models: List[P2PFLModel]) -> P2PFLModel:
        """
        Implements Equation 7 using Metropolis-Hastings weights.
        """
        if not models:
            raise NoModelsToAggregateError(f"({self.addr}) No models to aggregate")

        model_map = {m.get_contributors()[0]: m for m in models}
        self_model = model_map.get(self.addr)
        if self_model is None:
            raise NoModelsToAggregateError("Self model not found in aggregation.")

        # 1. Metropolis-Hastings Weights calculation
        # Each model should have its degree in additional_info
        my_degree = self_model.additional_info.get("degrees", len(models) - 1)
        
        weights = {}
        for addr, m in model_map.items():
            if addr == self.addr:
                continue
            neighbor_degree = m.additional_info.get("degrees", len(models) - 1)
            weights[addr] = 1.0 / (1.0 + max(my_degree, neighbor_degree))
        
        weights[self.addr] = 1.0 - sum(weights.values())

        # 2. Update State: x^{t+1} = x^t + Σ w_il * q^t(l)
        # Note: In our implementation, the received 'params' ARE already q^t(l)
        if not self.is_initialized or len(self.x_state) == 0:
            self.x_state = [p.copy() for p in self_model.get_parameters()]
            self.is_initialized = True

        sum_wq = [np.zeros_like(p) for p in self.x_state]

        sum_wq = [np.zeros_like(p) for p in self.x_state]
        for addr, m in model_map.items():
            w_il = weights.get(addr, 0.0)
            q_l = m.get_parameters()
            for i, p_q in enumerate(q_l):
                sum_wq[i] += w_il * p_q

        # Update x^{t+1}
        for i in range(len(self.x_state)):
            self.x_state[i] += sum_wq[i]

        # Return the new global model
        return self_model.build_copy(params=self.x_state)
