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

"""Quantization utilities for decentralized learning."""

import numpy as np

class StochasticQuantizer:
    """
    Implements the stochastic quantization operator Q(x) from Equation 6 (arXiv:2104.11375v1).
    Compersses a vector into b bits with scaling factor based on its infinity norm.
    """
    
    @staticmethod
    def quantize(x: np.ndarray, bits: int) -> np.ndarray:
        """
        Quantizes the input array x into the specified number of bits.
        
        Args:
            x: The input array (numpy).
            bits: Number of bits for quantization. 
                  If bits >= 32, returns original (no quantization).
        
        Returns:
            The quantized array.
        """
        if bits >= 32:
            return x
            
        # Scaling factor s based on bits (Equation 6 uses s as levels)
        # For b bits (including sign), we have 2^(b-1) - 1 positive levels.
        s = (2 ** (bits - 1)) - 1
        if s <= 0:
            return np.zeros_like(x)

        # Norm infinity (max absolute value)
        norm_inf = np.max(np.abs(x))
        if norm_inf == 0:
            return x

        # Normalize
        abs_x = np.abs(x)
        normalized_x = (s * abs_x) / norm_inf
        
        # Stochastic rounding
        # floor(normalized_x) + Bernoulli(normalized_x - floor(normalized_x))
        floor_x = np.floor(normalized_x)
        prob = normalized_x - floor_x
        rounded_x = floor_x + (np.random.rand(*x.shape) < prob).astype(float)
        
        # Rescale back
        return np.sign(x) * (norm_inf / s) * rounded_x
