"""
Renyi Differential Privacy Strategy for enhanced privacy accounting in federated learning.
Implements R ́enyi DP with subsampled-RDP for tighter privacy bounds.
"""
import numpy as np
from typing import List, Dict, Any, Tuple
import math
from p2pfl.learning.compression.dp_strategy import DifferentialPrivacyCompressor


class RenyiDifferentialPrivacyCompressor(DifferentialPrivacyCompressor):
    """
    Implements Renyi Differential Privacy for tighter privacy accounting.
    Uses RDP instead of basic (ε,δ)-DP for improved privacy bounds in iterative algorithms.
    """
    
    def __init__(self):
        super().__init__()
        
    def apply_strategy(
        self,
        params: List[np.ndarray],
        clip_norm: float = 1.0,
        rdp_epsilon: float = 1.0,  # Per-round RDP epsilon
        delta: float = 1e-5,
        noise_type: str = "gaussian",
        stability_constant: float = 1e-6,
        num_rounds: int = 1,
        subsampling_rate: float = 1.0  # For subsampled-RDP
    ) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        """
        Apply Renyi differential privacy with improved accounting.
        
        Args:
            params: Model update (delta) after local training
            clip_norm: Maximum L2 norm for clipping (C)
            rdp_epsilon: Per-round Renyi DP epsilon
            delta: The privacy budget (δ) for final (ε,δ)-DP conversion
            noise_type: Type of noise to add ("gaussian" or "laplace")
            stability_constant: A small constant to avoid division by zero when clipping
            num_rounds: Number of training rounds for privacy accounting
            subsampling_rate: Rate of subsampling (γ) for subsampled-RDP
            
        Returns:
            Tuple of (dp_params, dp_info) where dp_info contains privacy parameters
        """
        # Calculate noise scale based on RDP parameters
        if noise_type == "gaussian":
            # For Gaussian mechanism, per-round RDP alpha -> noise scale
            # In RDP, for Gaussian mechanism: ε_α = α * σ^2 / 2
            # So σ^2 = 2 * ε_α / α
            # We'll use α=2 as a common choice, so σ^2 = rdp_epsilon
            alpha = 2.0  # Common choice for RDP analysis
            noise_scale = math.sqrt(2.0 * rdp_epsilon / alpha) * clip_norm
        else:
            # For Laplace, RDP is less commonly used, default to standard approach
            noise_scale = clip_norm / rdp_epsilon

        # Handle empty input
        if not params:
            raise ValueError("RenyiDifferentialPrivacyCompressor: list 'params' must not be empty")

        # Step 1: Compute global L2 norm across all parameters
        flat_update = np.concatenate([p.flatten() for p in params])
        total_norm = np.linalg.norm(flat_update)

        # Step 2: Clip if necessary
        if total_norm > clip_norm:
            clip_factor = clip_norm / (total_norm + stability_constant)
            clipped_flat_update = flat_update * clip_factor
        else:
            clipped_flat_update = flat_update.copy()

        # Step 3: Add noise based on calculated scale
        if noise_type == "gaussian":
            noise = np.random.normal(0, noise_scale, size=clipped_flat_update.shape)
        else:  # laplace
            noise = np.random.laplace(0, noise_scale, size=clipped_flat_update.shape)
        
        noisy_flat_update = clipped_flat_update + noise

        # Unflatten the noisy update
        dp_params = []
        current_pos = 0
        for p in params:
            shape = p.shape
            size = p.size
            dtype = p.dtype
            dp_params.append(
                np.array(noisy_flat_update[current_pos : current_pos + size], dtype=dtype).reshape(shape)
            )
            current_pos += size

        # Calculate cumulative privacy using RDP composition
        cumulative_rdp_epsilon = self._compute_rdp_composition(
            rdp_epsilon, num_rounds, subsampling_rate
        )
        
        # Convert RDP to (ε,δ)-DP
        final_epsilon, final_delta = self._convert_rdp_to_approx_dp(
            cumulative_rdp_epsilon, delta, alpha
        )

        # Prepare info for privacy accounting
        dp_info = {
            "dp_applied": True,
            "clip_norm": clip_norm,
            "rdp_epsilon": rdp_epsilon,  # Per-round RDP epsilon
            "epsilon": final_epsilon, # Final (ε,δ)-DP epsilon after composition
            "delta": final_delta,
            "noise_type": noise_type,
            "noise_scale": noise_scale,
            "original_norm": float(total_norm),
            "was_clipped": bool(total_norm > clip_norm),
            "num_rounds": num_rounds,
            "subsampling_rate": subsampling_rate,
            "cumulative_rdp_epsilon": cumulative_rdp_epsilon,
            "rdp_alpha": alpha
        }

        return dp_params, dp_info
    
    def _compute_rdp_composition(self, rdp_epsilon: float, num_rounds: int, subsampling_rate: float = 1.0) -> float:
        """
        Compute cumulative RDP epsilon after multiple rounds of composition.
        If subsampling is used, applies subsampled-RDP amplification.
        
        Args:
            rdp_epsilon: Per-round RDP epsilon
            num_rounds: Number of rounds
            subsampling_rate: Rate of subsampling (γ), default 1.0 (no subsampling)
            
        Returns:
            Cumulative RDP epsilon
        """
        if subsampling_rate < 1.0:
            # Apply subsampled-RDP amplification
            # For subsampled Gaussian mechanism, the amplified RDP is approximately:
            # ε'_α ≈ γ² * α * ε_α where γ is subsampling rate
            amplified_epsilon = subsampling_rate * subsampling_rate * rdp_epsilon
            return amplified_epsilon * num_rounds
        else:
            # Standard composition: multiply by number of rounds
            return rdp_epsilon * num_rounds
    
    def _convert_rdp_to_approx_dp(self, rdp_epsilon: float, delta: float, alpha: float) -> Tuple[float, float]:
        """
        Convert RDP guarantee to (ε,δ)-DP using standard conversion.
        
        Args:
            rdp_epsilon: RDP epsilon value
            delta: Target δ for (ε,δ)-DP
            alpha: RDP order parameter
            
        Returns:
            Tuple of (ε, δ) for (ε,δ)-DP
        """
        # Standard conversion from RDP to (ε,δ)-DP:
        # For any α > 1, if M is (α, ε_α)-RDP, then M is (ε, δ)-DP with:
        # ε = ε_α - (ln(1/δ) / (α - 1))
        
        # To optimize over α, we can use a fixed α or optimize
        # Here we use the provided α for the conversion
        epsilon = rdp_epsilon - (math.log(1.0 / delta) / (alpha - 1.0))
        
        # Ensure epsilon is non-negative
        epsilon = max(0, epsilon)
        
        return epsilon, delta

    def compute_total_privacy_loss(
        self, 
        per_round_rdp_epsilon: float, 
        num_rounds: int, 
        delta: float,
        subsampling_rate: float = 1.0
    ) -> Tuple[float, float]:
        """
        Compute total privacy loss over multiple rounds.
        
        Args:
            per_round_rdp_epsilon: Per-round RDP epsilon
            num_rounds: Number of training rounds
            delta: Target δ for (ε,δ)-DP
            subsampling_rate: Rate of subsampling (γ)
            
        Returns:
            Tuple of (total_epsilon, total_delta) for (ε,δ)-DP
        """
        cumulative_rdp_epsilon = self._compute_rdp_composition(
            per_round_rdp_epsilon, num_rounds, subsampling_rate
        )
        
        # Optimize over multiple alpha values to find the best (ε,δ)-DP bound
        best_epsilon = float('inf')
        alphas = [1 + x * 0.1 for x in range(1, 100)]  # α > 1
        
        for alpha in alphas:
            rdp_value = cumulative_rdp_epsilon * alpha  # Simplified for demonstration
            epsilon = rdp_value - (math.log(1.0 / delta) / (alpha - 1.0))
            if epsilon < best_epsilon:
                best_epsilon = epsilon
        
        return best_epsilon, delta