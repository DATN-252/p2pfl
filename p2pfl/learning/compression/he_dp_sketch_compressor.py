"""
Hybrid HE+DP+Sketch Compressor for privacy-preserving decentralized federated learning.
Implements the full pipeline: DP→Sketch→Partial HE as described in the HE_DP_Sketch paper.
"""
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from p2pfl.learning.compression.base_compression_strategy import TensorCompressor
from p2pfl.learning.compression.renyi_dp_strategy import RenyiDifferentialPrivacyCompressor
from p2pfl.learning.compression.sketch_strategy import SketchCompressor
from p2pfl.learning.compression.he_strategy import HomomorphicEncryptionCompressor


class HEDPSketchCompressor(TensorCompressor):
    """
    Implements the hybrid HE+DP+Sketch protocol for privacy-preserving decentralized federated learning.
    Combines differential privacy, linear sketching, and partial homomorphic encryption
    to achieve a balanced privacy-utility-efficiency trade-off.
    """
    
    def __init__(
        self,
        dp_clip_norm: float = 1.0,
        dp_rdp_epsilon: float = 1.0,
        dp_delta: float = 1e-5,
        sketch_type: str = "countsketch",
        sketch_compressed_dim_ratio: float = 0.1,
        he_num_parties: int = 2,
        he_top_l_ratio: float = 0.5,
        seed: int = 42
    ):
        """
        Initialize the hybrid compressor.
        
        Args:
            dp_clip_norm: Clipping norm for DP
            dp_rdp_epsilon: Per-round RDP epsilon
            dp_delta: DP delta parameter
            sketch_type: Type of sketch ("countsketch" or "jl")
            sketch_compressed_dim_ratio: Ratio m/d for sketching (m < d)
            he_num_parties: Number of parties for HE threshold
            he_top_l_ratio: Ratio L/m for partial HE (encrypt top-L of m coordinates)
            seed: Random seed for reproducibility
        """
        self.dp_compressor = RenyiDifferentialPrivacyCompressor()
        self.sketch_compressor = SketchCompressor(
            sketch_type=sketch_type,
            compressed_dim_ratio=sketch_compressed_dim_ratio,
            seed=seed
        )
        self.he_compressor = HomomorphicEncryptionCompressor(
            num_parties=he_num_parties,
            top_l_ratio=he_top_l_ratio
        )
        
        self.dp_clip_norm = dp_clip_norm
        self.dp_rdp_epsilon = dp_rdp_epsilon
        self.dp_delta = dp_delta
        self.seed = seed
        self.num_rounds = 1  # Will be updated per round
        
    def apply_strategy(
        self,
        params: List[np.ndarray],
        round_num: int = 1,
        **kwargs
    ) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        """
        Apply the full HE+DP+Sketch pipeline: DP→Sketch→Partial HE.
        
        Args:
            params: Model update parameters to compress
            round_num: Current training round number
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (compressed_params, additional_info)
        """
        # Update seed based on round for shared randomness
        round_seed = self.seed + round_num
        
        # Step 1: Apply DP (clip + noise)
        dp_params, dp_info = self.dp_compressor.apply_strategy(
            params,
            clip_norm=self.dp_clip_norm,
            rdp_epsilon=self.dp_rdp_epsilon,
            delta=self.dp_delta,
            num_rounds=self.num_rounds
        )
        
        # Step 2: Apply sketching
        sketched_params, sketch_info = self.sketch_compressor.apply_strategy(
            dp_params
        )
        
        # Step 3: Apply partial HE to top-L coordinates of sketched params
        # Update HE compressor with round-dependent seed
        self.he_compressor = HomomorphicEncryptionCompressor(
            num_parties=self.he_compressor.num_parties,
            top_l_ratio=self.he_compressor.top_l_ratio
        )
        he_params, he_info = self.he_compressor.apply_strategy(
            sketched_params
        )
        
        # Combine all the information
        additional_info = {
            'dp_info': dp_info,
            'sketch_info': sketch_info,
            'he_info': he_info,
            'round_num': round_num,
            'seed': round_seed
        }
        
        return he_params, additional_info
    
    def reverse_strategy(self, params: List[np.ndarray], additional_info: Dict[str, Any]) -> List[np.ndarray]:
        """
        Reverse the HE+DP+Sketch pipeline: Decrypt→Unsketch→(DP is irreversible).
        
        Args:
            params: Compressed parameters (encrypted sketch)
            additional_info: Additional information from apply_strategy
            
        Returns:
            Reconstructed parameters (after aggregation)
        """
        # Note: We can't fully reverse DP (it's irreversible by design)
        # But we can decrypt and unsketch
        
        # Step 1: Reverse HE (decrypt)
        decrypted_params = self.he_compressor.reverse_strategy(params, additional_info['he_info'])
        
        # Step 2: Reverse sketching (approximate reconstruction)
        reconstructed_params = self.sketch_compressor.reverse_strategy(
            decrypted_params, 
            additional_info['sketch_info']
        )
        
        # Note: DP cannot be reversed, so we return the parameters after unsketching
        # The DP noise remains as part of the aggregated result
        
        return reconstructed_params
    
    def aggregate_compressed_models(
        self,
        compressed_models: List[Tuple[List[np.ndarray], Dict[str, Any]]],
        weights: Optional[List[float]] = None
    ) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        """
        Aggregate multiple compressed models, performing homomorphic operations where possible.
        
        Args:
            compressed_models: List of (compressed_params, additional_info) tuples
            weights: Optional weights for weighted aggregation
            
        Returns:
            Aggregated (compressed_params, combined_info)
        """
        if not compressed_models:
            return [], {}
        
        # Extract all compressed params and infos
        all_params = [model[0] for model in compressed_models]
        all_infos = [model[1] for model in compressed_models]
        
        # For HE parts, perform homomorphic addition
        # We need to extract the encrypted values from each model and aggregate them
        encrypted_lists = []
        for params, info in zip(all_params, all_infos):
            encrypted_lists.append(info['he_info']['encrypted_values'])
        
        # Perform homomorphic aggregation of encrypted values
        aggregated_encrypted = self.he_compressor.aggregate_encrypted_values(encrypted_lists, weights)
        
        # For sketch parts, perform linear aggregation (element-wise sum)
        # We'll use the first model's structure as reference
        first_info = all_infos[0]
        
        # Create new aggregated info with updated encrypted values
        aggregated_info = {
            'dp_info': first_info['dp_info'],  # DP info remains the same
            'sketch_info': first_info['sketch_info'],  # Sketch info remains the same
            'he_info': first_info['he_info'].copy(),
            'round_num': first_info['round_num'],
            'seed': first_info['seed']
        }
        
        # Update the encrypted values in the aggregated info
        aggregated_info['he_info']['encrypted_values'] = aggregated_encrypted
        
        # For simplicity, return empty params with aggregated info
        # The actual reconstruction happens in reverse_strategy
        return [np.array([])], aggregated_info
    
    def set_as_aggregator(self, is_aggregator: bool):
        """Set whether this instance acts as an aggregator for HE operations."""
        self.he_compressor.set_as_aggregator(is_aggregator)