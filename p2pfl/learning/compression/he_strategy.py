"""
Homomorphic Encryption Strategy for secure aggregation in decentralized federated learning.
Implements partial encryption of top-L coordinates using Paillier cryptosystem.
"""
import numpy as np
from abc import abstractmethod
from p2pfl.learning.compression.base_compression_strategy import TensorCompressor
from typing import List, Dict, Any, Tuple
import pickle

try:
    import phe  # Python Paillier Cryptosystem
    import phe.paillier as paillier
except ImportError as err:
    raise ImportError("Please install with `pip install p2pfl[he]`") from err


class HomomorphicEncryptionCompressor(TensorCompressor):
    """
    Implements homomorphic encryption for secure aggregation.
    Uses Paillier cryptosystem to encrypt only top-L coordinates,
    leaving others in plaintext but protected by DP.
    """
    
    def __init__(self, num_parties: int = 2, threshold: int = None, top_l_ratio: float = 0.5):
        """
        Initialize HE compressor.
        
        Args:
            num_parties: Number of parties in the system
            threshold: Threshold for decryption (default: num_parties//2 + 1)
            top_l_ratio: Ratio of coordinates to encrypt (L/m where m is sketch dimension)
        """
        self.num_parties = num_parties
        self.threshold = threshold or (num_parties // 2 + 1)
        self.top_l_ratio = top_l_ratio
        self.public_key, self.private_key = paillier.generate_paillier_keypair()
        self.is_aggregator = False  # Whether this node is responsible for aggregation
        
    def set_as_aggregator(self, is_aggregator: bool):
        """Set whether this instance acts as an aggregator."""
        self.is_aggregator = is_aggregator
        
    def apply_strategy(self, params: List[np.ndarray], **kwargs) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        """
        Apply HE to top-L coordinates of parameters.
        
        Args:
            params: List of parameter arrays to encrypt
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (encrypted_params, additional_info)
        """
        # Flatten all parameters to work with indices
        flattened_params = [param.flatten() for param in params]
        
        # Concatenate all parameters to find top-L across all layers
        all_params = np.concatenate(flattened_params)
        
        # Find top-L indices based on magnitude
        abs_params = np.abs(all_params)
        top_l_count = max(1, int(len(all_params) * self.top_l_ratio))
        top_l_indices = np.argpartition(abs_params, -top_l_count)[-top_l_count:]
        
        # Encrypt top-L coordinates
        encrypted_values = []
        plaintext_values = []
        plaintext_indices = []
        
        for idx in range(len(all_params)):
            if idx in top_l_indices:
                # Encrypt this value
                encrypted_val = self.public_key.encrypt(float(all_params[idx]))
                encrypted_values.append(encrypted_val)
            else:
                # Keep in plaintext
                plaintext_values.append(float(all_params[idx]))
                plaintext_indices.append(idx)
        
        # Store mapping to reconstruct the original structure
        param_shapes = [p.shape for p in params]
        param_sizes = [p.size for p in params]
        
        additional_info = {
            'param_shapes': param_shapes,
            'param_sizes': param_sizes,
            'top_l_indices': top_l_indices,
            'plaintext_indices': plaintext_indices,
            'encrypted_values': encrypted_values,
            'plaintext_values': plaintext_values,
            'public_key': self.public_key,
            'threshold': self.threshold
        }
        
        # Return empty arrays as we handle encryption separately
        return [np.array([]) for _ in params], additional_info
    
    def reverse_strategy(self, params: List[np.ndarray], additional_info: Dict[str, Any]) -> List[np.ndarray]:
        """
        Reverse HE operation - decrypt encrypted values and reconstruct parameters.
        
        Args:
            params: Placeholder (not used)
            additional_info: Additional information from apply_strategy
            
        Returns:
            Decrypted and reconstructed parameter arrays
        """
        if self.is_aggregator:
            # Perform threshold decryption of aggregated encrypted values
            decrypted_values = self._threshold_decrypt_aggregated_values(
                additional_info['encrypted_values'],
                additional_info['threshold']
            )
        else:
            # For non-aggregators, just return the values as they were
            decrypted_values = [enc_val for enc_val in additional_info['encrypted_values']]
        
        # Reconstruct the full parameter array
        all_params = np.zeros(sum(additional_info['param_sizes']))
        
        # Place decrypted values back in their positions
        top_l_indices = additional_info['top_l_indices']
        for i, idx in enumerate(top_l_indices):
            all_params[idx] = decrypted_values[i]
        
        # Place plaintext values back in their positions
        plaintext_indices = additional_info['plaintext_indices']
        plaintext_values = additional_info['plaintext_values']
        for i, idx in enumerate(plaintext_indices):
            all_params[idx] = plaintext_values[i]
        
        # Reshape back to original parameter shapes
        reconstructed_params = []
        start_idx = 0
        for shape in additional_info['param_shapes']:
            size = np.prod(shape)
            param = all_params[start_idx:start_idx+size].reshape(shape)
            reconstructed_params.append(param)
            start_idx += size
            
        return reconstructed_params
    
    def _threshold_decrypt_aggregated_values(self, encrypted_values, threshold):
        """
        Perform threshold decryption of aggregated encrypted values.
        In a real implementation, this would involve interaction with other parties.
        """
        # For simulation purposes, we'll decrypt directly
        # In a real implementation, this would require threshold decryption protocol
        decrypted_values = []
        for enc_val in encrypted_values:
            try:
                if isinstance(enc_val, paillier.EncryptedNumber):
                    # Decrypt using private key
                    decrypted_val = self.private_key.decrypt(enc_val)
                else:
                    decrypted_val = enc_val  # Already decrypted or plaintext
                decrypted_values.append(decrypted_val)
            except OverflowError:
                # Handle overflow - return 0 for aggregated values that overflow
                decrypted_values.append(0.0)
        
        return decrypted_values
    
    def aggregate_encrypted_values(self, encrypted_lists: List[List], weights: List[float] = None):
        """
        Perform homomorphic addition of encrypted values from multiple parties.
        This is the core of secure aggregation.
        
        Args:
            encrypted_lists: List of encrypted value lists from different parties
            weights: Optional weights for aggregation
            
        Returns:
            Aggregated encrypted values
        """
        if not encrypted_lists:
            return []
        
        # Initialize with first party's encrypted values
        aggregated = [enc_val for enc_val in encrypted_lists[0]]
        
        # Add encrypted values from other parties
        for party_idx in range(1, len(encrypted_lists)):
            party_encrypted = encrypted_lists[party_idx]
            for i in range(len(aggregated)):
                if i < len(party_encrypted):
                    # Homomorphic addition: E(a) * E(b) = E(a + b)
                    aggregated[i] = aggregated[i].ciphertext() + party_encrypted[i].ciphertext()
                    # Re-wrap with public key
                    aggregated[i] = paillier.EncryptedNumber(self.public_key, aggregated[i])
        
        return aggregated