"""
Linear Sketch Strategy for dimensionality reduction in federated learning.
Implements CountSketch and Johnson-Lindenstrauss (JL) projections for compression.
"""
import numpy as np
from abc import abstractmethod
from p2pfl.learning.compression.base_compression_strategy import TensorCompressor
from typing import List, Dict, Any, Tuple
import hashlib
import random


class SketchCompressor(TensorCompressor):
    """
    Implements linear sketching for dimensionality reduction.
    Supports CountSketch and JL (Johnson-Lindenstrauss) projections.
    """
    
    def __init__(self, sketch_type: str = "countsketch", compressed_dim_ratio: float = 0.1, seed: int = 42):
        """
        Initialize sketch compressor.
        
        Args:
            sketch_type: Type of sketch ("countsketch" or "jl")
            compressed_dim_ratio: Ratio of compressed dimension to original (m/d)
            seed: Random seed for reproducibility
        """
        self.sketch_type = sketch_type.lower()
        self.compressed_dim_ratio = compressed_dim_ratio
        self.seed = seed
        self.hash_functions = None
        self.sign_functions = None
        self.projection_matrix = None
        
    def apply_strategy(self, params: List[np.ndarray], **kwargs) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        """
        Apply sketching to parameters.
        
        Args:
            params: List of parameter arrays to sketch
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (sketched_params, additional_info)
        """
        # Set seed for reproducibility
        np.random.seed(self.seed)
        random.seed(self.seed)
        
        # Flatten all parameters to work in a single vector
        original_shapes = [p.shape for p in params]
        original_sizes = [p.size for p in params]
        all_params = np.concatenate([p.flatten() for p in params])
        original_dim = len(all_params)
        
        # Calculate compressed dimension
        compressed_dim = max(1, int(original_dim * self.compressed_dim_ratio))
        
        if self.sketch_type == "countsketch":
            sketched_params, sketch_info = self._apply_countsketch(all_params, compressed_dim)
        elif self.sketch_type == "jl":
            sketched_params, sketch_info = self._apply_jl_projection(all_params, compressed_dim)
        else:
            raise ValueError(f"Unsupported sketch type: {self.sketch_type}")
        
        additional_info = {
            'original_shapes': original_shapes,
            'original_sizes': original_sizes,
            'original_dim': original_dim,
            'compressed_dim': compressed_dim,
            'sketch_type': self.sketch_type,
            'sketch_info': sketch_info,
            'seed': self.seed
        }
        
        # Return the sketched vector
        return [sketched_params], additional_info
    
    def reverse_strategy(self, params: List[np.ndarray], additional_info: Dict[str, Any]) -> List[np.ndarray]:
        """
        Reverse sketching operation - reconstruct parameters from sketch.
        Note: This is an approximation due to information loss in sketching.
        
        Args:
            params: Sketched parameter array
            additional_info: Additional information from apply_strategy
            
        Returns:
            Reconstructed parameter arrays (approximation)
        """
        sketched_vector = params[0]
        
        if self.sketch_type == "countsketch":
            reconstructed = self._reverse_countsketch(
                sketched_vector, 
                additional_info['original_dim'],
                additional_info['sketch_info']
            )
        elif self.sketch_type == "jl":
            reconstructed = self._reverse_jl_projection(
                sketched_vector,
                additional_info['original_dim'],
                additional_info['sketch_info']
            )
        else:
            raise ValueError(f"Unsupported sketch type: {self.sketch_type}")
        
        # Reshape back to original parameter shapes
        reconstructed_params = []
        start_idx = 0
        for shape in additional_info['original_shapes']:
            size = np.prod(shape)
            param = reconstructed[start_idx:start_idx+size].reshape(shape)
            reconstructed_params.append(param)
            start_idx += size
            
        return reconstructed_params
    
    def _apply_countsketch(self, vector: np.ndarray, compressed_dim: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Apply CountSketch to the input vector."""
        n = len(vector)
        
        # Generate hash functions and sign functions
        # For CountSketch, we need random hash functions h: [n] -> [m] and sign functions s: [n] -> {-1, +1}
        hash_functions = []
        sign_functions = []
        
        for i in range(n):
            # Hash function: map index i to [0, compressed_dim)
            hash_val = int(hashlib.md5(f"{self.seed}_{i}_hash".encode()).hexdigest(), 16) % compressed_dim
            hash_functions.append(hash_val)
            
            # Sign function: map index i to {-1, +1}
            sign_val = 1 if int(hashlib.md5(f"{self.seed}_{i}_sign".encode()).hexdigest(), 16) % 2 == 0 else -1
            sign_functions.append(sign_val)
        
        # Create the sketch
        sketch = np.zeros(compressed_dim)
        for i in range(n):
            sketch[hash_functions[i]] += vector[i] * sign_functions[i]
        
        sketch_info = {
            'hash_functions': hash_functions,
            'sign_functions': sign_functions
        }
        
        return sketch, sketch_info
    
    def _reverse_countsketch(self, sketch: np.ndarray, original_dim: int, sketch_info: Dict[str, Any]) -> np.ndarray:
        """Reverse CountSketch using median-of-means reconstruction."""
        # For CountSketch, exact reconstruction is not possible, but we can estimate
        # This is a simplified version - in practice, more sophisticated reconstruction methods are used
        reconstructed = np.zeros(original_dim)
        
        # Use the stored hash and sign functions to map back
        hash_functions = sketch_info['hash_functions']
        sign_functions = sketch_info['sign_functions']
        
        # For each original position, collect all contributions from the sketch
        for i in range(min(original_dim, len(hash_functions))):
            if i < len(hash_functions):
                sketch_idx = hash_functions[i]
                sign = sign_functions[i]
                # This is a simple reconstruction - in practice, median-of-means or other techniques are used
                reconstructed[i] = sketch[sketch_idx] * sign
        
        return reconstructed
    
    def _apply_jl_projection(self, vector: np.ndarray, compressed_dim: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Apply Johnson-Lindenstrauss projection to the input vector."""
        original_dim = len(vector)
        
        # Generate random projection matrix
        # Common choices: Gaussian matrix (entries ~ N(0,1/sqrt(m))) or Rademacher matrix (entries = ±1/sqrt(m))
        # Using Gaussian matrix here
        projection_matrix = np.random.normal(0, 1/np.sqrt(compressed_dim), (compressed_dim, original_dim))
        
        # Apply projection: y = Ax
        sketched_vector = projection_matrix @ vector
        
        sketch_info = {
            'projection_matrix': projection_matrix
        }
        
        return sketched_vector, sketch_info
    
    def _reverse_jl_projection(self, sketched_vector: np.ndarray, original_dim: int, sketch_info: Dict[str, Any]) -> np.ndarray:
        """Reverse JL projection using pseudoinverse or transpose."""
        # Note: JL projection is not invertible, so this is an approximation
        # Using transpose of the projection matrix as a simple reconstruction method
        projection_matrix = sketch_info['projection_matrix']
        
        # Reconstruction using transpose (pseudoinverse would be more accurate but more expensive)
        reconstructed = projection_matrix.T @ sketched_vector
        
        # For better reconstruction, one might use iterative methods or other techniques
        return reconstructed
    
    def aggregate_sketches(self, sketches: List[np.ndarray]) -> np.ndarray:
        """
        Aggregate multiple sketches. One of the key properties of linear sketches
        is that they can be aggregated linearly.
        
        Args:
            sketches: List of sketched vectors from different parties
            
        Returns:
            Aggregated sketch
        """
        if not sketches:
            return np.array([])
        
        # Sum all sketches element-wise (linear aggregation property)
        aggregated = sketches[0].copy()
        for sketch in sketches[1:]:
            aggregated += sketch
            
        return aggregated


class CountSketchCompressor(SketchCompressor):
    """Specialized compressor for CountSketch."""
    
    def __init__(self, compressed_dim_ratio: float = 0.1, seed: int = 42):
        super().__init__("countsketch", compressed_dim_ratio, seed)


class JLProjectionCompressor(SketchCompressor):
    """Specialized compressor for Johnson-Lindenstrauss projections."""
    
    def __init__(self, compressed_dim_ratio: float = 0.1, seed: int = 42):
        super().__init__("jl", compressed_dim_ratio, seed)