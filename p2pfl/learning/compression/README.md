# Compression Strategies in P2PFL

This directory contains various compression strategies implemented in P2PFL, including traditional methods and advanced privacy-preserving techniques.

## Available Compression Strategies

### Traditional Compression
- **Quantization**: Post-Training Quantization (PTQ) for reducing model size
- **Top-K Sparsification**: Keeping only the top-K largest weights
- **Low-Rank Approximation**: Matrix decomposition-based compression
- **Zlib/LZMA Compression**: General-purpose data compression

### Privacy-Preserving Compression
- **Differential Privacy (DP)**: Local DP with clipping and noise addition
- **Renyi Differential Privacy (RDP)**: Enhanced privacy accounting with RDP
- **CountSketch/JL Projections**: Linear sketching for dimensionality reduction
- **Homomorphic Encryption (HE)**: Partial encryption with threshold decryption
- **HE+DP+Sketch**: Hybrid protocol combining all three techniques

## HE+DP+Sketch Protocol

The HE+DP+Sketch protocol implements the methodology described in the paper "Enhancing Privacy and Efficiency in Decentralized Federated Learning via Hybrid Homomorphic Encryption, Differential Privacy, and Sketch-Based Compression".

### Components

1. **Differential Privacy Layer**:
   - Gradient clipping to bound sensitivity
   - Gaussian mechanism with RDP accounting
   - Multi-round privacy composition

2. **Sketch Layer**:
   - CountSketch or Johnson-Lindenstrauss projections
   - Reduces dimension from O(d) to O(m) where m ≪ d
   - Preserves linear aggregation properties

3. **Homomorphic Encryption Layer**:
   - Partial encryption of top-L coordinates
   - Threshold decryption for secure aggregation
   - Preserves privacy of individual updates

### Usage

```python
from p2pfl.learning.compression import HEDPSketchCompressor

# Create the hybrid compressor
compressor = HEDPSketchCompressor(
    dp_clip_norm=1.0,           # Clipping norm for DP
    dp_rdp_epsilon=1.0,         # Per-round RDP epsilon
    dp_delta=1e-5,              # DP delta parameter
    sketch_type="countsketch",  # Type of sketch ("countsketch" or "jl")
    sketch_compressed_dim_ratio=0.1,  # Compress to 10% of original size
    he_num_parties=5,           # Number of parties in the network
    he_top_l_ratio=0.5,         # Encrypt top 50% of sketched coordinates
    seed=42                     # Random seed for reproducibility
)

# Apply the full pipeline: DP→Sketch→Partial HE
compressed_params, info = compressor.apply_strategy(
    local_model_params, 
    round_num=1
)

# For aggregation, use the aggregate_compressed_models method
aggregated_params, aggregated_info = compressor.aggregate_compressed_models(
    [model1, model2, ...]
)

# The aggregator can then reverse to get the final model
final_model = compressor.reverse_strategy(aggregated_params, aggregated_info)
```

### Parameters Explained

- `dp_clip_norm`: Maximum L2 norm for gradient clipping (controls sensitivity)
- `dp_rdp_epsilon`: Per-round Renyi DP parameter (controls privacy level)
- `sketch_compressed_dim_ratio`: Ratio m/d where m is compressed dimension and d is original (controls compression)
- `he_top_l_ratio`: Ratio L/m where L is number of coordinates to encrypt (controls HE overhead)

### Theoretical Guarantees

The implementation provides:
- **End-to-end (ε,δ)-DP**: Via post-processing invariance of sketching and encryption
- **Secure Aggregation**: Only sums are revealed through threshold decryption
- **Communication Efficiency**: From O(d) to O(m) message size with m ≪ d
- **Convergence**: Bounded by DP noise and sketch error terms

## API Reference

### HEDPSketchCompressor
The main class implementing the hybrid protocol with methods:
- `apply_strategy()`: Applies the full DP→Sketch→HE pipeline
- `reverse_strategy()`: Reverses HE→Sketch (DP is irreversible)
- `aggregate_compressed_models()`: Performs homomorphic aggregation
- `set_as_aggregator()`: Configures for aggregation role

### Individual Components
Each component (DP, Sketch, HE) is also available separately for modular use.

## Dependencies

To use the HE+DP+Sketch protocol, you need to install the required dependencies:

```bash
# For homomorphic encryption
pip install p2pfl[he]

# For differential privacy
pip install p2pfl[dp]

# For both HE and DP
pip install p2pfl[he,dp]
```

## References

- Nguyen, T. H. (2024). "Enhancing Privacy and Efficiency in Decentralized Federated Learning via Hybrid Homomorphic Encryption, Differential Privacy, and Sketch-Based Compression"
- Dwork, C., & Roth, A. (2014). "The algorithmic foundations of differential privacy"
- Mironov, I. (2017). "Renyi differential privacy"
- Charikar, M., et al. (2002). "Finding frequent items in data streams" (CountSketch)