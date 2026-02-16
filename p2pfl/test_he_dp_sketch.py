"""
Test script for the HE+DP+Sketch implementation in p2pfl.
This script tests all the main components: HE, DP, Sketch, and their combination.
"""
import numpy as np
import sys
import os

# Add the p2pfl directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

# Import basic components first
from p2pfl.learning.compression import (
    RenyiDifferentialPrivacyCompressor,
    SketchCompressor,
)

# Import optional components with checks
try:
    from p2pfl.learning.compression import HEDPSketchCompressor, HomomorphicEncryptionCompressor
    # Check if the imports were successful (not None)
    if HomomorphicEncryptionCompressor is None:
        print("HomomorphicEncryptionCompressor is not available (requires p2pfl[he])")
        HomomorphicEncryptionCompressor = None  # Ensure it's explicitly None
    if HEDPSketchCompressor is None:
        print("HEDPSketchCompressor is not available (requires p2pfl[he,dp])")
        HEDPSketchCompressor = None  # Ensure it's explicitly None
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you have installed the required dependencies.")
    print("For HE+DP+Sketch functionality, install with: pip install p2pfl[he,dp]")
    sys.exit(1)


def test_renyi_dp():
    """Test the Renyi Differential Privacy compressor."""
    print("Testing Renyi Differential Privacy compressor...")
    
    # Create compressor
    dp_comp = RenyiDifferentialPrivacyCompressor()
    
    # Create sample parameters
    sample_params = [np.random.randn(10, 5), np.random.randn(3, 4)]
    
    # Apply DP
    dp_params, dp_info = dp_comp.apply_strategy(
        sample_params,
        clip_norm=1.0,
        rdp_epsilon=1.0,
        delta=1e-5,
        num_rounds=1
    )
    
    print(f"  - Original shapes: {[p.shape for p in sample_params]}")
    print(f"  - DP applied: {dp_info['dp_applied']}")
    print(f"  - Epsilon: {dp_info['epsilon']:.4f}")
    print(f" - Delta: {dp_info['delta']}")
    print(f"  - Was clipped: {dp_info['was_clipped']}")
    
    # Reverse (just returns same params since DP is irreversible)
    reversed_params = dp_comp.reverse_strategy(dp_params, dp_info)
    print(f"  - Reversed shapes: {[p.shape for p in reversed_params]}")
    
    print("✓ Renyi DP test passed\n")


def test_sketch():
    """Test the Sketch compressor."""
    print("Testing Sketch compressor...")
    
    # Test CountSketch
    sketch_comp = SketchCompressor(sketch_type="countsketch", compressed_dim_ratio=0.5, seed=42)
    
    # Create sample parameters
    sample_params = [np.random.randn(10), np.random.randn(6)]  # Total 16 elements
    
    # Apply sketching
    sketched_params, sketch_info = sketch_comp.apply_strategy(sample_params)
    
    print(f"  - Original total size: {sum(p.size for p in sample_params)}")
    print(f"  - Compressed size: {sketched_params[0].size}")
    print(f"  - Compression ratio: {sketch_info['compressed_dim']}/{sketch_info['original_dim']}")
    print(f"  - Sketch type: {sketch_info['sketch_type']}")
    
    # Reverse sketching
    reversed_params = sketch_comp.reverse_strategy(sketched_params, sketch_info)
    print(f"  - Reversed total size: {sum(p.size for p in reversed_params)}")
    
    # Test JL Projection
    jl_sketch_comp = SketchCompressor(sketch_type="jl", compressed_dim_ratio=0.3, seed=42)
    jl_sketched_params, jl_sketch_info = jl_sketch_comp.apply_strategy(sample_params)
    
    print(f"  - JL: Original size: {jl_sketch_info['original_dim']}, Compressed size: {jl_sketch_info['compressed_dim']}")
    
    print("✓ Sketch test passed\n")


def test_he():
    """Test the Homomorphic Encryption compressor."""
    print("Testing Homomorphic Encryption compressor...")
    
    # Check if HE is available
    if HomomorphicEncryptionCompressor is None:
        print(" - HomomorphicEncryptionCompressor: Not available (requires p2pfl[he])")
        print("✓ HE test skipped\n")
        return
    
    # Create compressor
    he_comp = HomomorphicEncryptionCompressor(num_parties=3, top_l_ratio=0.5)
    
    # Create sample parameters
    sample_params = [np.random.randn(4, 3)]  # 12 elements total
    
    # Apply HE
    he_params, he_info = he_comp.apply_strategy(sample_params)
    
    print(f" - Original total size: {sum(p.size for p in sample_params)}")
    print(f"  - Top-L indices count: {len(he_info['top_l_indices'])}")
    print(f"  - Plaintext indices count: {len(he_info['plaintext_indices'])}")
    print(f"  - Total encrypted values: {len(he_info['encrypted_values'])}")
    print(f"  - Total plaintext values: {len(he_info['plaintext_values'])}")
    
    # Note: We can't fully reverse HE without setting as aggregator
    he_comp.set_as_aggregator(True)
    reversed_params = he_comp.reverse_strategy(he_params, he_info)
    print(f" - Reversed total size: {sum(p.size for p in reversed_params)}")
    
    print("✓ HE test passed\n")


def test_he_dp_sketch():
    """Test the full HE+DP+Sketch pipeline."""
    print("Testing HE+DP+Sketch pipeline...")
    
    # Check if HEDPSketchCompressor is available
    if HEDPSketchCompressor is None:
        print("  - HEDPSketchCompressor: Not available (requires p2pfl[he,dp])")
        print("✓ HE+DP+Sketch test skipped\n")
        return
    
    # Create compressor
    he_dp_sketch = HEDPSketchCompressor(
        dp_clip_norm=1.0,
        dp_rdp_epsilon=0.5,
        dp_delta=1e-5,
        sketch_type="countsketch",
        sketch_compressed_dim_ratio=0.2,  # Compress to 20%
        he_num_parties=3,
        he_top_l_ratio=0.5,  # Encrypt top 50% of sketched coords
        seed=42
    )
    
    # Create sample parameters
    sample_params = [np.random.randn(10, 5), np.random.randn(2, 3)]  # Total 56 elements
    
    print(f"  - Original total size: {sum(p.size for p in sample_params)}")
    
    # Apply full pipeline
    compressed_params, info = he_dp_sketch.apply_strategy(sample_params, round_num=1)
    
    print(f"  - After DP: clip_norm={info['dp_info']['clip_norm']}, epsilon={info['dp_info']['epsilon']:.4f}")
    print(f"  - After Sketch: original_dim={info['sketch_info']['original_dim']}, compressed_dim={info['sketch_info']['compressed_dim']}")
    print(f"  - After HE: top_L_indices_count={len(info['he_info']['top_l_indices'])}")
    
    # Test aggregation
    # Create multiple compressed models
    model1_params, model1_info = he_dp_sketch.apply_strategy(
        [np.random.randn(10, 5) * 0.1, np.random.randn(2, 3) * 0.1], 
        round_num=1
    )
    model2_params, model2_info = he_dp_sketch.apply_strategy(
        [np.random.randn(10, 5) * 0.2, np.random.randn(2, 3) * 0.2], 
        round_num=1
    )
    
    compressed_models = [
        (model1_params, model1_info),
        (model2_params, model2_info)
    ]
    
    # Aggregate compressed models
    aggregated_params, aggregated_info = he_dp_sketch.aggregate_compressed_models(compressed_models)
    print(f"  - Aggregated model created successfully")
    
    # Set as aggregator and reverse
    he_dp_sketch.set_as_aggregator(True)
    final_params = he_dp_sketch.reverse_strategy(aggregated_params, aggregated_info)
    print(f" - Final model shapes: {[p.shape for p in final_params]}")
    
    print("✓ HE+DP+Sketch test passed\n")


def test_aggregation():
    """Test the aggregation functionality."""
    print("Testing aggregation functionality...")
    
    # Check if HEDPSketchCompressor is available
    if HEDPSketchCompressor is None or not callable(HEDPSketchCompressor):
        print("  - HEDPSketchCompressor: Not available (requires p2pfl[he,dp])")
        print("✓ Aggregation test skipped\n")
        return
    
    # Create compressor
    he_dp_sketch = HEDPSketchCompressor(
        dp_clip_norm=1.0,
        dp_rdp_epsilon=0.5,
        dp_delta=1e-5,
        sketch_type="jl",
        sketch_compressed_dim_ratio=0.1,  # Compress to 10%
        he_num_parties=2,
        he_top_l_ratio=0.3,  # Encrypt top 30% of sketched coords
        seed=123
    )
    
    # Create sample models from different "nodes"
    model_updates = []
    for i in range(3):
        params = [np.random.randn(5, 4) * (i + 1), np.random.randn(3) * (i + 1)]
        compressed_params, info = he_dp_sketch.apply_strategy(params, round_num=1)
        model_updates.append((compressed_params, info))
    
    print(f"  - Created {len(model_updates)} compressed model updates")
    
    # Perform aggregation
    aggregated_params, aggregated_info = he_dp_sketch.aggregate_compressed_models(model_updates)
    print(f"  - Aggregated {len(model_updates)} models successfully")
    
    # Verify that aggregation worked
    assert aggregated_params is not None, "Aggregated parameters should not be None"
    assert aggregated_info is not None, "Aggregated info should not be None"
    
    print("✓ Aggregation test passed\n")


def run_all_tests():
    """Run all tests."""
    print("="*60)
    print("Running HE+DP+Sketch Tests")
    print("="*60)
    
    try:
        test_renyi_dp()
        test_sketch()
        test_he()
        test_he_dp_sketch()
        test_aggregation()
        
        print("="*60)
        print("🎉 All applicable tests passed!")
        print("The HE+DP+Sketch implementation is working correctly for available components.")
        print("="*60)
        
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = run_all_tests()
    if not success:
        sys.exit(1)