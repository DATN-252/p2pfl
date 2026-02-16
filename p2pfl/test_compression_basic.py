"""
Basic test script for compression strategies in p2pfl.
This script tests the basic functionality without requiring all optional dependencies.
"""
import numpy as np
import sys
import os

# Add the p2pfl directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

def test_basic_compressions():
    """Test basic compression strategies that don't require optional dependencies."""
    print("Testing basic compression strategies...")
    
    # Test basic imports
    from p2pfl.learning.compression import (
        PTQuantization,
        TopKSparsification,
        LowRankApproximation,
        ZlibCompressor,
        LZMACompressor
    )
    
    # Create sample parameters
    sample_params = [np.random.randn(10, 5), np.random.randn(3, 4)]
    
    print(f"  - Original shapes: {[p.shape for p in sample_params]}")
    print(f"  - Original sizes: {[p.size for p in sample_params]}")
    
    # Test PTQuantization
    try:
        quantizer = PTQuantization()
        quantized_params, quant_info = quantizer.apply_strategy(sample_params)
        print(f"  - PTQuantization: OK, info keys: {list(quant_info.keys())}")
    except Exception as e:
        print(f"  - PTQuantization: Error - {e}")
    
    # Test TopKSparsification
    try:
        topk = TopKSparsification()
        topk_params, topk_info = topk.apply_strategy(sample_params, k=0.5)  # Keep top 50%
        print(f"  - TopKSparsification: OK, kept {(topk_info.get('n_elements_kept', []) if isinstance(topk_info.get('n_elements_kept'), list) else [0])[0] if len(topk_info.get('n_elements_kept', [])) > 0 else 'unknown'} elements out of {sample_params[0].size}")
    except Exception as e:
        print(f"  - TopKSparsification: Error - {e}")
    
    # Test LowRankApproximation
    try:
        lra = LowRankApproximation()
        lra_params, lra_info = lra.apply_strategy(sample_params, threshold=0.9)
        print(f"  - LowRankApproximation: OK")
    except Exception as e:
        print(f"  - LowRankApproximation: Error - {e}")
    
    # Test ZlibCompressor
    try:
        zlib_comp = ZlibCompressor()
        # For zlib, we need to work with bytes, so we'll test with a simple approach
        sample_bytes = b'test data for compression'
        compressed_bytes = zlib_comp.apply_strategy(sample_bytes)
        print(f"  - ZlibCompressor: OK, compressed {len(sample_bytes)} -> {len(compressed_bytes)} bytes")
    except Exception as e:
        print(f"  - ZlibCompressor: Error - {e}")
    
    # Test LZMACompressor
    try:
        lzma_comp = LZMACompressor()
        sample_bytes = b'test data for compression'
        compressed_bytes = lzma_comp.apply_strategy(sample_bytes)
        print(f"  - LZMACompressor: OK, compressed {len(sample_bytes)} -> {len(compressed_bytes)} bytes")
    except Exception as e:
        print(f"  - LZMACompressor: Error - {e}")
    
    print("✓ Basic compression tests completed\n")


def test_dp_if_available():
    """Test DP strategies if available."""
    print("Testing DP strategies if available...")
    
    try:
        from p2pfl.learning.compression import DifferentialPrivacyCompressor
        
        # Create sample parameters
        sample_params = [np.random.randn(5, 4), np.random.randn(2, 3)]
        
        # Test DP
        dp_comp = DifferentialPrivacyCompressor()
        dp_params, dp_info = dp_comp.apply_strategy(
            sample_params,
            clip_norm=1.0,
            epsilon=1.0,
            delta=1e-5
        )
        
        print(f"  - DifferentialPrivacyCompressor: OK")
        print(f"    * Original total size: {sum(p.size for p in sample_params)}")
        print(f"    * Clip norm: {dp_info['clip_norm']}")
        print(f"    * Epsilon: {dp_info['epsilon']}")
        print(f"    * Was clipped: {dp_info['was_clipped']}")
        
    except ImportError:
        print("  - DifferentialPrivacyCompressor: Not available (requires p2pfl[dp])")
    except Exception as e:
        print(f"  - DifferentialPrivacyCompressor: Error - {e}")
    
    print("✓ DP tests completed (if available)\n")


def test_renyi_dp_if_available():
    """Test Renyi DP strategies if available."""
    print("Testing Renyi DP strategies if available...")
    
    try:
        from p2pfl.learning.compression import RenyiDifferentialPrivacyCompressor
        
        # Create sample parameters
        sample_params = [np.random.randn(4, 3), np.random.randn(2, 2)]
        
        # Test Renyi DP
        rdp_comp = RenyiDifferentialPrivacyCompressor()
        rdp_params, rdp_info = rdp_comp.apply_strategy(
            sample_params,
            clip_norm=1.0,
            rdp_epsilon=0.5,
            delta=1e-5,
            num_rounds=1
        )
        
        print(f"  - RenyiDifferentialPrivacyCompressor: OK")
        print(f"    * Original total size: {sum(p.size for p in sample_params)}")
        print(f"    * RDP epsilon: {rdp_info['rdp_epsilon']}")
        print(f"    * Final epsilon: {rdp_info['epsilon']:.4f}")
        print(f"    * Delta: {rdp_info['delta']}")
        
    except ImportError:
        print("  - RenyiDifferentialPrivacyCompressor: Not available (requires p2pfl[dp])")
    except Exception as e:
        print(f"  - RenyiDifferentialPrivacyCompressor: Error - {e}")
    
    print("✓ Renyi DP tests completed (if available)\n")


def test_sketch_if_available():
    """Test Sketch strategies if available."""
    print("Testing Sketch strategies if available...")
    
    try:
        from p2pfl.learning.compression import SketchCompressor
        
        # Create sample parameters
        sample_params = [np.random.randn(8), np.random.randn(4)]  # Total 12 elements
        
        # Test CountSketch
        sketch_comp = SketchCompressor(sketch_type="countsketch", compressed_dim_ratio=0.5, seed=42)
        sketched_params, sketch_info = sketch_comp.apply_strategy(sample_params)
        
        print(f"  - SketchCompressor (CountSketch): OK")
        print(f"    * Original total size: {sum(p.size for p in sample_params)}")
        print(f"    * Compressed size: {sketched_params[0].size}")
        print(f"    * Compression ratio: {sketch_info['compressed_dim']}/{sketch_info['original_dim']}")
        
        # Test JL Projection
        jl_sketch_comp = SketchCompressor(sketch_type="jl", compressed_dim_ratio=0.3, seed=42)
        jl_sketched_params, jl_sketch_info = jl_sketch_comp.apply_strategy(sample_params)
        
        print(f"  - SketchCompressor (JL): OK")
        print(f"    * JL: Original size: {jl_sketch_info['original_dim']}, Compressed size: {jl_sketch_info['compressed_dim']}")
        
    except ImportError:
        print("  - SketchCompressor: Not available")
    except Exception as e:
        print(f"  - SketchCompressor: Error - {e}")
    
    print("✓ Sketch tests completed (if available)\n")


def test_he_if_available():
    """Test HE strategies if available."""
    print("Testing HE strategies if available...")
    
    try:
        from p2pfl.learning.compression import HomomorphicEncryptionCompressor
        
        # Create sample parameters
        sample_params = [np.random.randn(4, 2)]  # 8 elements total
        
        # Test HE
        he_comp = HomomorphicEncryptionCompressor(num_parties=2, top_l_ratio=0.5)
        he_params, he_info = he_comp.apply_strategy(sample_params)
        
        print(f"  - HomomorphicEncryptionCompressor: OK")
        print(f"    * Original total size: {sum(p.size for p in sample_params)}")
        print(f"    * Top-L indices count: {len(he_info['top_l_indices'])}")
        print(f"    * Plaintext indices count: {len(he_info['plaintext_indices'])}")
        
    except ImportError:
        print("  - HomomorphicEncryptionCompressor: Not available (requires p2pfl[he])")
    except Exception as e:
        print(f"  - HomomorphicEncryptionCompressor: Error - {e}")
    
    print("✓ HE tests completed (if available)\n")


def test_he_dp_sketch_if_available():
    """Test HE+DP+Sketch pipeline if available."""
    print("Testing HE+DP+Sketch pipeline if available...")
    
    try:
        from p2pfl.learning.compression import HEDPSketchCompressor
        
        if HEDPSketchCompressor is None:
            print("  - HEDPSketchCompressor: Not available")
            return
            
        # Create sample parameters
        sample_params = [np.random.randn(6, 4), np.random.randn(2, 3)]  # Total 30 elements
        
        # Create compressor
        he_dp_sketch = HEDPSketchCompressor(
            dp_clip_norm=1.0,
            dp_rdp_epsilon=0.5,
            dp_delta=1e-5,
            sketch_type="countsketch",
            sketch_compressed_dim_ratio=0.3,  # Compress to 30%
            he_num_parties=2,
            he_top_l_ratio=0.5,  # Encrypt top 50% of sketched coords
            seed=42
        )
        
        print(f" - HEDPSketchCompressor: Created successfully")
        print(f"    * Original total size: {sum(p.size for p in sample_params)}")
        
        # Apply full pipeline
        compressed_params, info = he_dp_sketch.apply_strategy(sample_params, round_num=1)
        
        print(f"    * After DP: clip_norm={info['dp_info']['clip_norm']}, epsilon={info['dp_info']['epsilon']:.4f}")
        print(f"    * After Sketch: original_dim={info['sketch_info']['original_dim']}, compressed_dim={info['sketch_info']['compressed_dim']}")
        print(f"    * After HE: top_L_indices_count={len(info['he_info']['top_l_indices'])}")
        
    except ImportError as e:
        print(f" - HEDPSketchCompressor: Not available - {e}")
    except Exception as e:
        print(f" - HEDPSketchCompressor: Error - {e}")
    
    print("✓ HE+DP+Sketch tests completed (if available)\n")


def run_all_tests():
    """Run all tests."""
    print("="*70)
    print("Running Basic Compression Tests (Optional Dependencies Not Required)")
    print("="*70)
    
    try:
        test_basic_compressions()
        test_dp_if_available()
        test_renyi_dp_if_available()
        test_sketch_if_available()
        test_he_if_available()
        test_he_dp_sketch_if_available()
        
        print("="*70)
        print("🎉 Basic tests completed!")
        print("Some components require optional dependencies (p2pfl[dp], p2pfl[he])")
        print("to be fully functional.")
        print("="*70)
        
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