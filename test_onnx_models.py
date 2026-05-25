import os
import glob
import numpy as np
import onnx
try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except ImportError:
    ORT_AVAILABLE = False

def test_onnx_models(models_dir="models"):
    print(f"🔍 Searching for ONNX models in: {os.path.abspath(models_dir)}")
    
    # Recursively find all .onnx files
    onnx_files = glob.glob(os.path.join(models_dir, "**", "*.onnx"), recursive=True)
    
    if not onnx_files:
        print("❌ No ONNX models found.")
        return

    print(f"✅ Found {len(onnx_files)} ONNX models.\n")

    results = []
    for model_path in onnx_files:
        print(f"--- Testing: {os.path.basename(model_path)} ---")
        print(f"Path: {model_path}")
        
        status = {"name": os.path.basename(model_path), "structure": "❌", "inference": "❌", "error": None}
        try:
            # 1. Basic check with 'onnx' library
            model = onnx.load(model_path)
            onnx.checker.check_model(model)
            print("✅ ONNX structure check: PASSED")
            status["structure"] = "✅"
            
            # 2. Inference check with 'onnxruntime'
            if ORT_AVAILABLE:
                session = ort.InferenceSession(model_path)
                
                # Get input details
                inputs = session.get_inputs()
                input_info = []
                input_feed = {}
                
                for input_node in inputs:
                    name = input_node.name
                    shape = input_node.shape
                    dtype = input_node.type
                    input_info.append(f"{name}(shape={shape}, dtype={dtype})")
                    
                    # Generate dummy data
                    dummy_shape = [s if isinstance(s, int) and s > 0 else 1 for s in shape]
                    dummy_shape = [s if isinstance(s, int) else 1 for s in shape]
                    
                    if "float" in dtype:
                        input_feed[name] = np.random.randn(*dummy_shape).astype(np.float32)
                    elif "int64" in dtype:
                        input_feed[name] = np.random.randint(0, 10, size=dummy_shape).astype(np.int64)
                    else:
                        print(f"⚠️ Unknown dtype {dtype}, skipping inference.")
                        input_feed = None
                        break

                print(f"Inputs: {', '.join(input_info)}")
                
                if input_feed:
                    outputs = session.run(None, input_feed)
                    print(f"✅ Inference: PASSED (Output shape: {outputs[0].shape})")
                    print(f"Sample Output: {outputs[0][0]}")
                    status["inference"] = "✅"
            else:
                print("⚠️ onnxruntime not found. Skipping inference test.")
                status["inference"] = "N/A"
                
        except Exception as e:
            print(f"❌ Error testing model: {e}")
            status["error"] = str(e)
        
        results.append(status)
        print("-" * 40)

    # Summary Report
    print("\n" + "="*50)
    print("                TEST SUMMARY REPORT")
    print("="*50)
    print(f"{'Model Name':<30} | {'Struct':<6} | {'Infer':<6}")
    print("-" * 50)
    for res in results:
        print(f"{res['name'][:30]:<30} | {res['structure']:<6} | {res['inference']:<6}")
    print("="*50)

if __name__ == "__main__":
    test_onnx_models()
