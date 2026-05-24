#
# This file is part of the federated_learning_p2p (p2pfl) distribution
# (see https://github.com/pguijas/p2pfl).
#

import os
import torch
import numpy as np
from typing import List
from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel
from p2pfl.management.logger import logger
from p2pfl.settings import Settings

class ModelPackager:
    """Handles logic for packaging models based on weight consensus."""
    
    def __init__(self, epsilon: float = 1e-3, patience: int = 5):
        self.epsilon = epsilon
        self.patience = patience
        self.patience_counter = 0
        self.already_packaged = False 

    def check_and_package(self, models: List[P2PFLModel], aggregated_model: P2PFLModel, round_num: int, output_path: str, node_addr: str):
        """
        Check if models have converged and save. Only authorized node pushes to IS.
        """
        if not models or aggregated_model is None:
            return

        # --- OPTION 1: FIXED INTERVAL PACKAGING (Experiment mode) ---
        interval = Settings.training.PACKAGING_INTERVAL
        if interval > 0:
            if (round_num + 1) % interval == 0:
                # ONLY the authorized node saves and pushes to avoid race conditions (50 nodes writing to the same file)
                if node_addr == Settings.training.AUTHORIZED_PUSH_NODE:
                    logger.info("ModelPackager", f"Round {round_num}: Interval reached ({interval}). Authorized node saving & pushing...")
                    self._save_model_locally(aggregated_model, round_num, output_path)
                    self._trigger_inference_service(aggregated_model, round_num)
                else:
                    logger.debug("ModelPackager", f"Round {round_num}: Interval reached. Node {node_addr} not authorized (skipping to avoid race).")
            return # IMPORTANT: If interval is set, we skip consensus logic entirely to avoid extra saves

        # --- OPTION 2: CONSENSUS-BASED PACKAGING (Practical mode) ---
        # 1. Get aggregated parameters as a flat vector
        agg_params = np.concatenate([p.flatten() for p in aggregated_model.get_parameters()])
        
        # 2. Calculate distances
        distances = []
        for m in models:
            m_params = np.concatenate([p.flatten() for p in m.get_parameters()])
            dist = np.mean(np.abs(m_params - agg_params)) 
            distances.append(dist)
        
        max_dist = max(distances) if distances else 0
        
        # 3. Check consensus
        if max_dist < self.epsilon:
            self.patience_counter += 1
            logger.info("ModelPackager", f"Round {round_num}: Consensus detected (dist: {max_dist:.6f}). Patience: {self.patience_counter}/{self.patience}")
        else:
            self.patience_counter = 0

        # 4. Package if patience reached
        if self.patience_counter >= self.patience:
            # ONLY the authorized node saves and pushes
            if node_addr == Settings.training.AUTHORIZED_PUSH_NODE:
                logger.info("ModelPackager", f"Round {round_num}: Consensus patience reached. Authorized node saving & pushing...")
                self._save_model_locally(aggregated_model, round_num, output_path)
                self._trigger_inference_service(aggregated_model, round_num)
            else:
                logger.debug("ModelPackager", f"Round {round_num}: Consensus reached. Node {node_addr} not authorized.")
            
            self.patience_counter = 0

    def _save_model_locally(self, model: P2PFLModel, round_num: int, output_path: str):
        """Save model to local experiment folder."""
        try:
            # log_dir = os.path.join(output_path, "logs")
            log_dir = output_path
            os.makedirs(log_dir, exist_ok=True)
            
            # 1. Save PyTorch State Dict (Original)
            file_name = f"packaged_model_round_{round_num}.pt"
            save_path = os.path.abspath(os.path.join(log_dir, file_name))
            
            torch_model = model.get_model()
            if hasattr(torch_model, "state_dict"):
                torch.save(torch_model.state_dict(), save_path)
                logger.debug("ModelPackager", f"Model saved locally at round {round_num}")
                
                # 2. Export to ONNX (New)
                onnx_file_name = f"packaged_model_round_{round_num}.onnx"
                onnx_save_path = os.path.abspath(os.path.join(log_dir, onnx_file_name))
                self._export_to_onnx(torch_model, onnx_save_path)
                
        except Exception as e:
            logger.error("ModelPackager", f"Local save failed: {e}")

    def _export_to_onnx(self, model: torch.nn.Module, save_path: str):
        """
        Export a PyTorch model to ONNX format using heuristics to find input shape.
        """
        try:
            # Set to evaluation mode
            model.eval()
            
            dummy_input = None
            
            # Heuristic 1: Use Lightning's example_input_array if available
            if hasattr(model, "example_input_array") and model.example_input_array is not None:
                dummy_input = model.example_input_array
                logger.debug("ModelPackager", "ONNX: Using example_input_array for dummy input.")
            
            # Heuristic 2: Check hparams (common in LightningModule)
            elif hasattr(model, "hparams") and "input_size" in model.hparams:
                input_size = model.hparams["input_size"]
                if isinstance(input_size, int):
                    dummy_input = torch.randn(1, input_size)
                elif isinstance(input_size, (list, tuple)):
                    dummy_input = torch.randn(1, *input_size)
                logger.debug("ModelPackager", f"ONNX: Inferred dummy input from hparams.input_size: {input_size}")

            # Heuristic 3: Inspect layers for common input shapes
            else:
                for layer in model.modules():
                    if isinstance(layer, torch.nn.Linear):
                        dummy_input = torch.randn(1, layer.in_features)
                        logger.debug("ModelPackager", f"ONNX: Inferred dummy input from first Linear layer: {layer.in_features}")
                        break
                    elif isinstance(layer, torch.nn.Conv2d):
                        # Default to 3 channels, 32x32 if we can't be sure, but use layer's in_channels
                        dummy_input = torch.randn(1, layer.in_channels, 32, 32)
                        logger.debug("ModelPackager", f"ONNX: Inferred dummy input from first Conv2d layer: {layer.in_channels} channels")
                        break
            
            if dummy_input is not None:
                # Ensure dummy input is on the same device as model
                device = next(model.parameters()).device
                if isinstance(dummy_input, torch.Tensor):
                    dummy_input = dummy_input.to(device)
                
                torch.onnx.export(
                    model,
                    dummy_input,
                    save_path,
                    export_params=True,
                    opset_version=14, # Newer opset for better compatibility
                    do_constant_folding=True,
                    input_names=['input'],
                    output_names=['output'],
                    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
                )
                logger.info("ModelPackager", f"✅ ONNX Export success: {save_path}")
            else:
                logger.warning("ModelPackager", "⚠️ ONNX Export skipped: Could not determine dummy input shape.")
                
        except Exception as e:
            # We don't want to crash the whole training if ONNX export fails (e.g. missing onnx package)
            logger.warning("ModelPackager", f"⚠️ ONNX Export failed (Non-critical): {e}")

    def _trigger_inference_service(self, model: P2PFLModel, round_num: int):
        """Send the actual model weights via HTTP POST to the inference service."""
        import requests
        import base64
        import io
        
        url = "http://127.0.0.1:8000/reload"
        try:
            # 1. Serialize state_dict to a byte stream in memory
            torch_model = model.get_model()
            buffer = io.BytesIO()
            torch.save(torch_model.state_dict(), buffer)
            
            # 2. Encode bytes to Base64 string for JSON compatibility
            model_bytes = buffer.getvalue()
            model_b64 = base64.b64encode(model_bytes).decode('utf-8')
            
            # 3. Send payload
            payload = {
                "model_data": model_b64, 
                "round": round_num,
                "format": "pt_base64"
            }
            response = requests.post(url, json=payload, timeout=10)
            
            if response.status_code == 200:
                logger.info("ModelPackager", f"🚀 DIRECT PUSH success: Model Round {round_num} sent to Inference Service (~{len(model_bytes)/1024:.1f} KB)")
            else:
                logger.debug("ModelPackager", f"Direct Push ignored: Service returned {response.status_code}")
        except Exception as e:
            logger.debug("ModelPackager", f"Inference service not reachable: {e}")
