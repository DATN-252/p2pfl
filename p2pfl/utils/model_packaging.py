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
                logger.info("ModelPackager", f"Round {round_num}: Interval reached ({interval}). Saving model...")
                
                # ALWAYS save locally for backup
                self._save_model_locally(aggregated_model, round_num, output_path)

                if node_addr == Settings.training.AUTHORIZED_PUSH_NODE:
                    logger.info("ModelPackager", f"🌟 Node {node_addr} is AUTHORIZED. Pushing to IS...")
                    self._trigger_inference_service(aggregated_model, round_num)
                else:
                    logger.debug("ModelPackager", f"Node {node_addr} not authorized to push to Inference Service.")
                return 

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
            logger.info("ModelPackager", f"Round {round_num}: Consensus patience reached. Saving model...")
            
            # ALWAYS save locally for backup
            self._save_model_locally(aggregated_model, round_num, output_path)

            if node_addr == Settings.training.AUTHORIZED_PUSH_NODE:
                logger.info("ModelPackager", f"🌟 Node {node_addr} is AUTHORIZED. Pushing to IS...")
                self._trigger_inference_service(aggregated_model, round_num)
            else:
                logger.debug("ModelPackager", f"Node {node_addr} not authorized to push to Inference Service.")
            
            self.patience_counter = 0

    def _save_model_locally(self, model: P2PFLModel, round_num: int, output_path: str):
        """Save model to local experiment folder."""
        try:
            log_dir = os.path.join(output_path, "logs")
            os.makedirs(log_dir, exist_ok=True)
            file_name = f"packaged_model_round_{round_num}.pt"
            save_path = os.path.abspath(os.path.join(log_dir, file_name))
            
            torch_model = model.get_model()
            if hasattr(torch_model, "state_dict"):
                torch.save(torch_model.state_dict(), save_path)
                logger.debug("ModelPackager", f"Model saved locally at round {round_num}")
        except Exception as e:
            logger.error("ModelPackager", f"Local save failed: {e}")

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

    def _save_model(self, model: P2PFLModel, round_num: int, output_path: str, dist: float):
        """Save the model weights locally and trigger direct push."""
        try:
            # Always save locally first for backup
            log_dir = os.path.join(output_path, "logs")
            os.makedirs(log_dir, exist_ok=True)
            file_name = f"packaged_model_round_{round_num}.pt"
            save_path = os.path.abspath(os.path.join(log_dir, file_name))
            
            torch_model = model.get_model()
            if hasattr(torch_model, "state_dict"):
                torch.save(torch_model.state_dict(), save_path)
                logger.info("ModelPackager", f"✅ Model SAVED at round {round_num} to {save_path}")
                
                # --- DIRECT PUSH TRIGGER ---
                self._trigger_inference_service(model, round_num)
            else:
                logger.error("ModelPackager", "Model does not have state_dict, skipping.")
        except Exception as e:
            logger.error("ModelPackager", f"Failed to save/push model: {e}")
