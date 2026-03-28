import os
import json
import threading
from typing import Dict, Any
from p2pfl.utils.monitor import Monitor # NEW IMPORT

class ExperimentLogger:
    """
    Manages logging of evaluated metrics for a specific node during an experiment.
    """
    # Class-level lock to ensure thread-safe file writing across instances if needed
    _file_lock = threading.Lock()

    def __init__(self, output_dir: str, node_id: str):
        self.output_dir = output_dir
        self.node_id = node_id
        self.log_file_path = os.path.join(output_dir, f"node_{self.node_id.replace(':', '_')}.jsonl")
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        # We don't delete here anymore, instead we let the launcher manage folders
        # to preserve history if needed, but we ensure consistency within this instance
        self.__recorded_rounds = set()

    def record_metrics(self, round_num: int, metrics: Dict[str, Any]):
        """
        Records evaluated metrics for a given round.
        """
        if round_num is None:
            return

        with self._file_lock:
            # Check if this round was already recorded by this specific logger instance
            if round_num in self.__recorded_rounds:
                return 
                
            # Clean up keys and values
            clean_metrics = {}
            for k, v in metrics.items():
                clean_key = str(k).replace('\n', '').strip()
                clean_val = v.strip().replace('\n', '') if isinstance(v, str) else v
                clean_metrics[clean_key] = clean_val
            
            # --- Include system usage in metrics ---
            system_usage = Monitor.get_current_usage()
            clean_metrics["system_cpu_usage"] = system_usage["cpu_percent"]
            clean_metrics["system_ram_usage"] = system_usage["ram_percent"]
            if system_usage["gpu_percent"] is not None:
                clean_metrics["system_gpu_usage"] = system_usage["gpu_percent"]
            # --- End Include system usage ---

            log_entry = {
                "round": round_num,
                "metrics": clean_metrics
            }
            
            # Double check: read the file to see if round is already there (Slow but 100% safe)
            # This is the "Nuclear Option" for consistency
            already_in_file = False
            if os.path.exists(self.log_file_path):
                try:
                    with open(self.log_file_path, 'r') as f:
                        for line in f:
                            if line.strip():
                                entry = json.loads(line)
                                if entry.get("round") == round_num:
                                    already_in_file = True
                                    break
                except Exception:
                    pass

            if not already_in_file:
                # Ensure directory exists (again, as a safety measure for distributed/parallel runs)
                os.makedirs(os.path.dirname(self.log_file_path), exist_ok=True)
                
                try:
                    with open(self.log_file_path, 'a') as f:
                        f.write(json.dumps(log_entry) + '\n')
                except FileNotFoundError:
                    # On Windows, path length might be an issue. 
                    # Try using the long path prefix if needed, but for now just log it.
                    from p2pfl.management.logger import logger
                    logger.error(f"ExperimentLogger", f"❌ Failed to write metrics to {self.log_file_path}. Path may be too long.")
            
            self.__recorded_rounds.add(round_num)
