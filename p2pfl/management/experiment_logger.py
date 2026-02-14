import os
import json
from typing import Dict, Any

class ExperimentLogger:
    """
    Manages logging of evaluated metrics for a specific node during an experiment.
    Results are stored in a JSON Lines (.jsonl) file, making them easy to append
    and parse for plotting scripts.
    """
    def __init__(self, output_dir: str, node_id: str):
        """
        Initializes the ExperimentLogger.

        Args:
            output_dir (str): The base directory for experiment logs.
            node_id (str): The unique identifier for the node (e.g., its address).
        """
        self.output_dir = output_dir
        self.node_id = node_id
        # Use .jsonl (JSON Lines) for easy appending and parsing of individual JSON objects
        self.log_file_path = os.path.join(output_dir, f"node_{self.node_id}.jsonl")
        
        # Ensure the output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

    def record_metrics(self, round_num: int, metrics: Dict[str, Any]):
        """
        Records evaluated metrics for a given round.

        Args:
            round_num (int): The current training round number.
            metrics (Dict[str, Any]): A dictionary of metrics (e.g., {'test_acc': 0.98}).
        """
        # Clean up keys and values to remove accidental newlines/whitespace
        clean_metrics = {
            str(k).strip(): (v.strip() if isinstance(v, str) else v)
            for k, v in metrics.items()
        }
        log_entry = {
            "round": round_num,
            "metrics": clean_metrics
        }
        with open(self.log_file_path, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')