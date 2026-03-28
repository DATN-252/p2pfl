import os
import time
import threading
import psutil
import json
import subprocess
from typing import Dict, List, Any

class Monitor(threading.Thread):
    """
    Monitors system resources (CPU, RAM, GPU) during experiment execution.
    """
    def __init__(self, interval: float = 1.0):
        super().__init__()
        self.interval = interval
        self.stopped = threading.Event()
        self.cpu_usages: List[float] = []
        self.ram_usages: List[float] = []
        self.gpu_usages: List[float] = []
        self.start_time = 0.0
        self.end_time = 0.0

    def run(self):
        self.start_time = time.time()
        # Initial call to cpu_percent to initialize it
        psutil.cpu_percent(interval=None)
        
        while not self.stopped.is_set():
            time.sleep(self.interval)
            
            # CPU usage since last call
            self.cpu_usages.append(psutil.cpu_percent(interval=None))
            
            # RAM usage
            self.ram_usages.append(psutil.virtual_memory().percent)
            
            # GPU usage (if available)
            gpu_usage = self._get_gpu_usage()
            if gpu_usage is not None:
                self.gpu_usages.append(gpu_usage)
        self.end_time = time.time()

    def stop(self):
        self.stopped.set()

    def _get_gpu_usage(self) -> float | None:
        try:
            # Try using nvidia-smi
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                check=False
            )
            if result.returncode == 0:
                usages = [float(x) for x in result.stdout.strip().split('\n') if x.strip()]
                if usages:
                    return sum(usages) / len(usages)
        except Exception:
            pass
        return None

    def get_metrics(self) -> Dict[str, Any]:
        duration = self.end_time - self.start_time
        avg_cpu = sum(self.cpu_usages) / len(self.cpu_usages) if self.cpu_usages else 0.0
        avg_ram = sum(self.ram_usages) / len(self.ram_usages) if self.ram_usages else 0.0
        avg_gpu = sum(self.gpu_usages) / len(self.gpu_usages) if self.gpu_usages else 0.0
        
        return {
            "execution_time_seconds": duration,
            "average_cpu_usage_percent": avg_cpu,
            "average_ram_usage_percent": avg_ram,
            "average_gpu_usage_percent": avg_gpu if self.gpu_usages else None,
            "peak_cpu_usage_percent": max(self.cpu_usages) if self.cpu_usages else 0.0,
            "peak_ram_usage_percent": max(self.ram_usages) if self.ram_usages else 0.0,
            "peak_gpu_usage_percent": max(self.gpu_usages) if self.gpu_usages else None
        }

    def save_metrics(self, file_path: str):
        metrics = self.get_metrics()
        with open(file_path, 'w') as f:
            json.dump(metrics, f, indent=4)

    @staticmethod
    def get_current_usage() -> Dict[str, float | None]:
        """
        Static method to get a snapshot of current system resource usage.
        """
        # CPU usage (interval=None gives the usage since the last call or first call)
        cpu = psutil.cpu_percent(interval=None)
        
        # RAM usage
        ram = psutil.virtual_memory().percent
        
        # GPU usage
        gpu = None
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                check=False
            )
            if result.returncode == 0:
                usages = [float(x) for x in result.stdout.strip().split('\n') if x.strip()]
                if usages:
                    gpu = sum(usages) / len(usages)
        except Exception:
            pass
            
        return {
            "cpu_percent": cpu,
            "ram_percent": ram,
            "gpu_percent": gpu
        }
