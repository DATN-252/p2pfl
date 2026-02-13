import numpy as np
import torch
from typing import Dict, List, Tuple
from p2pfl.learning.aggregators.aggregator import Aggregator, NoModelsToAggregateError
from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel

class ChocoSGD(Aggregator):
    """
    CHOCO-SGD: Decentralized Optimization with Compressed Communication.
    
    Paper: "Decentralized Stochastic Optimization and Gossip Algorithms with Compressed Communication" (Koloskova et al., ICML 2019)
    
    Update rule:
    1. Local Update: x_{i, t+1/2} = x_{i, t} - lr * grad
    2. Gossip Step:  x_{i, t+1}   = x_{i, t+1/2} + consensus_step * sum(x_hat_j - x_hat_i)
    
    Attributes:
        consensus_step (gamma): Step size for consensus (0 < gamma < 1).
        compression_ratio: Ratio of parameters to keep (e.g., 0.1 for 10%).
    """
    SUPPORTS_PARTIAL_AGGREGATION = True

    def __init__(self, 
                 consensus_step: float = 0.1, 
                 lr: float = 0.01, 
                 compression_ratio: float = 0.1):
        super().__init__()
        self.consensus_step = consensus_step # Hệ số đồng thuận (gamma trong bài báo)
        self.lr = lr                         # Tốc độ học (eta)
        self.compression_ratio = compression_ratio
        
        # MEMORY: Lưu trữ ước lượng model của hàng xóm và chính mình
        # Cấu trúc: { 'node_id': [param_layer_1, param_layer_2, ...] }
        self.estimates: Dict[str, List[np.ndarray]] = {}
        
        # Buffer để lưu sai số chưa gửi (Error Compensation)
        self.error_buffer: List[np.ndarray] = [] 

    def _compress(self, vector: np.ndarray) -> np.ndarray:
        """
        Toán tử nén Top-k (Layer-wise).
        Giữ lại k% phần tử có giá trị tuyệt đối lớn nhất, còn lại gán bằng 0.
        """
        if self.compression_ratio >= 1.0:
            return vector
            
        # Tính số lượng phần tử cần giữ
        k = int(vector.size * self.compression_ratio)
        if k == 0: return np.zeros_like(vector)
        
        # Tìm ngưỡng giá trị (threshold)
        # np.argpartition giúp tìm top-k nhanh hơn sort
        abs_v = np.abs(vector)
        idx = np.argpartition(abs_v.flatten(), -k)[-k:]
        threshold = abs_v.flatten()[idx[0]]
        
        # Tạo mask: 1 nếu >= threshold, 0 nếu < threshold
        mask = (abs_v >= threshold).astype(vector.dtype)
        
        return vector * mask

    def aggregate(self, models: list[P2PFLModel]) -> P2PFLModel:
        if not models:
            raise NoModelsToAggregateError("No models received")

        # -----------------------------
        # 1. Tìm model của chính mình (Local Model)
        # -----------------------------
        self_model = None
        if not hasattr(self, 'addr'):
             # Trong thực tế, bạn cần gán self.addr trước khi gọi aggregate
             raise AttributeError("Aggregator needs 'self.addr' to identify local model.")

        for m in models:
            if self.addr in m.get_contributors():
                self_model = m
                break
        
        if self_model is None:
            raise RuntimeError("Self model not found.")

        # Khởi tạo Memory nếu lần đầu chạy
        params_current = self_model.get_parameters()
        if self.addr not in self.estimates:
            self.estimates[self.addr] = [np.copy(p) for p in params_current]
            self.error_buffer = [np.zeros_like(p) for p in params_current]

        # -----------------------------
        # 2. Xử lý dữ liệu đến (Incoming Compression Simulation)
        # -----------------------------
        
        for m in models:
            contributors = m.get_contributors()
            if not contributors: continue
            sender_id = contributors[0]
            
            # Bỏ qua chính mình ở bước này
            if sender_id == self.addr: continue
            
            # Nếu gặp hàng xóm mới, khởi tạo ước lượng bằng 0
            if sender_id not in self.estimates:
                self.estimates[sender_id] = [np.zeros_like(p) for p in params_current]
            
            # Cập nhật memory của hàng xóm
            # Thực tế: estimates[j] += decompress(message)
            # Giả định ở đây: models chứa tham số mới nhất của họ
            self.estimates[sender_id] = [np.copy(p) for p in m.get_parameters()]

        # -----------------------------
        # 3. Thuật toán CHOCO-SGD
        # -----------------------------
        
        # Bước A: Local Gradient Step
        # x_{t+1/2} = x_t - lr * grad
        grad_dict = self_model.get_gradients()
        # Ensure gradients are in the same order as parameters and convert to numpy
        grad_list = [grad_dict[name].cpu().numpy() for name, _ in self_model.model.named_parameters()]
        x_temp = [p - self.lr * g for p, g in zip(params_current, grad_list)]
        
        # Bước B: Consensus Step (Đồng thuận dựa trên Memory)
        # adjustment = sum(x_hat_j - x_hat_i)
        
        x_hat_self = self.estimates[self.addr]
        adjustment = [np.zeros_like(p) for p in x_temp]
        
        neighbor_count = 0
        for neighbor_id, x_hat_neighbor in self.estimates.items():
            if neighbor_id == self.addr: continue
            
            neighbor_count += 1
            for l in range(len(adjustment)):
                # Cộng dồn sự chênh lệch giữa ước lượng hàng xóm và ước lượng của mình
                adjustment[l] += (x_hat_neighbor[l] - x_hat_self[l])
        
        # Áp dụng đồng thuận: x_{t+1} = x_{t+1/2} + gamma * adjustment
        # Hệ số thường được chia trung bình hoặc giữ nguyên tùy biến thể
        # Ở đây dùng trung bình để ổn định: gamma / degree
        w_consensus = self.consensus_step / max(1, neighbor_count) if neighbor_count > 0 else 0
        
        new_params = []
        for l in range(len(x_temp)):
            new_params.append(x_temp[l] + w_consensus * adjustment[l])

        # -----------------------------
        # 4. Chuẩn bị gửi đi (Outgoing Compression & Error Compensation)
        # -----------------------------
        # Phần này cập nhật self.estimates[self.addr] để dùng cho vòng sau
        # q_i = Compress(x_{t+1} - x_hat_i + error)
        
        for l in range(len(new_params)):
            # Tính sự thay đổi mong muốn
            delta = new_params[l] - self.estimates[self.addr][l]
            
            # Cộng bù lỗi từ vòng trước
            delta_with_error = delta + self.error_buffer[l]
            
            # Nén
            compressed_delta = self._compress(delta_with_error)
            
            # Cập nhật ước lượng về chính mình (đây là cái hàng xóm sẽ thấy)
            self.estimates[self.addr][l] += compressed_delta
            
            # Tính sai số mới để bù vào vòng sau: e = (delta + e) - q
            self.error_buffer[l] = delta_with_error - compressed_delta

        # Trả về model mới
        return self_model.build_copy(
            params=new_params,
            gradients_estimate=None,
            num_samples=self_model.get_num_samples(),
            contributors=[self.addr],
        )