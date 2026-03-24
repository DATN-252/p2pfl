#
# This file is part of the federated_learning_p2p (p2pfl) distribution
# (see https://github.com/pguijas/p2pfl).
# Copyright (c) 2024 Pedro Guijas Bravo.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#

"""Complete BalanceFL and Peer-to-Peer Distillation implementation for fraud detection."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import copy
from typing import Dict, Optional, List
from torch.utils.data import Sampler
from lightning import LightningModule
from torchmetrics import Precision, Recall, F1Score, Accuracy

from p2pfl.learning.frameworks.pytorch.lightning_model import LightningModel
from p2pfl.settings import Settings
from p2pfl.utils.seed import set_seed
from p2pfl.examples.fraud.model.balance_fl import BalanceFL

# --- Task 2.1: Class Balanced Sampling (Two-Stage Process) ---
class TwoStageBalancedSampler(Sampler):
    def __init__(self, labels: torch.Tensor):
        self.labels = labels
        self.classes = torch.unique(labels).tolist()
        self.class_indices = {int(c): torch.where(labels == c)[0] for c in self.classes}
        self.max_samples = max(len(idx) for idx in self.class_indices.values())
        self.total_size = self.max_samples * len(self.classes)

    def __iter__(self):
        indices = []
        for _ in range(self.total_size):
            c = random.choice(self.classes)
            idx = random.choice(self.class_indices[c])
            indices.append(idx.item())
        return iter(indices)

    def __len__(self):
        return self.total_size

class FraudDetectionMLP(LightningModule):
    """Refactored MLP supporting BalanceFL and Peer-to-Peer Neighbor Distillation."""

    def __init__(self, input_size: int = 12, hidden_size: int = 256, learning_rate: float = 0.001, 
                 alpha: float = 1.0, beta: float = 0.05, peer_alpha: float = 0.5, temperature: float = 2.0,
                 tversky_alpha: float = 0.7, tversky_beta: float = 0.3):
        super().__init__()
        set_seed(Settings.general.SEED, "pytorch")
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        
        # BalanceFL & Peer KD Hyperparameters
        self.alpha = alpha        # Weight for Global KD (absent classes)
        self.beta = beta          # Weight for Smooth Regularization
        self.peer_alpha = peer_alpha # Weight for Neighbor Distillation
        self.temperature = temperature
        
        # Tversky Loss Hyperparameters
        self.tversky_alpha = tversky_alpha
        self.tversky_beta = tversky_beta
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.1),
        )
        self.classifier = nn.Linear(64, 1)

        # Global Model Container
        self.teacher_model_container = [None]
        # Peer Models Container (Neighbors)
        self.neighbor_teachers = [] 
        
        self.absent_classes = []
        self.accuracy = Accuracy(task="binary")
        self.precision = Precision(task="binary")
        self.recall = Recall(task="binary")
        self.f1 = F1Score(task="binary")
        self.p_aug_dict: Dict[int, float] = {}

    def on_train_start(self):
        """Pre-prepare the global consensus teacher."""
        teacher = copy.deepcopy(self)
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad = False
        self.teacher_model_container[0] = teacher

    def forward(self, x: Optional[torch.Tensor], perturbed_h: Optional[torch.Tensor] = None) -> torch.Tensor:
        if perturbed_h is not None:
            return self.classifier(perturbed_h)
        if x is not None:
            h = self.feature_extractor(x)
            return self.classifier(h)
        raise ValueError("Inputs missing.")

    def training_step(self, batch, batch_idx):
        x = batch["features"]
        y_logits_target = batch["label"].float().unsqueeze(1)
        y_labels = batch["label"].long().squeeze()
        
        h = self.feature_extractor(x)
        z_local = self.classifier(h)
        
        # --- 1. Feature Augmentation (BalanceFL) ---
        sigma = BalanceFL.calculate_global_covariance(h, y_labels)
        if not self.p_aug_dict:
            self.p_aug_dict = BalanceFL.calculate_max_imbalance_probabilities(y_labels)
            present_classes = torch.unique(y_labels).tolist()
            self.absent_classes = [c for c in [0, 1] if c not in present_classes]

        z_aug = z_local
        if sigma is not None:
            sigma += torch.eye(sigma.size(0), device=self.device) * 1e-6
            try:
                dist = torch.distributions.MultivariateNormal(torch.zeros(h.size(1), device=self.device), sigma)
                h_augmented = h.clone()
                for i in range(h.size(0)):
                    if random.random() < self.p_aug_dict.get(int(y_labels[i].item()), 0.0):
                        h_augmented[i] += dist.sample()
                z_aug = self.classifier(h_augmented)
            except: pass

        # --- 2. Tversky Classification Loss (instead of BCE + summations) ---
        loss_tversky = BalanceFL.compute_tversky_loss(
            z_aug, 
            y_logits_target, 
            alpha=self.tversky_alpha, 
            beta=self.tversky_beta
        )

        # --- 3. Global Knowledge Inheritance (BalanceFL L_KD) ---
        loss_global_kd = torch.tensor(0.0, device=self.device)
        global_teacher = self.teacher_model_container[0]
        if global_teacher is not None and self.absent_classes:
            with torch.no_grad():
                z_global = global_teacher(x)
            p_g = torch.sigmoid(z_global / self.temperature)
            p_l = torch.sigmoid(z_local / self.temperature)
            dist_g = torch.stack([1 - p_g, p_g], dim=1).squeeze()
            dist_l = torch.stack([1 - p_l, p_l], dim=1).squeeze()
            kl = F.kl_div(dist_l.log(), dist_g, reduction='none')
            for c in self.absent_classes:
                loss_global_kd += kl[:, c].mean()

        # --- 4. Peer-to-Peer Distillation (Neighbor Distillation) ---
        loss_peer_kd = torch.tensor(0.0, device=self.device)
        if self.neighbor_teachers:
            neighbor_logits_list = []
            with torch.no_grad():
                for neighbor in self.neighbor_teachers:
                    neighbor.to(self.device)
                    neighbor_logits_list.append(neighbor(x))
            
            # Ensemble neighbors (Tri thức tập thể của hàng xóm)
            z_neighbors_ensemble = torch.mean(torch.stack(neighbor_logits_list), dim=0)
            
            p_peer = torch.sigmoid(z_neighbors_ensemble / self.temperature)
            p_local = torch.sigmoid(z_local / self.temperature)
            dist_peer = torch.stack([1 - p_peer, p_peer], dim=1).squeeze()
            dist_local = torch.stack([1 - p_local, p_local], dim=1).squeeze()
            
            # Chưng cất từ tất cả các lớp của hàng xóm (không chỉ lớp bị thiếu)
            loss_peer_kd = F.kl_div(dist_local.log(), dist_peer, reduction='batchmean')

        # --- 5. Smooth Regularization ---
        p_softmax = torch.sigmoid(z_aug)
        dist_s = torch.stack([1 - p_softmax, p_softmax], dim=1).squeeze()
        loss_reg = (dist_s * torch.log(dist_s + 1e-9)).sum(dim=1).mean()

        # --- Total Objective ---
        # As requested: replace adding losses with Tversky Loss
        # We use Tversky Loss as the primary objective which naturally handles imbalance
        loss_total = loss_tversky 
        
        self.log("train_loss", loss_total, prog_bar=True)
        self.log("l_tversky", loss_tversky)
        self.log("l_global_kd", loss_global_kd)
        self.log("l_peer_kd", loss_peer_kd)
        self.log("l_reg", loss_reg)
        return loss_total

    def validation_step(self, batch, batch_idx):
        """Validation step with full metrics."""
        x = batch["features"]
        y = batch["label"].float().unsqueeze(1)

        y_hat_logits = self(x)
        loss = BalanceFL.compute_tversky_loss(y_hat_logits, y, alpha=self.tversky_alpha, beta=self.tversky_beta)
        y_hat_probs = torch.sigmoid(y_hat_logits)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_accuracy", self.accuracy(y_hat_probs, y))
        self.log("val_precision", self.precision(y_hat_probs, y))
        self.log("val_recall", self.recall(y_hat_probs, y))
        self.log("val_f1", self.f1(y_hat_probs, y))

    def test_step(self, batch, batch_idx):
        """Test step with full metrics logging."""
        x = batch["features"]
        y = batch["label"].float().unsqueeze(1)

        y_hat_logits = self(x)
        y_hat_probs = torch.sigmoid(y_hat_logits)

        self.log("test_accuracy", self.accuracy(y_hat_probs, y), prog_bar=True)
        self.log("test_precision", self.precision(y_hat_probs, y), prog_bar=True)
        self.log("test_recall", self.recall(y_hat_probs, y), prog_bar=True)
        self.log("test_f1", self.f1(y_hat_probs, y), prog_bar=True)


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

def model_build_fn(**kwargs) -> LightningModel:
    compression = kwargs.pop("compression", None)
    input_size = kwargs.pop("input_size", 12)
    mlp_model = FraudDetectionMLP(input_size=input_size, **kwargs)
    return LightningModel(mlp_model, compression=compression)
