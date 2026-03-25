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

"""Optimized BalanceFL and Peer-to-Peer Distillation for fraud detection."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from typing import Dict, Optional, List
from torch.utils.data import Sampler
from lightning import LightningModule
from torchmetrics import Precision, Recall, F1Score, Accuracy

from p2pfl.learning.frameworks.pytorch.lightning_model import LightningModel
from p2pfl.settings import Settings
from p2pfl.utils.seed import set_seed
from p2pfl.examples.fraud.model.balance_fl import BalanceFL

# --- Task 2.1: Class Balanced Sampling ---
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
    def __init__(self, input_size: int = 12, hidden_size: int = 256, learning_rate: float = 0.001, 
                 beta: float = 0.05, peer_alpha: float = 0.5, temperature: float = 2.0):
        super().__init__()
        set_seed(Settings.general.SEED, "pytorch")
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        
        self.beta = beta          
        self.peer_alpha = peer_alpha 
        self.temperature = temperature
        
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

        self.neighbor_teachers = [] 
        
        self.accuracy = Accuracy(task="binary")
        self.precision = Precision(task="binary")
        self.recall = Recall(task="binary")
        self.f1 = F1Score(task="binary")
        self.p_aug_dict: Dict[int, float] = {}

    def on_train_start(self):
        # Peer Teachers - Move to device ONCE at start of round
        for neighbor in self.neighbor_teachers:
            neighbor.to(self.device)
            neighbor.eval()
            for param in neighbor.parameters():
                param.requires_grad = False

    def forward(self, x: Optional[torch.Tensor], perturbed_h: Optional[torch.Tensor] = None) -> torch.Tensor:
        if perturbed_h is not None:
            return self.classifier(perturbed_h)
        if x is not None:
            h = self.feature_extractor(x)
            return self.classifier(h)
        raise ValueError("Inputs missing.")

    def training_step(self, batch, batch_idx):
        x = batch["features"]
        # Sử dụng 'label' thay vì 'is_fraud' vì transform trả về 'label'
        y_target = batch["label"].float().unsqueeze(1) if batch["label"].dim() == 1 else batch["label"].float()
        y_labels = batch["label"].long().squeeze()
        
        h = self.feature_extractor(x)
        z_local = self.classifier(h)
        
        # --- 1. BalanceFL: Feature Augmentation ---
        sigma = BalanceFL.calculate_global_covariance(h, y_labels)
        if not self.p_aug_dict:
            self.p_aug_dict = BalanceFL.calculate_max_imbalance_probabilities(y_labels)

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

        # --- 2. Losses ---
        loss_ce = F.binary_cross_entropy_with_logits(z_aug, y_target)

        # Peer KD (Neighbor Distillation) - Optimized (no .to(device) here)
        loss_peer_kd = torch.tensor(0.0, device=self.device)
        if self.neighbor_teachers:
            with torch.no_grad():
                neighbor_logits = [neighbor(x) for neighbor in self.neighbor_teachers]
                z_neighbors_ensemble = torch.mean(torch.stack(neighbor_logits), dim=0)
            
            p_peer = torch.sigmoid(z_neighbors_ensemble / self.temperature)
            p_local = torch.sigmoid(z_local / self.temperature)
            dist_peer = torch.stack([1 - p_peer, p_peer], dim=1).squeeze()
            dist_local = torch.stack([1 - p_local, p_local], dim=1).squeeze()
            loss_peer_kd = F.kl_div(dist_local.log(), dist_peer, reduction='batchmean')

        # Smooth Regularization
        p_s = torch.sigmoid(z_aug)
        dist_s = torch.stack([1 - p_s, p_s], dim=1).squeeze()
        loss_reg = (dist_s * torch.log(dist_s + 1e-9)).sum(dim=1).mean()

        loss_total = loss_ce + (self.beta * loss_reg) + (self.peer_alpha * loss_peer_kd)
        
        self.log("train_loss", loss_total, prog_bar=True)
        return loss_total

    def validation_step(self, batch, batch_idx):
        x = batch["features"]
        y = batch["label"].float().unsqueeze(1)
        y_hat = self(x)
        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        probs = torch.sigmoid(y_hat)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_f1", self.f1(probs, y))

    def test_step(self, batch, batch_idx):
        x = batch["features"]
        y = batch["label"].float().unsqueeze(1)
        y_hat = self(x)
        probs = torch.sigmoid(y_hat)
        self.log("test_accuracy", self.accuracy(probs, y), prog_bar=True)
        self.log("test_precision", self.precision(probs, y), prog_bar=True)
        self.log("test_recall", self.recall(probs, y), prog_bar=True)
        self.log("test_f1", self.f1(probs, y), prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

def model_build_fn(**kwargs) -> LightningModel:
    compression = kwargs.pop("compression", None)
    input_size = kwargs.pop("input_size", 12)
    mlp_model = FraudDetectionMLP(input_size=input_size, **kwargs)
    return LightningModel(mlp_model, compression=compression)
