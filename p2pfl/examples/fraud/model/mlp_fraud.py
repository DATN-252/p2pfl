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

"""Complete BalanceFL implementation for fraud detection based on IPSN'22 paper."""

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
    """
    Two-stage balanced sampler for long-tail distributions.
    1. Uniformly select a class from the available classes.
    2. Uniformly select a sample from that class.
    """
    def __init__(self, labels: torch.Tensor):
        self.labels = labels
        self.classes = torch.unique(labels).tolist()
        self.class_indices = {
            int(c): torch.where(labels == c)[0] 
            for c in self.classes
        }
        self.max_samples = max(len(idx) for idx in self.class_indices.values())
        self.total_size = self.max_samples * len(self.classes)

    def __iter__(self):
        indices = []
        for _ in range(self.total_size):
            # Stage 1: Uniformly select a class
            c = random.choice(self.classes)
            # Stage 2: Uniformly select a sample from that class
            idx = random.choice(self.class_indices[c])
            indices.append(idx.item())
        return iter(indices)

    def __len__(self):
        return self.total_size

class FraudDetectionMLP(LightningModule):
    """Refactored MLP for fraud detection implementing all BalanceFL techniques (IPSN'22)."""

    def __init__(self, input_size: int = 12, hidden_size: int = 256, learning_rate: float = 0.001, 
                 alpha: float = 1.0, beta: float = 0.05, temperature: float = 2.0):
        super().__init__()
        set_seed(Settings.general.SEED, "pytorch")
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        
        # BalanceFL Hyperparameters
        self.alpha = alpha  # Weight for L_KD
        self.beta = beta    # Weight for L_reg
        self.temperature = temperature
        
        # Task 2: Refactor architecture into feature_extractor and classifier
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

        # Global Model (Teacher) for Knowledge Inheritance
        self.teacher_model = None
        self.absent_classes = []
        
        # Metrics
        self.accuracy = Accuracy(task="binary")
        self.precision = Precision(task="binary")
        self.recall = Recall(task="binary")
        self.f1 = F1Score(task="binary")
        
        # Store augmentation probabilities
        self.p_aug_dict: Dict[int, float] = {}

    def on_train_start(self):
        """
        At the start of each local training round, the current model (received from server)
        is deep-copied to serve as the 'Global Model' (Teacher) for Knowledge Inheritance.
        """
        self.teacher_model = copy.deepcopy(self)
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False

    def forward(self, x: Optional[torch.Tensor], perturbed_h: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with optional feature injection."""
        if perturbed_h is not None:
            return self.classifier(perturbed_h)
            
        if x is not None:
            h = self.feature_extractor(x)
            return self.classifier(h)
        
        raise ValueError("Either x or perturbed_h must be provided.")

    def training_step(self, batch, batch_idx):
        """Implement complete BalanceFL local training objective."""
        x = batch["features"]
        y_logits_target = batch["is_fraud"].float().unsqueeze(1)
        y_labels = batch["is_fraud"].long().squeeze()
        
        # 1. Feature Extraction (Local Model)
        h = self.feature_extractor(x)
        z_local = self.classifier(h)
        
        # --- Task 2.2: Feature-Level Data Augmentation ---
        # Step 1: Calculate Global Covariance Matrix
        sigma = BalanceFL.calculate_global_covariance(h, y_labels)
        
        # Step 2: Calculate Augmentation Probabilities (Once per Node round)
        if not self.p_aug_dict:
            self.p_aug_dict = BalanceFL.calculate_max_imbalance_probabilities(y_labels)
            # Identify absent classes for KD
            all_classes = [0, 1]
            present_classes = torch.unique(y_labels).tolist()
            self.absent_classes = [c for c in all_classes if c not in present_classes]

        # Step 3: Noise Injection (L_CE is calculated on augmented features)
        z_aug = z_local
        if sigma is not None:
            sigma += torch.eye(sigma.size(0), device=self.device) * 1e-6
            try:
                dist = torch.distributions.MultivariateNormal(
                    torch.zeros(h.size(1), device=self.device), 
                    sigma
                )
                
                h_augmented = h.clone()
                for i in range(h.size(0)):
                    cls_i = int(y_labels[i].item())
                    p_aug = self.p_aug_dict.get(cls_i, 0.0)
                    
                    if random.random() < p_aug:
                        epsilon = dist.sample()
                        h_augmented[i] = h[i] + epsilon
                
                z_aug = self.classifier(h_augmented)
            except Exception:
                z_aug = z_local
        
        # --- Total Local Loss (Task 3) ---
        
        # 1. Cross-Entropy Loss (L_CE)
        loss_ce = F.binary_cross_entropy_with_logits(z_aug, y_logits_target)

        # 2. Knowledge Inheritance (L_KD - for absent classes)
        loss_kd = torch.tensor(0.0, device=self.device)
        if self.teacher_model is not None and self.absent_classes:
            with torch.no_grad():
                z_global = self.teacher_model(x)
            
            # Binary version of KL Divergence using sigmoid distributions [1-p, p]
            p_g = torch.sigmoid(z_global / self.temperature)
            p_l = torch.sigmoid(z_local / self.temperature)
            
            dist_g = torch.stack([1 - p_g, p_g], dim=1).squeeze()
            dist_l = torch.stack([1 - p_l, p_l], dim=1).squeeze()
            
            # Ensure safe log
            kl = F.kl_div(dist_l.log(), dist_g, reduction='none')
            for c in self.absent_classes:
                loss_kd += kl[:, c].mean()

        # 3. Smooth Regularization (L_reg - Negative Entropy)
        p_aug_softmax = torch.sigmoid(z_aug)
        dist_aug = torch.stack([1 - p_aug_softmax, p_aug_softmax], dim=1).squeeze()
        loss_reg = (dist_aug * torch.log(dist_aug + 1e-9)).sum(dim=1).mean()

        # Final Objective: L_total = L_CE + alpha * L_KD + beta * L_reg
        loss_total = loss_ce + self.alpha * loss_kd + self.beta * loss_reg
        
        self.log("train_loss", loss_total, prog_bar=True)
        self.log("l_ce", loss_ce)
        self.log("l_kd", loss_kd)
        self.log("l_reg", loss_reg)
        
        return loss_total

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        x = batch["features"]
        y = batch["is_fraud"].float().unsqueeze(1)
        
        y_hat_logits = self(x)
        loss = F.binary_cross_entropy_with_logits(y_hat_logits, y)
        self.log("val_loss", loss, prog_bar=True)

    def test_step(self, batch, batch_idx):
        """Test step with metrics."""
        x = batch["features"]
        y = batch["is_fraud"].float().unsqueeze(1)
        
        y_hat_logits = self(x)
        y_hat_probs = torch.sigmoid(y_hat_logits)
        
        self.log("test_accuracy", self.accuracy(y_hat_probs, y), prog_bar=True)
        self.log("test_recall", self.recall(y_hat_probs, y), prog_bar=True)
        self.log("test_f1", self.f1(y_hat_probs, y), prog_bar=True)

    def configure_optimizers(self):
        """Configure optimizer."""
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


def model_build_fn(**kwargs) -> LightningModel:
    """Build function for BalanceFL model."""
    compression = kwargs.pop("compression", None)
    input_size = kwargs.pop("input_size", 12)
    mlp_model = FraudDetectionMLP(input_size=input_size, **kwargs)
    return LightningModel(mlp_model, compression=compression)
