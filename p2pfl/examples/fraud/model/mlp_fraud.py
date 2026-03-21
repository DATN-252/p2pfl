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

"""Refactored efficient MLP model for fraud detection with BalanceFL techniques."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from typing import Dict, Optional
from torch.utils.data import Sampler
from lightning import LightningModule
from torchmetrics import Precision, Recall, F1Score, Accuracy

from p2pfl.learning.frameworks.pytorch.lightning_model import LightningModel
from p2pfl.settings import Settings
from p2pfl.utils.seed import set_seed
from p2pfl.examples.fraud.model.balance_fl import BalanceFL

# Task 1: Implement Class Balanced Sampling (Two-Stage Process)
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
        # align with the majority class
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
    """Refactored MLP for fraud detection with Feature-Level Data Augmentation."""

    def __init__(self, input_size: int = 12, hidden_size: int = 256, learning_rate: float = 0.001, pos_weight: float = 1.0):
        super().__init__()
        set_seed(Settings.general.SEED, "pytorch")
        self.save_hyperparameters()
        self.learning_rate = learning_rate

        # Task 2: Refactor into feature_extractor and classifier
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

        self.register_buffer("pos_weight_tensor", torch.tensor([pos_weight]))
        
        # Store augmentation probabilities
        self.p_aug_dict: Dict[int, float] = {}

        # Metrics
        self.accuracy = Accuracy(task="binary")
        self.precision = Precision(task="binary")
        self.recall = Recall(task="binary")
        self.f1 = F1Score(task="binary")

    def forward(self, x: Optional[torch.Tensor], perturbed_h: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with optional feature injection."""
        if perturbed_h is not None:
            return self.classifier(perturbed_h)
            
        if x is not None:
            h = self.feature_extractor(x)
            return self.classifier(h)
        
        raise ValueError("Either x or perturbed_h must be provided.")

    def training_step(self, batch, batch_idx):
        """Task 3: Implement Feature-Level Data Augmentation in the Training Loop."""
        x = batch["features"]
        y_logits_target = batch["label"].float().unsqueeze(1) if batch["label"].dim() == 1 else batch["label"].float()
        y_labels = batch["label"].long().squeeze()
        
        # Step C: Forward pass - Get original features h
        h = self.feature_extractor(x)
        
        # Step A: Calculate overall covariance matrix (Sigma) of the feature vectors
        sigma = BalanceFL.calculate_global_covariance(h, y_labels)
        
        # Step B: Calculate class-specific augmentation probability p_aug
        if not self.p_aug_dict:
            # Note: In practice, this should be pre-calculated from global labels if possible.
            # Here we demonstrate the logic using batch labels as a fallback.
            self.p_aug_dict = BalanceFL.calculate_max_imbalance_probabilities(y_labels)

        # Step C: Apply noise injection based on p_aug and Multivariate Gaussian N(0, Sigma)
        if sigma is not None:
            # Regularize Sigma for stability
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
                
                y_hat_logits = self.classifier(h_augmented)
            except Exception:
                y_hat_logits = self.classifier(h)
        else:
            y_hat_logits = self.classifier(h)

        loss = F.binary_cross_entropy_with_logits(y_hat_logits, y_logits_target, pos_weight=self.pos_weight_tensor)
        
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step without augmentation."""
        x = batch["features"]
        y = batch["label"].float().unsqueeze(1) if batch["label"].dim() == 1 else batch["label"].float()
        
        y_hat_logits = self(x)
        loss = F.binary_cross_entropy_with_logits(y_hat_logits, y, pos_weight=self.pos_weight_tensor)
        self.log("val_loss", loss, prog_bar=True)

    def test_step(self, batch, batch_idx):
        """Test step."""
        x = batch["features"]
        y = batch["label"].float().unsqueeze(1) if batch["label"].dim() == 1 else batch["label"].float()
        
        y_hat_logits = self(x)
        loss = F.binary_cross_entropy_with_logits(y_hat_logits, y, pos_weight=self.pos_weight_tensor)
        y_hat_probs = torch.sigmoid(y_hat_logits)
        
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_accuracy", self.accuracy(y_hat_probs, y), prog_bar=True)
        self.log("test_precision", self.precision(y_hat_probs, y), prog_bar=True)
        self.log("test_recall", self.recall(y_hat_probs, y), prog_bar=True)
        self.log("test_f1", self.f1(y_hat_probs, y), prog_bar=True)

    def configure_optimizers(self):
        """Configure optimizer."""
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


def model_build_fn(**kwargs) -> LightningModel:
    """
    Build function to create the fraud detection model.
    """
    # Separate compression from other parameters
    compression = kwargs.pop("compression", None)

    # Ensure input_size matches updated transforms (12 features)
    if "input_size" not in kwargs:
        kwargs["input_size"] = 12

    # Initialize the core MLP
    mlp_model = FraudDetectionMLP(**kwargs)
    
    # Wrap it in LightningModel and pass compression
    return LightningModel(mlp_model, compression=compression)
