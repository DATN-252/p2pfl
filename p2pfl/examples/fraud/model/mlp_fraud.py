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

"""Refactored efficient MLP model for fraud detection."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from torchmetrics import Precision, Recall, F1Score, Accuracy

from p2pfl.learning.frameworks.pytorch.lightning_model import LightningModel
from p2pfl.settings import Settings
from p2pfl.utils.seed import set_seed


class FraudDetectionMLP(LightningModule):
    """Efficient MLP for fraud detection with BatchNorm and Weighted Loss."""

    def __init__(self, input_size: int = 15, hidden_size: int = 256, learning_rate: float = 0.001, pos_weight: float = 1.0):
        super().__init__()
        set_seed(Settings.general.SEED, "pytorch")
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        
        # Using LayerNorm instead of BatchNorm for better stability with imbalanced classes
        self.model = nn.Sequential(
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
            
            nn.Linear(64, 1)
        )

        self.register_buffer("pos_weight_tensor", torch.tensor([pos_weight]))

        # Metrics
        self.accuracy = Accuracy(task="binary")
        self.precision = Precision(task="binary")
        self.recall = Recall(task="binary")
        self.f1 = F1Score(task="binary")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """Training step with weighted loss."""
        x = batch["features"]
        y = batch["label"].float().unsqueeze(1) if batch["label"].dim() == 1 else batch["label"].float()
        
        y_hat_logits = self(x)
        loss = F.binary_cross_entropy_with_logits(y_hat_logits, y, pos_weight=self.pos_weight_tensor)
        
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        x = batch["features"]
        y = batch["label"].float().unsqueeze(1) if batch["label"].dim() == 1 else batch["label"].float()
        
        y_hat_logits = self(x)
        loss = F.binary_cross_entropy_with_logits(y_hat_logits, y, pos_weight=self.pos_weight_tensor)
        self.log("val_loss", loss, prog_bar=True)

    def test_step(self, batch, batch_idx):
        """Test step with full metrics log."""
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
    
    # Ensure input_size matches updated transforms (15 features)
    if "input_size" not in kwargs:
        kwargs["input_size"] = 15
    
    # Initialize the core MLP
    mlp_model = FraudDetectionMLP(**kwargs)
    
    # Wrap it in LightningModel and pass compression
    return LightningModel(mlp_model, compression=compression)
