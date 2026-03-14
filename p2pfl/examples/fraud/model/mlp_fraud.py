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

"""MLP model for fraud detection."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from torchmetrics import Precision, Recall, F1Score, Accuracy

from p2pfl.learning.frameworks.pytorch.lightning_model import LightningModel


class FraudDetectionMLP(LightningModule):
    """Simple MLP for fraud detection on tabular data."""

    def __init__(self, input_size: int = 11, hidden_size: int = 128, learning_rate: float = 0.001, pos_weight: float = 1.0):
        """
        Initialize the MLP model.

        Args:
            input_size: Number of input features.
            hidden_size: Number of hidden units.
            learning_rate: Learning rate for optimizer.
            pos_weight: Weight for positive class (fraud).

        """
        super().__init__()
        self.save_hyperparameters()

        # Network layers with BatchNorm for stability
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        
        self.fc2 = nn.Linear(hidden_size, 64)
        self.bn2 = nn.BatchNorm1d(64)
        
        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        
        self.fc4 = nn.Linear(32, 1)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)

        self.learning_rate = learning_rate
        
        self.register_buffer("pos_weight_tensor", torch.tensor([pos_weight]))

        # Metrics
        self.accuracy = Accuracy(task="binary")
        self.precision = Precision(task="binary")
        self.recall = Recall(task="binary")
        self.f1 = F1Score(task="binary")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Simplified forward pass for stability."""
        # Using a safer approach for BatchNorm and skipping complex state checks
        # that can cause deadlocks in multi-threaded Ray environments.
        x = self.fc1(x)
        if x.shape[0] > 1:
            x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        if x.shape[0] > 1:
            x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        if x.shape[0] > 1:
            x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        return self.fc4(x)

    def training_step(self, batch, batch_idx):
        """Training step with weighted loss."""
        x = batch["features"]
        y = batch["label"]
        y = y.float().unsqueeze(1) if y.dim() == 1 else y.float()
        
        y_hat_logits = self(x)
        loss = F.binary_cross_entropy_with_logits(y_hat_logits, y, pos_weight=self.pos_weight_tensor)
        
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        x = batch["features"]
        y = batch["label"]
        y = y.float().unsqueeze(1) if y.dim() == 1 else y.float()
        
        y_hat_logits = self(x)
        loss = F.binary_cross_entropy_with_logits(y_hat_logits, y, pos_weight=self.pos_weight_tensor)
        self.log("val_loss", loss, prog_bar=True)

    def test_step(self, batch, batch_idx):
        """Test step with full metrics."""
        x = batch["features"]
        y = batch["label"]
        y = y.float().unsqueeze(1) if y.dim() == 1 else y.float()
        
        y_hat_logits = self(x)
        loss = F.binary_cross_entropy_with_logits(y_hat_logits, y, pos_weight=self.pos_weight_tensor)

        # Calculate probabilities for metrics
        y_hat_probs = torch.sigmoid(y_hat_logits)
        
        # Calculate metrics
        acc = self.accuracy(y_hat_probs, y)
        prec = self.precision(y_hat_probs, y)
        rec = self.recall(y_hat_probs, y)
        f1 = self.f1(y_hat_probs, y)

        # Log metrics
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_accuracy", acc, prog_bar=True)
        self.log("test_precision", prec, prog_bar=True)
        self.log("test_recall", rec, prog_bar=True)
        self.log("test_f1", f1, prog_bar=True)

    def configure_optimizers(self):
        """Configure optimizer."""
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


def model_build_fn(**kwargs) -> LightningModel:
    """
    Build function to create the fraud detection model.
    """
    # Separate compression from other parameters
    compression = kwargs.pop("compression", None)
    
    # Ensure input_size matches updated transforms
    if "input_size" not in kwargs:
        kwargs["input_size"] = 11
    
    # Initialize the core MLP with the remaining parameters (like pos_weight)
    mlp_model = FraudDetectionMLP(**kwargs)
    
    # Wrap it in LightningModel and pass compression
    return LightningModel(mlp_model, compression=compression)
