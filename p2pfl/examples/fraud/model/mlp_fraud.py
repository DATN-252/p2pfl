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

from p2pfl.learning.frameworks.pytorch.lightning_model import LightningModel


class FraudDetectionMLP(LightningModule):
    """Simple MLP for fraud detection on tabular data."""

    def __init__(self, input_size: int = 7, hidden_size: int = 128, learning_rate: float = 0.001):
        """
        Initialize the MLP model.

        Args:
            input_size: Number of input features (7 numeric features from transforms).
            hidden_size: Number of hidden units.
            learning_rate: Learning rate for optimizer.

        """
        super().__init__()
        self.save_hyperparameters()

        # Network layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)

        self.learning_rate = learning_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc4(x))
        return x

    def training_step(self, batch, batch_idx):
        """Training step."""
        x = batch["features"]
        y = batch["label"]
        y = y.float().unsqueeze(1) if y.dim() == 1 else y.float()
        y_hat = self(x)
        loss = F.binary_cross_entropy(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        x = batch["features"]
        y = batch["label"]
        y = y.float().unsqueeze(1) if y.dim() == 1 else y.float()
        y_hat = self(x)
        loss = F.binary_cross_entropy(y_hat, y)
        self.log("val_loss", loss, prog_bar=True)

    def test_step(self, batch, batch_idx):
        """Test step."""
        x = batch["features"]
        y = batch["label"]
        y = y.float().unsqueeze(1) if y.dim() == 1 else y.float()
        y_hat = self(x)
        loss = F.binary_cross_entropy(y_hat, y)

        # Calculate accuracy
        y_pred = (y_hat > 0.5).float()
        acc = (y_pred == y).float().mean()

        self.log("test_loss", loss, prog_bar=True)
        self.log("test_accuracy", acc, prog_bar=True)

    def configure_optimizers(self):
        """Configure optimizer."""
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


def model_build_fn(**kwargs) -> LightningModel:
    """
    Build function to create the fraud detection model.

    Args:
        **kwargs: Additional keyword arguments (e.g., compression).

    Returns:
        The fraud detection model wrapped in LightningModel.

    """
    return LightningModel(FraudDetectionMLP(input_size=7, hidden_size=128))
