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
from torchmetrics import Accuracy, Precision, Recall, F1Score
from p2pfl.settings import Settings
from p2pfl.utils.seed import set_seed
from p2pfl.learning.frameworks.pytorch.lightning_model import LightningModel


class FraudDetectionMLP(LightningModule):
    """Simple MLP for fraud detection on tabular data."""

    def __init__(
        self, 
        input_size: int = 8, 
        hidden_sizes: list[int] = [128], 
        learning_rate: float = 0.001,
    ):
        """
        Initialize the MLP model.

        Args:
            input_size: Number of input features (7 numeric features from transforms).
            hidden_sizes: Number of hidden units.
            learning_rate: Learning rate for optimizer.

        """
        super().__init__()
        self.save_hyperparameters()
        set_seed(Settings.general.SEED, "pytorch")
        self.learning_rate = learning_rate

        # Metric
        self.test_acc = Accuracy(task="binary")
        self.test_prec = Precision(task="binary")
        self.test_rec = Recall(task="binary")
        self.test_f1 = F1Score(task="binary")

        # Network layers
        layers = torch.nn.ModuleList()
        # Input layer
        layers.append(torch.nn.Linear(input_size, hidden_sizes[0]))
        layers.append(torch.nn.ReLU())

        # Hidden layers
        for i in range(len(hidden_sizes) - 1):
            layers.append(torch.nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            layers.append(torch.nn.ReLU())

        # Output layer
        layers.append(torch.nn.Linear(hidden_sizes[-1], 1))

        self.register_buffer('pos_weight', torch.tensor([184]))
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the MLP."""
        x = torch.flatten(x, start_dim=1)

        return self.model(x)

    def training_step(self, batch, batch_idx):
        """Training step."""
        x = batch["features"]
        y = batch["label"]
        y = y.float().unsqueeze(1) if y.dim() == 1 else y.float()
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        x = batch["features"]
        y = batch["label"]
        y = y.float().unsqueeze(1) if y.dim() == 1 else y.float()
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("val_loss", loss, prog_bar=True)

    def test_step(self, batch, batch_idx):
        """Test step."""
        x = batch["features"]
        y = batch["label"]
        y = y.float().unsqueeze(1) if y.dim() == 1 else y.float()
        y_hat = self(x)
        loss = self.criterion(y_hat, y)

        probs = torch.sigmoid(y_hat)
        # Calculate accuracy
        preds = (probs > 0.1).float()
        acc = self.test_acc(preds, y)
        prec = self.test_prec(preds, y)
        rec = self.test_rec(preds, y)
        f1 = self.test_f1(preds, y)

        self.log("\ntest_loss", loss, prog_bar=True)
        self.log("test_acc\n", acc, prog_bar=True)
        self.log("test_precision\n", prec, prog_bar=True)
        self.log("test_recall\n", rec, prog_bar=True)
        self.log("test_f1\n", f1, prog_bar=True)
        return loss

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
    return LightningModel(FraudDetectionMLP(input_size=8, hidden_sizes=[128]))
