#
# This file is part of the federated_learning_p2p (p2pfl) distribution
# (see https://github.com/pguijas/p2pfl).
#

"""ResNet-50 based model for AID Scene Classification with Focal Loss."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from torchvision import models
from torchmetrics import Accuracy, F1Score

from p2pfl.learning.frameworks.pytorch.lightning_model import LightningModel
from p2pfl.settings import Settings
from p2pfl.utils.seed import set_seed


class FocalLoss(nn.Module):
    """
    Focal Loss for Multi-class classification.
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class ResNetAID(LightningModule):
    """ResNet-50 for AID Scene Classification (30 classes)."""

    def __init__(self, num_classes: int = 30, learning_rate: float = 1e-4, alpha: float = 1.0, gamma: float = 2.0):
        super().__init__()
        set_seed(Settings.general.SEED, "pytorch")
        self.save_hyperparameters()
        self.learning_rate = learning_rate

        # Load Pre-trained ResNet-50
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        # Replace the final fully connected layer
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

        # Loss & Metrics
        self.criterion = FocalLoss(alpha=alpha, gamma=gamma)
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.f1 = F1Score(task="multiclass", num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch["features"], batch["label"]
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        
        acc = self.train_acc(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["features"], batch["label"]
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        
        acc = self.val_acc(y_hat, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch["features"], batch["label"]
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        
        acc = self.test_acc(y_hat, y)
        f1 = self.f1(y_hat, y)
        
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_accuracy", acc, prog_bar=True)
        self.log("test_f1", f1, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


def model_build_fn(**kwargs) -> LightningModel:
    """Build function for the AID ResNet model."""
    compression = kwargs.pop("compression", None)
    
    # Default values for AID
    if "num_classes" not in kwargs:
        kwargs["num_classes"] = 30
    
    core_model = ResNetAID(**kwargs)
    return LightningModel(core_model, compression=compression)
