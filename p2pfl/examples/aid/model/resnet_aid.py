#
# This file is part of the federated_learning_p2p (p2pfl) distribution
# (see https://github.com/pguijas/p2pfl).
#

"""Optimized ResNet model for AID Scene Classification."""

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
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()

class ResNetAID(LightningModule):
    def __init__(self, model_type: str = "resnet18", num_classes: int = 30, 
                 learning_rate: float = 1e-4, alpha: float = 1.0, gamma: float = 2.0,
                 freeze_backbone: bool = True):
        super().__init__()
        set_seed(Settings.general.SEED, "pytorch")
        self.save_hyperparameters()
        self.learning_rate = learning_rate

        # Load Pre-trained Model (ResNet-18 is much faster)
        if model_type == "resnet18":
            self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        else:
            self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        # Freeze layers if requested
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

        # Replace the final fully connected layer (always trainable)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)
        for param in self.model.fc.parameters():
            param.requires_grad = True

        # Loss & Metrics
        self.criterion = FocalLoss(alpha=alpha, gamma=gamma)
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch["features"], batch["label"]
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.train_acc(y_hat, y), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["features"], batch["label"]
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_acc(y_hat, y), prog_bar=True)

    def configure_optimizers(self):
        # Only optimize parameters that require gradients
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        return torch.optim.Adam(trainable_params, lr=self.learning_rate)

def model_build_fn(**kwargs) -> LightningModel:
    compression = kwargs.pop("compression", None)
    if "num_classes" not in kwargs: kwargs["num_classes"] = 30
    # Default to resnet18 and freeze for speed
    if "model_type" not in kwargs: kwargs["model_type"] = "resnet18"
    if "freeze_backbone" not in kwargs: kwargs["freeze_backbone"] = True
    
    core_model = ResNetAID(**kwargs)
    return LightningModel(core_model, compression=compression)
