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

"""Momentum Learner for Q-DFedAvgM."""

import logging
import traceback
import torch
import numpy as np
from torch.utils.data import DataLoader

from p2pfl.learning.aggregators.aggregator import Aggregator
from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset
from p2pfl.learning.frameworks import Framework
from p2pfl.learning.frameworks.learner import Learner
from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel
from p2pfl.learning.frameworks.pytorch.lightning_dataset import PyTorchExportStrategy
from p2pfl.management.logger import logger
from p2pfl.settings import Settings
from p2pfl.utils.seed import set_seed

class MomentumLearner(Learner):
    """
    Learner that implements the local update rule with Heavy-Ball momentum (Eq 4).
    y^{t,k+1} = y^{t,k} - eta * g^{t,k} + theta * (y^{t,k} - y^{t,k-1})
    """

    def __init__(self, model: P2PFLModel | None = None, data: P2PFLDataset | None = None, aggregator: Aggregator | None = None) -> None:
        super().__init__(model, data, aggregator)
        # Previous local iterate y^{t, k-1}
        self.y_prev: list[torch.Tensor] | None = None
        self.eta = 0.01  # learning rate
        self.theta = 0.9 # momentum factor

    def fit(self, apply_update: bool = True) -> P2PFLModel:
        """Fit the model using manual loop for momentum control."""
        set_seed(Settings.general.SEED, self.get_framework())
        
        # Get PyTorch model and data
        pt_model = self.get_model().get_model()
        pt_data = self.get_data().export(PyTorchExportStrategy, train=True)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pt_model.to(device)
        pt_model.train()

        # Extract parameters for manual update
        params = list(pt_model.parameters())
        
        # Initialize y_prev if it's the first time
        if self.y_prev is None:
            self.y_prev = [p.clone().detach() for p in params]

        # Get eta and theta from model params or defaults
        self.eta = getattr(pt_model, "lr_rate", self.eta)
        self.theta = getattr(pt_model, "momentum", self.theta)

        try:
            for epoch in range(self.epochs): # K local iterations
                for batch in pt_data:
                    # Move batch to device
                    if isinstance(batch, list | tuple):
                        batch = [b.to(device) if isinstance(b, torch.Tensor) else b for b in batch]
                        x, y = batch[0], batch[1]
                    else:
                        x = batch['image'].to(device)
                        y = batch['label'].to(device)

                    # Compute Gradient g^{t,k}
                    pt_model.zero_grad()
                    output = pt_model(x)
                    loss = torch.nn.functional.cross_entropy(output, y)
                    loss.backward()

                    # Heavy-Ball Momentum Update (Equation 4)
                    with torch.no_grad():
                        for i, p in enumerate(params):
                            y_curr = p.clone().detach()
                            # momentum_term = theta * (y_curr - y_prev)
                            momentum_term = self.theta * (y_curr - self.y_prev[i])
                            # update: y_next = y_curr - eta * grad + momentum_term
                            p.copy_(y_curr - self.eta * p.grad + momentum_term)
                            # Update y_prev for next iteration
                            self.y_prev[i].copy_(y_curr)

            # Set model contribution
            self.get_model().set_contribution([self.addr], self.get_data().get_num_samples())
            self.add_callback_info_to_model()
            
            # Update P2PFLModel with new parameters (numpy)
            new_params = [p.cpu().detach().numpy() for p in params]
            self.get_model().set_parameters(new_params)

            return self.get_model()

        except Exception as e:
            logger.error(self.addr, f"Momentum Fit error: {e}")
            raise e

    def interrupt_fit(self) -> None:
        pass

    def evaluate(self) -> dict[str, float]:
        """Simple evaluation."""
        pt_model = self.get_model().get_model()
        pt_data = self.get_data().export(PyTorchExportStrategy, train=False)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pt_model.to(device)
        pt_model.eval()
        
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in pt_data:
                if isinstance(batch, list | tuple):
                    x, y = batch[0].to(device), batch[1].to(device)
                else:
                    x, y = batch['image'].to(device), batch['label'].to(device)
                outputs = pt_model(x)
                _, predicted = torch.max(outputs.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        
        acc = correct / total
        return {"test_acc": acc}

    def get_framework(self) -> str:
        return Framework.PYTORCH.value
