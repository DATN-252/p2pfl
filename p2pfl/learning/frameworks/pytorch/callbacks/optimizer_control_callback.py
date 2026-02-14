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

"""A callback to conditionally skip the optimizer step and calculate delta."""

from typing import Any

import lightning as L
import torch
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.trainer.states import TrainerFn

from p2pfl.learning.frameworks.callback import P2PFLCallback


class OptimizerControlCallback(Callback, P2PFLCallback):
    """
    Calculates accumulated delta (weights_start - weights_end) over an epoch.
    This serves as an accumulated pseudo-gradient for Gradient Tracking algorithms.
    """

    def __init__(self):
        self._apply_update = True
        self.additional_info: dict[str, Any] = {}
        self._weights_start = None

    @staticmethod
    def get_name() -> str:
        """Get the name of the callback."""
        return "gradient_delta_calculator"

    def get_info(self) -> Any:
        """Get the additional information."""
        return self.additional_info

    def set_apply_update(self, apply_update: bool):
        """Set whether to apply the optimizer update."""
        self._apply_update = apply_update

    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule):
        """Store weights at the start of the epoch."""
        self._weights_start = [p.detach().cpu().clone() for p in pl_module.parameters()]

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        """Calculate the accumulated delta at the end of the epoch."""
        if self._weights_start is not None:
            weights_end = [p.detach().cpu() for p in pl_module.parameters()]
            
            # Delta = weights_start - weights_end
            delta = [s - e for s, e in zip(self._weights_start, weights_end)]
            
            self.additional_info["delta"] = [d.numpy() for d in delta]
            self._weights_start = None

    def on_before_optimizer_step(self, trainer: L.Trainer, pl_module: L.LightningModule, optimizer: Any):
        """
        Prevent the local update ONLY if explicitly requested.
        For Accumulated Gradient Tracking, we usually want this to be TRUE.
        """
        if trainer.state.fn == TrainerFn.FITTING and not self._apply_update:
            optimizer.zero_grad()

