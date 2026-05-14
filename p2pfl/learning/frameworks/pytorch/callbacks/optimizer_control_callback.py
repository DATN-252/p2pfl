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
        # Use state_dict to ensure alignment with LightningModel.get_parameters()
        self._weights_start = {k: v.detach().cpu().clone() for k, v in pl_module.state_dict().items()}

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        """Calculate the accumulated delta at the end of the epoch."""
        if self._weights_start is not None:
            weights_end = pl_module.state_dict()
            delta = []
            
            for k, start_val in self._weights_start.items():
                end_val = weights_end[k].detach().cpu()
                # Delta = weights_start - weights_end
                delta.append((start_val - end_val).numpy())
            
            self.additional_info["delta"] = delta

            # If we don't want to apply the update, revert to original weights
            # We do this AFTER calculating delta
            if not self._apply_update:
                pl_module.load_state_dict(self._weights_start)

            self._weights_start = None

    def on_before_optimizer_step(self, trainer: L.Trainer, pl_module: L.LightningModule, optimizer: Any):
        """
        DO NOT zero the grad here anymore.
        We need the optimizer to step so we can calculate 'delta' (weights_start - weights_end).
        If apply_update is False, we revert the weights in on_train_epoch_end.
        """
        pass
