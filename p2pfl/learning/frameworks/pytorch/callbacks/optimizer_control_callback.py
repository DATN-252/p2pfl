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
    This callback serves two purposes for Consensus-then-Gradient algorithms:
    1. When apply_update=False, it calculates the 'delta' (-lr * grad) and
       stores it in `additional_info`.
    2. It then calls optimizer.zero_grad() to prevent the optimizer from
       applying the update, effectively skipping the step.
    """

    def __init__(self):
        self._apply_update = True
        self.additional_info: dict[str, Any] = {}

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

    def on_before_optimizer_step(self, trainer: L.Trainer, pl_module: L.LightningModule, optimizer: Any):
        """Hook called before the optimizer step."""
        if trainer.state.fn == TrainerFn.FITTING and not self._apply_update:
            # 1. Calculate and store the delta before gradients are cleared.
            lr = pl_module.lr_rate
            delta = []
            for param in pl_module.parameters():
                if param.grad is not None:
                    delta.append(-lr * param.grad.cpu())
                else:
                    delta.append(torch.zeros_like(param.cpu()))
            
            # Store the computed delta for the aggregators to use.
            self.additional_info["delta"] = [d.detach().cpu().numpy() for d in delta]

            # 2. Prevent the update by zeroing the gradients before the step.
            optimizer.zero_grad()

