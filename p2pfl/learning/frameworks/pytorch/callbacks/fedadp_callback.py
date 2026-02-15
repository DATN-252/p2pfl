#
# This file is part of the federated_learning_p2p (p2pfl) distribution
# (see https://github.com/pguijas/p2pfl).
# Copyright (c) 2022 Pedro Guijas Bravo.
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

"""Callback for FEDADP operations (PyTorch Lighting)."""

import copy
from typing import Any

import lightning as pl
import numpy as np
import torch
from lightning.pytorch.callbacks import Callback

from p2pfl.learning.frameworks.callback import P2PFLCallback


class FEDADPCallback(Callback, P2PFLCallback):
    """
    Callback for FEDADP operations to use with PyTorch Lightning.

    At the beginning of the training, the callback stores the global model. Then, after training, it computes the delta (new parameters minus old parameters).
    """

    def __init__(self) -> None:
        """Initialize the callback."""
        super().__init__()
        # self.initial_model_params: list[torch.Tensor] = []
        self.additional_info: dict[str, Any] = {}

    @staticmethod
    def get_name() -> str:
        """Return the name of the callback."""
        return "fedadp"

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """
        Store the global model and the initial learning rate.

        Args:
            trainer: The trainer
            pl_module: The model.

        """

        initial_model_params = copy.deepcopy(self._get_parameters(pl_module))
        self.additional_info["global_model"] = initial_model_params

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """
        Compute and store the delta (new parameters minus old parameters).

        Args:
            trainer: The trainer
            pl_module: The model.

        """
        post_params = self._get_parameters(pl_module)
        pre_params = self.additional_info["global_model"]
        delta = [post - pre for post, pre in zip(post_params, pre_params)]
        self.additional_info["delta"] = [d.detach().cpu().numpy() for d in delta]

    def _get_parameters(self, pl_module: pl.LightningModule) -> list[torch.Tensor]:
        return [param.cpu() for _, param in pl_module.state_dict().items()]

