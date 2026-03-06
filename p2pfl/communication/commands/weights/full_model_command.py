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

"""FullModelCommand."""

from collections.abc import Callable
from p2pfl.communication.commands.command import Command
from p2pfl.learning.aggregators.aggregator import Aggregator
from p2pfl.learning.frameworks.learner import Learner
from p2pfl.management.logger import logger
from p2pfl.node_state import NodeState


class FullModelCommand(Command):
    """FullModelCommand."""

    def __init__(self, state: NodeState, stop: Callable[[], None], aggregator: Aggregator, learner: Learner) -> None:
        """Initialize FullModelCommand."""
        self.state = state
        self.stop = stop
        self.aggregator = aggregator
        self.learner = learner

    @staticmethod
    def get_name() -> str:
        """Get the command name."""
        return "add_model"

    def execute(self, source: str, round: int, weights: bytes | None = None, **kwargs) -> None:
        """Execute the command (Non-blocking)."""
        if weights is None:
            return

        if self.state.round is not None and round < self.state.round:
            return

        # CENTRALIZED SYNC: If client is behind, jump to the round received from server
        is_future_round = self.state.round is not None and round > self.state.round
        should_sync = self.state.round == round or (self.state.is_centralized and is_future_round)

        if should_sync and not self.state.aggregated_model_event.is_set():
            try:
                if is_future_round:
                    logger.info(self.state.addr, f"⏩ Synchronization: Jumping from Round {self.state.round} to {round} (Centralized mode).")
                    # Update round in state
                    while self.state.round < round:
                        self.state.increase_round()

                self.learner.set_model(weights)
                self.state.aggregated_model_event.set()
            except Exception as e:
                logger.error(self.state.addr, f"Error adding/syncing full model: {e}")
        else:
            try:
                # FIX: Extract weight (num_samples) from kwargs
                weight = kwargs.get("weight", 1)
                model = self.learner.get_model().build_copy(params=weights, contributors=[source], num_samples=weight)
                self.aggregator.add_model(model, round_num=round)
            except Exception as e:
                logger.error(self.state.addr, f"Error buffering full model: {e}")
