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

"""PartialModelCommand command."""

from collections.abc import Callable
from p2pfl.communication.commands.command import Command
from p2pfl.communication.commands.message.models_agregated_command import ModelsAggregatedCommand
from p2pfl.communication.commands.message.pre_send_model_command import PreSendModelCommand
from p2pfl.communication.protocols.communication_protocol import CommunicationProtocol
from p2pfl.learning.aggregators.aggregator import Aggregator
from p2pfl.learning.frameworks.learner import Learner
from p2pfl.management.logger import logger
from p2pfl.node_state import NodeState


class PartialModelCommand(Command):
    """PartialModelCommand."""

    def __init__(self, state: NodeState, stop: Callable[[], None], aggregator: Aggregator, comm_proto: CommunicationProtocol, learner: Learner) -> None:
        """Initialize PartialModelCommand."""
        self.state = state
        self.stop = stop
        self.aggregator = aggregator
        self.communication_protocol = comm_proto
        self.learner = learner

    @staticmethod
    def get_name() -> str:
        """Get the command name."""
        return "partial_model"

    def execute(self, source: str, round: int, weights: bytes | None = None, contributors: list[str] | None = None, num_samples: int | None = None, **kwargs) -> None:
        """Execute the command (Non-blocking)."""
        if weights is None or contributors is None or num_samples is None:
            return

        if self.state.round is not None and round < self.state.round:
            return

        try:
            model = self.learner.get_model().build_copy(params=weights, num_samples=num_samples, contributors=list(contributors))
            models_added = self.aggregator.add_model(model, round_num=round)
            if models_added != []:
                self.communication_protocol.broadcast(self.communication_protocol.build_msg(ModelsAggregatedCommand.get_name(), models_added, round=self.state.round))
            else:
                PreSendModelCommand.remove_hashed(self.state, self.get_name(), contributors, round)
        except Exception as e:
            logger.error(self.state.addr, f"Error adding model: {e}")
