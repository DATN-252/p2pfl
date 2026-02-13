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

"""VoteTrainSetCommand."""

import contextlib
from p2pfl.communication.commands.command import Command
from p2pfl.node_state import NodeState


class VoteTrainSetCommand(Command):
    """VoteTrainSetCommand."""

    def __init__(self, state: NodeState) -> None:
        """Initialize the command."""
        self.state = state

    @staticmethod
    def get_name() -> str:
        """Get the command name."""
        return "vote_train_set"

    def execute(self, source: str, round: int, *args, **kwargs) -> None:
        """Execute the command."""
        # build vote dict
        votes = args
        tmp_votes = {}
        for i in range(0, len(votes), 2):
            tmp_votes[votes[i]] = int(votes[i + 1])
        
        # set votes for specific round (non-blocking)
        with self.state.train_set_votes_lock:
            if round not in self.state.train_set_votes:
                self.state.train_set_votes[round] = {}
            self.state.train_set_votes[round][source] = tmp_votes
            
        # Notify wait loop
        with contextlib.suppress(Exception):
            self.state.wait_votes_ready_lock.release()
