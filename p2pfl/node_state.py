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
"""Node state."""

import threading

from p2pfl.experiment import Experiment
from p2pfl.management.logger import logger
from p2pfl.management.experiment_logger import ExperimentLogger # NEW IMPORT


class NodeState:
    """
    Class to store the main state of a learning node.
    """

    def __init__(self, addr: str) -> None:
        """Initialize the node state."""
        self.addr = addr
        self.status = "Idle"
        
        # Reference to the logger for path retrieval
        self.experiment_logger: ExperimentLogger | None = None # NEW

        # Aggregator
        self.models_aggregated_lock = threading.Lock()
        self.models_aggregated: dict[str, list[str]] = {}

        # Other neis state (only round)
        self.nei_status: dict[str, int] = {}

        # Train Set
        self._train_set: list[str] = []
        # Round -> Source -> Vote (dict[str, int])
        self.train_set_votes: dict[int, dict[str, dict[str, int]]] = {}

        # Actual experiment
        self.experiment: Experiment | None = None
        self.buffered_initial_weights: bytes | None = None

        # For PreSendModelCommand state
        self.sending_models: dict[str, dict[str, float]] = {}
        self.sending_models_lock = threading.Lock()

        # Locks
        self.train_set_votes_lock = threading.Lock()
        self.start_thread_lock = threading.Lock()
        self.wait_votes_ready_lock = threading.Lock()
        self.model_initialized_lock = threading.Lock()
        self.model_initialized_lock.acquire()
        self.aggregated_model_event = threading.Event()
        self.aggregated_model_event.set()

        # Conditions for efficient waiting
        self.round_condition = threading.Condition()
        self.train_set_condition = threading.Condition()

    @property
    def round(self) -> int | None:
        """Get the round."""
        return self.experiment.round if self.experiment is not None else None

    @property
    def total_rounds(self) -> int | None:
        """Get the total rounds."""
        return self.experiment.total_rounds if self.experiment is not None else None

    @property
    def exp_name(self) -> str | None:
        """Get the actual experiment name."""
        return self.experiment.exp_name if self.experiment is not None else None

    @property
    def train_set(self) -> list[str]:
        """Get the train set."""
        return self._train_set

    @train_set.setter
    def train_set(self, value: list[str]) -> None:
        """Set the train set and notify waiting threads."""
        with self.train_set_condition:
            self._train_set = value
            self.train_set_condition.notify_all()

    def set_experiment(
        self,
        exp_name: str,
        total_rounds: int,
        **kwargs,
    ) -> None:
        """Start a new experiment."""
        self.status = "Learning"
        with self.round_condition:
            if self.experiment is None:
                self.experiment = Experiment(exp_name, total_rounds, **kwargs)
            self.round_condition.notify_all()
        logger.experiment_started(self.addr, self.experiment)

    def increase_round(self) -> None:
        """Increase the round number."""
        if self.experiment is None:
            logger.warning(self.addr, "Attempted to increase round but experiment is not initialized. Ignoring.")
            return

        # Clear old votes
        new_round = self.experiment.round + 1
        with self.train_set_votes_lock:
            rounds_to_clear = [r for r in self.train_set_votes.keys() if r < new_round]
            for r in rounds_to_clear:
                del self.train_set_votes[r]

        with self.round_condition:
            self.experiment.increase_round()
            self.round_condition.notify_all()
        
        self.train_set = [] 
        self.models_aggregated = {}
        logger.experiment_started(self.addr, self.experiment)

    def clear(self) -> None:
        """Clear the state."""
        self.__init__(self.addr)

    def wait_for_initialization(self, timeout: float = 10.0) -> bool:
        """Wait for the experiment to be initialized."""
        with self.round_condition:
            if self.round is None:
                self.round_condition.wait(timeout=timeout)
        return self.round is not None

    def wait_for_train_set(self, timeout: float = 20.0) -> bool:
        """Wait for the train set to be determined."""
        with self.train_set_condition:
            if len(self.train_set) == 0:
                self.train_set_condition.wait(timeout=timeout)
        return len(self.train_set) > 0

    def __str__(self) -> str:
        """Return a String representation of the node state."""
        return (
            f"NodeState(addr={self.addr}, status={self.status}, exp_name={self.exp_name}, "
            f"round={self.round}, total_rounds={self.total_rounds})"
        )
