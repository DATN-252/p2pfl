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

"""Abstract aggregator."""

import threading
from collections import defaultdict
from typing import Any
from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel
from p2pfl.management.logger import logger
from p2pfl.settings import Settings
from p2pfl.utils.node_component import NodeComponent

# --- START MODEL PACKAGING ADDITION ---
from p2pfl.utils.model_packaging import ModelPackager
# --- END MODEL PACKAGING ADDITION ---

class NoModelsToAggregateError(Exception):
    """Exception raised when there are no models to aggregate."""
    pass

class Aggregator(NodeComponent):
    """Class to manage the aggregation of models."""

    SUPPORTS_PARTIAL_AGGREGATION: bool = False
    requires_gradient_only: bool = False

    def __init__(self, disable_partial_aggregation: bool = False, learning_rate: float = 0.01) -> None:
        """Initialize the aggregator."""
        self.__train_set: list[str] = []
        self.__models: list[P2PFLModel] = []
        self.partial_aggregation: bool = self.__class__.SUPPORTS_PARTIAL_AGGREGATION
        if self.partial_aggregation and disable_partial_aggregation:
            self.partial_aggregation = False

        self.learning_rate = learning_rate
        self.each_trained_round = defaultdict(int)

        NodeComponent.__init__(self)

        self.__agg_lock = threading.RLock()
        self._finish_aggregation_event = threading.Event()
        self._finish_aggregation_event.set()
        # Round-aware buffering: round_num -> list of models
        self.__unhandled_models: dict[int, list[P2PFLModel]] = defaultdict(list)
        self.__local_model_backup: P2PFLModel | None = None
        self.__current_round: int | None = None
        self.__comm_cost: int = 0
        self.__last_comm_cost: int = 0

        # --- START MODEL PACKAGING ADDITION ---
        # Highly relaxed threshold for quick testing (epsilon 0.1, patience 1)
        # self.packager = ModelPackager(epsilon=0.1, patience=1)
        # --- END MODEL PACKAGING ADDITION ---

    def aggregate(self, models: list[P2PFLModel]) -> P2PFLModel:
        raise NotImplementedError

    def get_comm_cost(self) -> int:
        """Get the communication cost for the current round."""
        return self.__comm_cost

    def get_last_comm_cost(self) -> int:
        """Get the communication cost for the last finished round."""
        return self.__last_comm_cost

    def get_required_callbacks(self) -> list[str]:
        return []

    def normalize_addr(self, addr: str) -> str:
        """Remove P2PFL version suffixes from addresses."""
        if addr and isinstance(addr, str) and "-" in addr:
            return addr.split("-")[0]
        return str(addr)

    def set_nodes_to_aggregate(self, nodes_to_aggregate: list[str], round_num: int | None = None) -> None:
        with self.__agg_lock:
            if not self._finish_aggregation_event.is_set():
                logger.warning(self.addr, f"Force clearing aggregator state to start new round (Current: {self.__current_round}, New: {round_num}).")
                self.clear()

            self.__train_set = nodes_to_aggregate
            self.__current_round = round_num
            self._finish_aggregation_event.clear()
            
            # PROACTIVE RECOVERY: Process models that arrived early for this specific round
            if round_num is not None:
                to_process = self.__unhandled_models.pop(round_num, [])
                if to_process:
                    logger.info(self.addr, f"♻️ Proactively processing {len(to_process)} early models for round {round_num}.")
                    for m in to_process:
                        self.add_model(m, round_num)
                
                # Clean up very old rounds to prevent memory leaks
                old_rounds = [r for r in self.__unhandled_models.keys() if r < round_num]
                for r in old_rounds:
                    del self.__unhandled_models[r]

    def clear(self) -> None:
        with self.__agg_lock:
            if self.__current_round is not None:
                self.__last_comm_cost = self.__comm_cost
            self.__train_set = []
            self.__models = []
            self.__current_round = None
            self.__comm_cost = 0
            # Preserve __unhandled_models for future rounds
            self._finish_aggregation_event.set()

    def get_aggregated_models(self) -> list[str]:
        models_added = []
        for n in self.__models:
            models_added += n.get_contributors()
        return models_added

    def force_add_local_model(self, model: P2PFLModel) -> None:
        """Forcefully add the local model."""
        with self.__agg_lock:
            self.__local_model_backup = model
            norm_self = self.normalize_addr(self.addr)
            already_present = False
            for m in self.__models:
                if any(self.normalize_addr(c) == norm_self for c in m.get_contributors()):
                    already_present = True
                    break
            
            if not already_present:
                self.__models.append(model)
                logger.info(self.addr, f"✅ [FORCE] Local model added for round {self.__current_round}. ({len(self.__models)}/{len(self.__train_set)})")
                if self.__train_set and len(self.__models) >= len(self.__train_set):
                    self._finish_aggregation_event.set()

    def add_model(self, model: P2PFLModel, round_num: int | None = None) -> list[str]:
        contributors = model.get_contributors()
        if not contributors:
            return []

        norm_self = self.normalize_addr(self.addr)
        norm_contributors = [self.normalize_addr(c) for c in contributors]
        is_local = norm_self in norm_contributors

        with self.__agg_lock:
            # Local models are usually added directly by TrainStage via force_add_local_model
            if is_local:
                self.force_add_local_model(model)
                return self.get_aggregated_models()

            # Handle round mismatch
            if round_num is not None and self.__current_round is not None:
                if round_num < self.__current_round:
                    logger.debug(self.addr, f"Ignoring model from past round {round_num} (Current: {self.__current_round}) from {contributors}")
                    return []
                if round_num > self.__current_round:
                    logger.debug(self.addr, f"Buffering model from future round {round_num} (Current: {self.__current_round}) from {contributors}")
                    self.__unhandled_models[round_num].append(model)
                    return []

            # If current_round is not set yet or matches
            if not self.__train_set:
                if round_num is not None:
                    self.__unhandled_models[round_num].append(model)
                return []

            norm_train_set = {self.normalize_addr(t) for t in self.__train_set}
            if all(c in norm_train_set for c in norm_contributors):
                current_contributors = {self.normalize_addr(c) for m in self.__models for c in m.get_contributors()}
                if not any(c in current_contributors for c in norm_contributors):
                    self.__models.append(model)
                    self.__comm_cost += model.size
                    logger.info(self.addr, f"🧩 Model added for round {self.__current_round} ({len(self.__models)}/{len(self.__train_set)}) from {contributors}")
                    if len(self.__models) >= len(self.__train_set):
                        self._finish_aggregation_event.set()
                    return self.get_aggregated_models()
                else:
                    logger.debug(self.addr, f"🚫 Model from {contributors} already aggregated for round {self.__current_round}.")
            else:
                # If not in train set, might be for a future round where the train set is different
                if round_num is not None:
                    self.__unhandled_models[round_num].append(model)
        return []

    def wait_and_get_aggregation(self, timeout: int = Settings.training.AGGREGATION_TIMEOUT, state: Any = None) -> P2PFLModel:
        """
        Wait for models and return the aggregation.
        Implements Dynamic Patience: if 'state' is provided, it will be more patient 
        if missing nodes are still active.
        """
        # Initial wait
        self._finish_aggregation_event.wait(timeout=timeout)
        
        # Dynamic Patience: If we still don't have all models, check if missing nodes are alive
        if not self._finish_aggregation_event.is_set() and state is not None:
            max_patience_rounds = 5 # Maximum extra wait iterations
            for i in range(max_patience_rounds):
                missing = self.get_missing_models()
                # Check if any missing node is still in our round or previous (meaning they are just slow)
                # If they moved to a FUTURE round, then we really missed them.
                still_active = False
                with state.round_condition:
                    for m_node in missing:
                        m_round = state.nei_status.get(m_node, -1)
                        if m_round != -1 and m_round <= (self.__current_round or 0):
                            still_active = True
                            break
                
                if still_active:
                    logger.info(self.addr, f"⏳ Dynamic Patience (iteration {i+1}): Missing nodes still active. Waiting {timeout//2}s more...")
                    if self._finish_aggregation_event.wait(timeout=timeout // 2):
                        break
                else:
                    break

        with self.__agg_lock:
            if not self.__models:
                if self.__local_model_backup:
                    logger.warning(self.addr, f"⚠️ Aggregation list empty for round {self.__current_round} after timeout. Using local backup.")
                    self.__models = [self.__local_model_backup]
                elif self.__unhandled_models.get(self.__current_round or -1):
                    # Try to recover from unhandled if any (shouldn't happen with proactive recovery but as a safety)
                    self.__models = [self.__unhandled_models[self.__current_round].pop(0)]
                    logger.info(self.addr, f"✅ Recovered from unhandled for round {self.__current_round}.")
            
            if not self.__models:
                raise NoModelsToAggregateError(f"({self.addr}) No models available to aggregate for round {self.__current_round}.")

            try:
                result = self.aggregate(self.__models)
                
                # --- START MODEL PACKAGING ADDITION ---
                # if state is not None and hasattr(state, "experiment_logger") and state.experiment_logger is not None:
                #     self.packager.check_and_package(
                #         models=self.__models, 
                #         aggregated_model=result, 
                #         round_num=self.__current_round or 0, 
                #         output_path=state.experiment_logger.output_dir,
                #         node_addr=self.normalize_addr(self.addr)
                #     )
                # --- END MODEL PACKAGING ADDITION ---
                
            finally:
                self.clear()
            return result

    def get_missing_models(self) -> set:
        agg_models = []
        for m in self.__models:
            agg_models += m.get_contributors()
        return set(self.__train_set) - set(agg_models)

    def __get_partial_aggregation(self, except_nodes: list[str]) -> P2PFLModel:
        models_to_aggregate = [m for m in self.__models if all(n not in except_nodes for n in m.get_contributors())]
        return self.aggregate(models_to_aggregate)

    def __get_remaining_model(self, except_nodes) -> P2PFLModel:
        for m in self.__models:
            if all(n not in except_nodes for n in m.get_contributors()):
                return m
        raise NoModelsToAggregateError("No remaining models available.")

    def get_model(self, except_nodes) -> P2PFLModel:
        if self.partial_aggregation:
            return self.__get_partial_aggregation(except_nodes)
        else:
            return self.__get_remaining_model(except_nodes)

    def set_trained_round(self, addr):
        self.each_trained_round[addr] += 1
