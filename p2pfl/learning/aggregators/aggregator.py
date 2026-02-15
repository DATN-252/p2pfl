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
from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel
from p2pfl.management.logger import logger
from p2pfl.settings import Settings
from p2pfl.utils.node_component import NodeComponent

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
        self.__unhandled_models: list[P2PFLModel] = []
        self.__local_model_backup: P2PFLModel | None = None

    def aggregate(self, models: list[P2PFLModel]) -> P2PFLModel:
        raise NotImplementedError

    def get_required_callbacks(self) -> list[str]:
        return []

    def normalize_addr(self, addr: str) -> str:
        """Remove P2PFL version suffixes from addresses."""
        if addr and isinstance(addr, str) and "-" in addr:
            return addr.split("-")[0]
        return str(addr)

    def set_nodes_to_aggregate(self, nodes_to_aggregate: list[str]) -> None:
        with self.__agg_lock:
            if not self._finish_aggregation_event.is_set():
                logger.warning(self.addr, "Force clearing aggregator state to start new round.")
                self.clear()

            self.__train_set = nodes_to_aggregate
            self._finish_aggregation_event.clear()
            # Try to process unhandled models for the new round
            to_process = self.__unhandled_models
            self.__unhandled_models = []
            for m in to_process:
                self.add_model(m)

    def clear(self) -> None:
        with self.__agg_lock:
            self.__train_set = []
            self.__models = []
            # Note: we don't clear __unhandled_models here anymore
            # to preserve models from future rounds that arrived early.
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
            # Check if local node already contributed to any model in __models
            already_present = False
            for m in self.__models:
                if any(self.normalize_addr(c) == norm_self for c in m.get_contributors()):
                    already_present = True
                    break
            
            if not already_present:
                self.__models.append(model)
                logger.info(self.addr, f"✅ [FORCE] Local model added. ({len(self.__models)}/{len(self.__train_set)})")
                if len(self.__models) >= len(self.__train_set) > 0:
                    self._finish_aggregation_event.set()

    def add_model(self, model: P2PFLModel) -> list[str]:
        contributors = model.get_contributors()
        if not contributors:
            return []

        norm_self = self.normalize_addr(self.addr)
        norm_contributors = [self.normalize_addr(c) for c in contributors]
        is_local = norm_self in norm_contributors

        with self.__agg_lock:
            if is_local:
                self.force_add_local_model(model)
                return self.get_aggregated_models()

            # Check if we are even expecting models
            if not self.__train_set:
                self.__unhandled_models.append(model)
                return []

            # Check if all contributors of this model are in our train_set
            norm_train_set = {self.normalize_addr(t) for t in self.__train_set}
            if all(c in norm_train_set for c in norm_contributors):
                # Check if any of these contributors have already been added
                current_contributors = {self.normalize_addr(c) for m in self.__models for c in m.get_contributors()}
                if not any(c in current_contributors for c in norm_contributors):
                    self.__models.append(model)
                    logger.info(self.addr, f"🧩 Model added ({len(self.__models)}/{len(self.__train_set)}) from {contributors}")
                    if len(self.__models) >= len(self.__train_set):
                        self._finish_aggregation_event.set()
                    return self.get_aggregated_models()
                else:
                    logger.debug(self.addr, f"🚫 Model from {contributors} already aggregated.")
            else:
                # If the aggregation is full, save to unhandled
                if len(self.__models) >= len(self.__train_set):
                    self.__unhandled_models.append(model)
                else:
                    logger.debug(self.addr, f"🚫 Contributors {norm_contributors} not in train_set.")
        return []

    def wait_and_get_aggregation(self, timeout: int = Settings.training.AGGREGATION_TIMEOUT) -> P2PFLModel:
        self._finish_aggregation_event.wait(timeout=timeout)
        
        with self.__agg_lock:
            if not self.__models:
                if self.__local_model_backup:
                    logger.warning(self.addr, "⚠️ Aggregation empty. Using local backup.")
                    self.__models = [self.__local_model_backup]
                elif self.__unhandled_models:
                    self.__models = [self.__unhandled_models.pop(0)]
                    logger.info(self.addr, "✅ Recovered from unhandled.")
            
            if not self.__models:
                raise NoModelsToAggregateError(f"({self.addr}) No models after fallback. Expected: {len(self.__train_set)}")

            try:
                result = self.aggregate(self.__models)
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
