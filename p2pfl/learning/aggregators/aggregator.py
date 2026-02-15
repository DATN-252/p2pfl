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
        
        # Backup for local model to ensure we never have an empty aggregation
        self.__local_model_backup: P2PFLModel | None = None

    def aggregate(self, models: list[P2PFLModel]) -> P2PFLModel:
        raise NotImplementedError

    def get_required_callbacks(self) -> list[str]:
        return []

    def normalize_addr(self, addr: str) -> str:
        """Remove P2PFL version suffixes from addresses."""
        if addr and "-" in addr:
            return addr.split("-")[0]
        return addr

    def set_nodes_to_aggregate(self, nodes_to_aggregate: list[str]) -> None:
        with self.__agg_lock:
            if not self._finish_aggregation_event.is_set():
                logger.warning(self.addr, "Force clearing aggregator state to start new round.")
                self.clear()

            self.__train_set = nodes_to_aggregate
            self._finish_aggregation_event.clear()
            for m in self.__unhandled_models:
                self.add_model(m)
            self.__unhandled_models = []

    def clear(self) -> None:
        with self.__agg_lock:
            self.__train_set = []
            self.__models = []
            self.__unhandled_models = []
            self._finish_aggregation_event.set()
            # Note: we don't clear __local_model_backup here, 
            # it's updated round by round in TrainStage

    def get_aggregated_models(self) -> list[str]:
        models_added = []
        for n in self.__models:
            models_added += n.get_contributors()
        return models_added

    def force_add_local_model(self, model: P2PFLModel) -> None:
        """Forcefully add the local model and keep a backup."""
        with self.__agg_lock:
            self.__local_model_backup = model
            norm_self_addr = self.normalize_addr(self.addr)
            if not any(norm_self_addr in [self.normalize_addr(c) for c in m.get_contributors()] for m in self.__models):
                self.__models.append(model)
                logger.info(self.addr, f"✅ [FORCE] Local model added. Total: {len(self.__models)}")
                if len(self.__models) >= len(self.__train_set):
                    self._finish_aggregation_event.set()

    def add_model(self, model: P2PFLModel) -> list[str]:
        contributors = model.get_contributors()
        if not contributors:
            return []

        norm_self_addr = self.normalize_addr(self.addr)
        norm_contributors = [self.normalize_addr(c) for c in contributors]
        norm_train_set = [self.normalize_addr(t) for t in self.__train_set]
        is_local = norm_self_addr in norm_contributors

        with self.__agg_lock:
            if is_local:
                self.force_add_local_model(model)
                return self.get_aggregated_models()

            if len(self.__train_set) > len(self.__models):
                if all(c in norm_train_set for c in norm_contributors):
                    any_model_added = any(any(self.normalize_addr(c) in [self.normalize_addr(curr_c) for curr_c in m.get_contributors()] for m in self.__models) for c in contributors)
                    if not any_model_added:
                        self.__models.append(model)
                        logger.info(self.addr, f"🧩 Model added ({len(self.__models)}/{len(self.__train_set)}) from {contributors}")
                        if len(self.__models) >= len(self.__train_set):
                            self._finish_aggregation_event.set()
                        return self.get_aggregated_models()
            else:
                self.__unhandled_models.append(model)
        return []

    def wait_and_get_aggregation(self, timeout: int = Settings.training.AGGREGATION_TIMEOUT) -> P2PFLModel:
        # Wait for aggregation event
        self._finish_aggregation_event.wait(timeout=timeout)
        
        with self.__agg_lock:
            # SUPER FALLBACK: If list is empty, use the backup local model
            if not self.__models:
                if self.__local_model_backup:
                    logger.warning(self.addr, "⚠️ Aggregation list empty after timeout. Using local model backup.")
                    self.__models = [self.__local_model_backup]
                else:
                    # Search in unhandled as last resort
                    for i, m in enumerate(self.__unhandled_models):
                        self.__models = [self.__unhandled_models.pop(i)]
                        logger.info(self.addr, "✅ Recovered a model from unhandled.")
                        break
            
            if not self.__models:
                raise NoModelsToAggregateError(f"({self.addr}) No models available to aggregate after timeout and fallback.")

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
        raise NoModelsToAggregateError("No remaining models available for aggregation.")

    def get_model(self, except_nodes) -> P2PFLModel:
        if self.partial_aggregation:
            return self.__get_partial_aggregation(except_nodes)
        else:
            return self.__get_remaining_model(except_nodes)

    def set_trained_round(self, addr):
        self.each_trained_round[addr] += 1
