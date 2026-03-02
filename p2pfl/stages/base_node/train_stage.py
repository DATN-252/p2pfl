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
"""Train stage."""

from typing import Any

from p2pfl.communication.commands.message.metrics_command import MetricsCommand
from p2pfl.communication.commands.message.models_agregated_command import ModelsAggregatedCommand
from p2pfl.communication.commands.message.models_ready_command import ModelsReadyCommand
from p2pfl.communication.commands.weights.partial_model_command import PartialModelCommand
from p2pfl.communication.protocols.communication_protocol import CommunicationProtocol
from p2pfl.learning.aggregators.aggregator import Aggregator, NoModelsToAggregateError
from p2pfl.learning.frameworks.learner import Learner
from p2pfl.management.experiment_logger import ExperimentLogger # NEW IMPORT
from p2pfl.management.logger import logger
from p2pfl.node_state import NodeState
from p2pfl.stages.stage import EarlyStopException, Stage, check_early_stop
from p2pfl.stages.stage_factory import StageFactory


class TrainStage(Stage):
    """Train stage."""

    @staticmethod
    def name():
        """Return the name of the stage."""
        return "TrainStage"

    @staticmethod
    def execute(
        state: NodeState | None = None,
        communication_protocol: CommunicationProtocol | None = None,
        learner: Learner | None = None,
        aggregator: Aggregator | None = None,
        experiment_logger: ExperimentLogger | None = None, # NEW PARAM
        **kwargs,
    ) -> type["Stage"] | None:
        """Execute the stage."""
        if state is None or communication_protocol is None or aggregator is None or learner is None:
            raise Exception("Invalid parameters on TrainStage.")

        try:
            check_early_stop(state)

            # Set Models To Aggregate
            direct_neighbors = list(communication_protocol.get_neighbors(only_direct=True).keys())
            expected_contributors = [state.addr] + direct_neighbors
            aggregator.set_nodes_to_aggregate(expected_contributors, round_num=state.round)

            check_early_stop(state)


            # Train
            if state.addr in state.train_set:
                check_early_stop(state)
                logger.info(state.addr, "🏋️‍♀️ Preparing training...")
                learner.fit(apply_update=not aggregator.requires_gradient_only)
                logger.info(state.addr, "🎓 Training step done.")
            else:
                logger.info(state.addr, "💤 Skipping training...")

            check_early_stop(state)

            # Aggregate Model
            current_model = learner.get_model()
            if state.addr not in state.train_set:
                n_s = learner.get_data().get_num_samples()
                current_model.set_contribution([state.addr], n_s) 
            
            # Use force_add_local_model instead of add_model for the local node
            logger.info(state.addr, "🛠️ Calling force_add_local_model...")
            aggregator.force_add_local_model(current_model)

            import time
            time.sleep(5) # wait for continuous voting

            TrainStage.__send_model_direct(state, communication_protocol, current_model)
            check_early_stop(state)
            
            # Set aggregated model
            aggregator.set_trained_round(state.addr)
            agg_model = aggregator.wait_and_get_aggregation(state=state)
            logger.info(state.addr, "🧠 Aggregation complete. Setting new model.")
            learner.set_model(agg_model)

            # Share that aggregation is done
            communication_protocol.broadcast(communication_protocol.build_msg(ModelsReadyCommand.get_name(), [], round=state.round))

            # Next stage
            return StageFactory.get_stage("GossipModelStage")
        except EarlyStopException:
            return None

    @staticmethod
    def __evaluate(state: NodeState, learner: Learner, communication_protocol: CommunicationProtocol, aggregator: Aggregator, experiment_logger: ExperimentLogger | None) -> None: # NEW param
        logger.info(state.addr, "🔬 Evaluating...")
        results = learner.evaluate()

        # Add communication cost to results
        results["communication_cost"] = aggregator.get_last_comm_cost()

        logger.info(state.addr, f"📈 Evaluated. Results: {results}")
        
        # NEW: Record metrics with experiment_logger
        if experiment_logger:
            experiment_logger.record_metrics(state.round, results)

        # Send metrics
        if len(results) > 0:
            logger.info(state.addr, "📢 Broadcasting metrics.")
            flattened_metrics = [str(item) for pair in results.items() for item in pair]
            communication_protocol.broadcast(
                communication_protocol.build_msg(
                    MetricsCommand.get_name(),
                    flattened_metrics,
                    round=state.round,
                )
            )

    @staticmethod
    def __gossip_model_aggregation(
        state: NodeState,
        communication_protocol: CommunicationProtocol,
        aggregator: Aggregator,
    ) -> None:
        """
        Gossip model aggregation.

        CAREFULL:
            - Full connected trainset to increase aggregation speed. On real scenarios, this won't
            be possible, private networks and firewalls.
            - Needed because the trainset can split the networks (and neighbors that are not in the
            trainset won't receive the aggregation).
        """

        # Anonymous functions
        def early_stopping_fn():
            return state.round is None

        def get_candidates_fn() -> list[str]:
            candidates = set(state.train_set) - {state.addr}
            return [n for n in candidates if len(TrainStage.__get_remaining_nodes(n, state)) != 0]

        def status_fn() -> Any:
            return [
                (
                    n,
                    TrainStage.__get_aggregated_models(n, state),
                )  # reemplazar por Aggregator - borrarlo de node
                for n in communication_protocol.get_neighbors(only_direct=False)
                if (n in state.train_set)
            ]

        def model_fn(node: str) -> tuple[Any, str, int, list[str]]:
            if state.round is None:
                raise Exception("Round not initialized.")
            try:
                model = aggregator.get_model(TrainStage.__get_aggregated_models(node, state))
            except NoModelsToAggregateError:
                logger.debug(state.addr, f"❔ No models to aggregate for {node}.")
                return (
                    None,
                    PartialModelCommand.get_name(),
                    state.round,
                    [],
                )
            model_msg = communication_protocol.build_weights(
                PartialModelCommand.get_name(),
                state.round,
                model.encode_parameters(),
                model.get_contributors(),
                model.get_num_samples(),
            )
            return (
                model_msg,
                PartialModelCommand.get_name(),
                state.round,
                model.get_contributors(),
            )

        # Gossip
        communication_protocol.gossip_weights(
            early_stopping_fn,
            get_candidates_fn,
            status_fn,
            model_fn,
            create_connection=True,
        )

    @staticmethod
    def __get_aggregated_models(node: str, state: NodeState) -> list[str]:
        try:
            return state.models_aggregated[node]
        except KeyError:
            return []

    @staticmethod
    def __get_remaining_nodes(node: str, state: NodeState) -> set[str]:
        return set(state.train_set) - set(TrainStage.__get_aggregated_models(node, state))
    
    @staticmethod
    def __send_model_direct(
        state: NodeState,
        communication_protocol: CommunicationProtocol,
        model: Any
    ) -> None:
        """
        Gửi mô hình chỉ cho các hàng xóm kết nối trực tiếp (Direct Neighbors).
        Không sử dụng cơ chế Broadcast hay Gossip lan truyền.
        """
        direct_neighbors = communication_protocol.get_neighbors(only_direct=True)
        
        if not direct_neighbors:
            logger.info(state.addr, "⚠️ No direct neighbors to send model.")
            return

        logger.info(state.addr, f"📤 Sending model to {len(direct_neighbors)} direct neighbors: {list(direct_neighbors.keys())}")

        from p2pfl.communication.commands.weights.partial_model_command import PartialModelCommand
        from p2pfl.communication.commands.message.pre_send_model_command import PreSendModelCommand
        
        contributors = model.get_contributors()
        
        pre_send_args = [PartialModelCommand.get_name()] + contributors
        pre_send_msg = communication_protocol.build_msg(
            cmd=PreSendModelCommand.get_name(),
            args=pre_send_args,
            round=state.round
        )

        model_msg = communication_protocol.build_weights(
            cmd=PartialModelCommand.get_name(),
            round=state.round or 0,
            serialized_model=model.encode_parameters(), 
            contributors=model.get_contributors(),
            weight=model.get_num_samples(),
        )

        for neighbor_addr in direct_neighbors:
            try:
                communication_protocol.send(neighbor_addr, pre_send_msg)
                communication_protocol.send(neighbor_addr, model_msg)
            except Exception as e:
                logger.error(state.addr, f"❌ Failed to send model to {neighbor_addr}: {e}")
