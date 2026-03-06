#
# Centralized Stages.
#
import time
from typing import Any

from p2pfl.communication.commands.message.model_initialized_command import ModelInitializedCommand
from p2pfl.communication.commands.message.models_agregated_command import ModelsAggregatedCommand
from p2pfl.communication.commands.weights.full_model_command import FullModelCommand
from p2pfl.communication.commands.weights.init_model_command import InitModelCommand
from p2pfl.communication.protocols.communication_protocol import CommunicationProtocol
from p2pfl.learning.aggregators.aggregator import Aggregator
from p2pfl.learning.frameworks.learner import Learner
from p2pfl.management.experiment_logger import ExperimentLogger
from p2pfl.management.logger import logger
from p2pfl.node_state import NodeState
from p2pfl.settings import Settings
from p2pfl.stages.stage import Stage, check_early_stop
from p2pfl.stages.stage_factory import StageFactory


class CentralizedStartStage(Stage):
    """Initializes roles and starts the workflow."""

    @staticmethod
    def name():
        return "CentralizedStartStage"

    @staticmethod
    def execute(
        state: NodeState | None = None,
        learner: Learner | None = None,
        communication_protocol: CommunicationProtocol | None = None,
        **kwargs,
    ) -> type["Stage"] | None:
        if state is None or learner is None or communication_protocol is None:
            raise Exception("Invalid parameters on CentralizedStartStage.")

        if state.is_server:
            logger.info(state.addr, "👑 Node is SERVER. Broadcasting initial model.")
            # Server starts by broadcasting its initial model
            encoded_model = learner.get_model().encode_parameters()
            communication_protocol.broadcast(
                communication_protocol.build_weights(InitModelCommand.get_name(), state.round, encoded_model)
            )
            # Notify initialization
            communication_protocol.broadcast(communication_protocol.build_msg(ModelInitializedCommand.get_name()))
            return StageFactory.get_stage("CentralizedServerStage")
        else:
            logger.info(state.addr, "💻 Node is CLIENT. Waiting for initialization.")
            state.model_initialized_lock.acquire()
            return StageFactory.get_stage("CentralizedClientStage")


class CentralizedServerStage(Stage):
    """Server loop: Wait Updates -> Aggregate -> Broadcast."""

    @staticmethod
    def name():
        return "CentralizedServerStage"

    @staticmethod
    def execute(
        state: NodeState | None = None,
        learner: Learner | None = None,
        communication_protocol: CommunicationProtocol | None = None,
        aggregator: Aggregator | None = None,
        experiment_logger: ExperimentLogger | None = None,
        nodes: int | None = None,
        **kwargs,
    ) -> type["Stage"] | None:
        if state is None or learner is None or aggregator is None or nodes is None:
            raise Exception("Invalid parameters on CentralizedServerStage.")

        if check_early_stop(state, raise_exception=False):
            return None

        target_clients = int(nodes) - 1
        logger.info(state.addr, f"⏳ Round {state.round}: Server waiting for {target_clients} updates.")
        
        # 1. Sync neighbors
        neighbors = communication_protocol.get_neighbors(only_direct=False)
        while len(neighbors) < target_clients:
            time.sleep(2.0)
            neighbors = communication_protocol.get_neighbors(only_direct=False)
        
        aggregator.set_nodes_to_aggregate(list(neighbors.keys()), round_num=state.round)

        # 2. Wait for updates (Academic sync mode)
        start_time = time.time()
        while len(aggregator.get_aggregated_models()) < target_clients:
            time.sleep(1.0)
            if check_early_stop(state, raise_exception=False):
                return None
            if time.time() - start_time > Settings.training.AGGREGATION_TIMEOUT:
                logger.warning(state.addr, "⏰ Timeout. Aggregating partial models.")
                break

        # 3. Aggregate and Log
        agg_model = aggregator.wait_and_get_aggregation(timeout=0, state=state)
        learner.set_model(agg_model)
        
        # FIX: Call base class Stage._evaluate
        Stage._evaluate(state, learner, aggregator, experiment_logger)

        if state.round >= state.total_rounds:
            logger.info(state.addr, "🏁 Training Finished.")
            return None

        # 4. Next Round
        state.increase_round()
        encoded_model = learner.get_model().encode_parameters()
        communication_protocol.broadcast(
            communication_protocol.build_weights(FullModelCommand.get_name(), state.round, encoded_model)
        )
        communication_protocol.broadcast(communication_protocol.build_msg(ModelsAggregatedCommand.get_name()))
        
        return StageFactory.get_stage("CentralizedServerStage")


class CentralizedClientStage(Stage):
    """Client loop: Wait Model -> Train -> Send Update."""

    @staticmethod
    def name():
        return "CentralizedClientStage"

    @staticmethod
    def execute(
        state: NodeState | None = None,
        learner: Learner | None = None,
        communication_protocol: CommunicationProtocol | None = None,
        aggregator: Aggregator | None = None,
        experiment_logger: ExperimentLogger | None = None,
        **kwargs,
    ) -> type["Stage"] | None:
        if state is None or learner is None or communication_protocol is None:
            raise Exception("Invalid parameters on CentralizedClientStage.")

        if check_early_stop(state, raise_exception=False):
            return None

        # 1. Wait for Global Model
        while not state.aggregated_model_event.is_set():
            time.sleep(0.5)
        state.aggregated_model_event.clear()

        # 2. Evaluate Global Model
        # FIX: Call base class Stage._evaluate
        Stage._evaluate(state, learner, aggregator, experiment_logger)

        # 3. Local Train
        logger.info(state.addr, f"🚂 Round {state.round}: Client training...")
        learner.fit()

        # 4. Send Update
        current_model = learner.get_model()
        num_samples = learner.get_data().get_num_samples()
        current_model.set_contribution([state.addr], num_samples)
        
        msg = communication_protocol.build_weights(
            FullModelCommand.get_name(), 
            state.round, 
            current_model.encode_parameters(),
            contributors=[state.addr],
            weight=num_samples
        )
        communication_protocol.broadcast(msg)

        if state.round >= state.total_rounds:
            return None

        state.increase_round()
        return StageFactory.get_stage("CentralizedClientStage")
