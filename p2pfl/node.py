import contextlib
import os
import random
import threading
import time
import traceback
from typing import Any

from p2pfl.communication.commands.message.metrics_command import MetricsCommand
from p2pfl.communication.commands.message.model_initialized_command import ModelInitializedCommand
from p2pfl.communication.commands.message.models_agregated_command import ModelsAggregatedCommand
from p2pfl.communication.commands.message.models_ready_command import ModelsReadyCommand
from p2pfl.communication.commands.message.pre_send_model_command import PreSendModelCommand
from p2pfl.communication.commands.message.start_learning_command import StartLearningCommand
from p2pfl.communication.commands.message.stop_learning_command import StopLearningCommand
from p2pfl.communication.commands.message.vote_train_set_command import VoteTrainSetCommand
from p2pfl.communication.commands.weights.full_model_command import FullModelCommand
from p2pfl.communication.commands.weights.init_model_command import InitModelCommand
from p2pfl.communication.commands.weights.partial_model_command import PartialModelCommand
from p2pfl.communication.protocols.communication_protocol import CommunicationProtocol
from p2pfl.communication.protocols.protobuff.grpc import GrpcCommunicationProtocol
from p2pfl.exceptions import LearnerRunningException, NodeRunningException, ZeroRoundsException
from p2pfl.learning.aggregators.aggregator import Aggregator
from p2pfl.learning.aggregators.fedavg import FedAvg
from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset
from p2pfl.learning.frameworks.learner import Learner
from p2pfl.learning.frameworks.learner_factory import LearnerFactory
from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel
from p2pfl.learning.frameworks.simulation import try_init_learner_with_ray
from p2pfl.management.experiment_logger import ExperimentLogger # NEW IMPORT
from p2pfl.management.logger import logger
from p2pfl.node_state import NodeState
from p2pfl.settings import Settings
from p2pfl.stages.workflows import LearningWorkflow

# Disbalbe grpc log (pytorch causes warnings)
if logger.get_level_name(logger.get_level()) != "DEBUG":
    os.environ["GRPC_VERBOSITY"] = "NONE"


class Node:
    """
    Represents a learning node in the federated learning network.

    The following example shows how to create a node with a MLP model and a MnistFederatedDM dataset. Then, the node is
    started, connected to another node, and the learning process is started.

    >>> node = Node(
    ...     MLP(),
    ...     MnistFederatedDM(),
    ... )
    >>> node.start()
    >>> node.connect("127.0.0.1:666")
    >>> node.set_start_learning(rounds=2, epochs=1)

    Args:
        model: Model to be used in the learning process.
        data: Dataset to be used in the learning process.
        addr: The address of the node.
        learner: The learner class to be used.
        aggregator: The aggregator class to be used.
        protocol: The communication protocol to be used.
        experiment_folder_path: Path to the experiment's logging folder. # NEW
        **kwargs: Additional arguments.

    .. todo::
        Instanciate the aggregator dynamically.

    .. todo::
        Connect nodes dynamically (while learning).

    """

    def __init__(
        self,
        model: P2PFLModel,
        data: P2PFLDataset,
        addr: str = "",
        learner: Learner | None = None,
        aggregator: Aggregator | None = None,
        protocol: CommunicationProtocol | None = None,
        experiment_folder_path: str = None, # NEW PARAM
        **kwargs,
    ) -> None:
        """Initialize a node."""
        # Communication protol (and get addr)
        self._communication_protocol = GrpcCommunicationProtocol() if protocol is None else protocol
        self.addr = self._communication_protocol.set_addr(addr)

        # Aggregator
        self.aggregator = FedAvg() if aggregator is None else aggregator
        self.aggregator.set_addr(self.addr)

        # Learner
        if learner is None:  # if no learner, use factory default
            learner = LearnerFactory.create_learner(model)()
        self.learner = try_init_learner_with_ray(learner)
        self.learner.set_addr(self.addr)
        self.learner.set_model(model)
        self.learner.set_data(data)
        self.learner.indicate_aggregator(self.aggregator)

        # State
        self.__running = False
        self.state = NodeState(self.addr)

        # Custom Experiment Logger # NEW
        self.experiment_logger: ExperimentLogger | None = None
        if experiment_folder_path:
            self.experiment_logger = ExperimentLogger(experiment_folder_path, self.addr)
            self.state.experiment_logger = self.experiment_logger # NEW: Linked to state

        # Workflow
        self.learning_workflow = LearningWorkflow()

        # Commands
        commands = [
            StartLearningCommand(self.__start_learning_thread),
            StopLearningCommand(self.state, self.aggregator, self.learner),
            ModelInitializedCommand(self.state),
            VoteTrainSetCommand(self.state),
            ModelsAggregatedCommand(self.state),
            ModelsReadyCommand(self.state),
            MetricsCommand(self.state),
            InitModelCommand(self.state, self.stop, self.aggregator, self.learner),
            PartialModelCommand(self.state, self.stop, self.aggregator, self._communication_protocol, self.learner),
            FullModelCommand(self.state, self.stop, self.aggregator, self.learner),
            PreSendModelCommand(self.state),
        ]
        self._communication_protocol.add_command(commands)

    #############################
    #  Neighborhood management  #
    #############################

    def connect(self, addr: str) -> bool:
        """
        Connect a node to another.

        Warning:
            Adding nodes while learning is running is not fully supported.

        Args:
            addr: The address of the node to connect to.

        Returns:
            True if the node was connected, False otherwise.

        """
        # Check running
        self.assert_running(True)
        # Connect
        return self._communication_protocol.connect(addr)

    def get_neighbors(self, only_direct: bool = False) -> dict[str, Any]:
        """
        Return the neighbors of the node.

        Args:
            only_direct: If True, only the direct neighbors will be returned.

        Returns:
            The list of neighbors.

        """
        return self._communication_protocol.get_neighbors(only_direct)

    def disconnect(self, addr: str) -> None:
        """
        Disconnects a node from another.

        Args:
            addr: The address of the node to disconnect from.

        """
        # Check running
        self.assert_running(True)
        # Disconnect
        logger.info(self.addr, f"Removing {addr}...")
        self._communication_protocol.disconnect(addr, disconnect_msg=True)

    #######################################
    #   Node Management (servicer loop)   #
    #######################################

    """
    -> reemplazarlo por un decorador (y esto creo que se puede reemplazar por el estado del comm proto -> incluso importarlo de ahí)
    """

    def assert_running(self, running: bool) -> None:
        """
        Assert that the node is running or not running.

        Args:
            running: True if the node must be running, False otherwise.

        Raises:
            NodeRunningException: If the node is not running and running is True, or if the node is running and running
            is False.

        """
        running_state = self.__running
        if running_state != running:
            raise NodeRunningException(f"Node is {'not ' if running_state else ''}running.")

    def start(self, wait: bool = False) -> None:
        """
        Start the node: server and neighbors(gossip and heartbeat).

        Args:
            wait: If True, the function will wait until the server is terminated.

        Raises:
            NodeRunningException: If the node is already running.

        """
        # Check not running
        self.assert_running(False)
        # Set running
        self.__running = True

        # P2PFL Web Services
        logger.register_node(self.addr)
        # Communication Protocol
        self._communication_protocol.start()
        if wait:
            self._communication_protocol.wait_for_termination()
            logger.info(self.addr, "gRPC terminated.")

    def stop(self) -> None:
        """
        Stop the node: server and neighbors(gossip and heartbeat).

        Raises:
            NodeRunningException: If the node is not running.

        """
        logger.info(self.addr, "Stopping node...")
        try:
            # Stop learner
            if self.learner:
                self.learner.shutdown()
            # Stop server
            self._communication_protocol.stop()
            # Set not running
            self.__running = False
            # State
            self.state.clear()
            # Unregister node
            logger.unregister_node(self.addr)
        except Exception:
            pass

    ##########################
    #    Learning Setters    #
    ##########################

    def set_learner(self, learner: Learner) -> None:
        """
        Set the learner to be used in the learning process.

        Args:
            learner: The learner to be used in the learning process.

        Raises:
            LearnerRunningException: If the learner is already set.

        """
        if self.state.round is not None:
            raise LearnerRunningException("Learner cannot be set after learning is started.")
        self.learner = learner

    def set_model(self, model: P2PFLModel) -> None:
        """
        Set the model to be used in the learning process (by the learner).

        Args:
            model: Model to be used in the learning process.

        Raises:
            LearnerRunningException: If the learner is already set.

        """
        if self.state.round is not None:
            raise LearnerRunningException("Data cannot be set after learner is set.")
        self.learner.set_model(model)

    def set_data(self, data: P2PFLDataset) -> None:
        """
        Set the data to be used in the learning process (by the learner).

        Args:
            data: Dataset to be used in the learning process.

        Raises:
            LearnerRunningException: If the learner is already set.

        """
        # Cannot change during training (raise)
        if self.state.round is not None:
            raise LearnerRunningException("Data cannot be set after learner is set.")
        self.learner.set_data(data)

    ##########################
    #    Learning Getters    #
    ##########################

    def get_model(self) -> P2PFLModel:
        """
        Get the model.

        Returns:
            The current model of the node.

        """
        return self.learner.get_model()

    def get_data(self) -> P2PFLDataset:
        """
        Get the data.

        Returns:
            The current data of the node.

        """
        return self.learner.get_data()

    ###############################################
    #         Network Learning Management         #
    ###############################################

    def __start_learning_thread(self, rounds: int, epochs: int, trainset_size: int, experiment_name: str, nodes: int | None = None) -> None:
        learning_thread = threading.Thread(
            target=self.__start_learning,
            args=(rounds, epochs, trainset_size, experiment_name, self.experiment_logger, nodes), # NEW: Pass experiment_logger
            name="learning_thread-" + self.addr,
        )
        learning_thread.daemon = True
        learning_thread.start()

    def set_start_learning(self, rounds: int = 1, epochs: int = 1, trainset_size: int = 4, experiment_name="experiment") -> str:
        """
        Start the learning process in the entire network.

        Args:
            rounds: Number of rounds of the learning process.
            epochs: Number of epochs of the learning process.
            trainset_size: Size of the trainset.
            experiment_name: The name of the experiment.

        Raises:
            ZeroRoundsException: If rounds is less than 1.

        """
        self.assert_running(True)

        if rounds < 1:
            raise ZeroRoundsException("Rounds must be greater than 0.")

        if self.state.round is None:
            # Get total nodes in simulation
            total_nodes = len(self.get_neighbors(only_direct=False)) + 1
            # Broadcast start Learning
            logger.info(self.addr, f"🚀 Broadcasting start learning to {total_nodes} nodes...")
            experiment_name = f"{experiment_name}-{time.time()}"
            self._communication_protocol.broadcast(
                self._communication_protocol.build_msg(
                    StartLearningCommand.get_name(), [str(rounds), str(epochs), str(trainset_size), experiment_name, str(total_nodes)]
                )
            )
            # Set model initialized
            self.state.model_initialized_lock.release()
            # Broadcast initialize model
            self._communication_protocol.broadcast(self._communication_protocol.build_msg(ModelInitializedCommand.get_name()))
            # Learning Thread
            self.__start_learning_thread(rounds, epochs, trainset_size, experiment_name, total_nodes)
            return experiment_name
        else:
            logger.info(self.addr, "Learning already started")
            return ""

    def set_stop_learning(self) -> None:
        """Stop the learning process in the entire network."""
        if self.state.round is not None:
            # send stop msg
            self._communication_protocol.broadcast(self._communication_protocol.build_msg(StopLearningCommand.get_name()))
            # stop learning
            self.__stop_learning()
        else:
            logger.info(self.addr, "Learning already stopped")

    ##################################
    #         Local Learning         #
    ##################################

    def __start_learning(self, rounds: int, epochs: int, trainset_size: int, experiment_name: str, experiment_logger: ExperimentLogger | None, nodes: int | None = None) -> None: # NEW param
        # Set seed
        if hasattr(self, 'learner') and self.learner.get_model() is not None:
            neighbors = self.get_neighbors(only_direct=True)
            degree = int(len(neighbors))
            logger.info(self.addr, f"🔍 Neighbors found: {list(neighbors.keys())} | Degree: {degree}")
            self.learner.get_model().set_degrees(degree)
        try:
            # Initialize experiment with metadata
            self.state.set_experiment(
                experiment_name,
                rounds,
                dataset_name=self.learner.get_data().dataset_name,
                model_name=self.learner.get_model().__class__.__name__,
                aggregator_name=self.aggregator.__class__.__name__,
                framework_name=self.learner.get_model().get_framework(),
                learning_rate=getattr(self.learner.get_model().get_model(), "lr_rate", None),
                batch_size=self.learner.get_data().batch_size,
                epochs_per_round=epochs,
            )

            # Run learning workflow
            self.learning_workflow.run(
                rounds=rounds,
                epochs=epochs,
                trainset_size=trainset_size,
                experiment_name=experiment_name,
                state=self.state,
                learner=self.learner,
                communication_protocol=self._communication_protocol,
                aggregator=self.aggregator,
                generator=random.Random(Settings.general.SEED),
                experiment_logger=experiment_logger, # NEW: Pass experiment_logger
                nodes=nodes,
            )

        except Exception as e:
            logger.error(self.addr, f"Error {type(e).__name__}: {e}\n{traceback.format_exc()}")
            self.stop()

    def __stop_learning(self) -> None:
        logger.info(self.addr, "Stopping learning...")
        # Learner
        try:
            self.learner.interrupt_fit()
        except Exception:
            pass
            
        # Aggregator
        self.aggregator.clear()
        
        # --- GRACEFUL WAIT ---
        # Wait a few seconds for final messages/logs to propagate before clearing state
        time.sleep(2)
        
        # State
        self.state.clear()
        logger.experiment_finished(self.addr)
        logger.finish()
        
        # Try to free wait locks
        with contextlib.suppress(Exception):
            self.state.wait_votes_ready_lock.release()
