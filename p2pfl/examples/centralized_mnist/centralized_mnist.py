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

"""Example of a centralized Federated Learning MNIST experiment using FedAvg."""

import argparse
import numpy as np
import time
from typing import List

from p2pfl.learning.aggregators.centralized_fedavg import CentralizedFedAvgServer, CentralizedFedAvgClient
from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset
from p2pfl.learning.dataset.partition_strategies import RandomIIDPartitionStrategy, DirichletPartitionStrategy
from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel
from p2pfl.management.logger import logger
from p2pfl.settings import Settings


def centralized_mnist_experiment(
    num_clients: int = 5,
    rounds: int = 10,
    epochs_per_round: int = 1,
    fraction_of_clients: float = 1
) -> None:
    """
    Run a centralized federated learning experiment on MNIST using FedAvg.
    
    Args:
        num_clients: Number of clients participating in federated learning
        rounds: Number of federated learning rounds
        epochs_per_round: Number of local training epochs per round
        fraction_of_clients: Fraction of clients to participate in each round
    """
    logger.info("Starting training", f"Starting centralized FL experiment with {num_clients} clients for {rounds} rounds")
    
    # Load and partition the dataset
    logger.info("Starting training", "Loading MNIST dataset...")
    data = P2PFLDataset.from_huggingface("p2pfl/MNIST")
    
    partitions = data.generate_partitions(
        num_clients,
        # RandomIIDPartitionStrategy,  # type: ignore
        DirichletPartitionStrategy,  # type: ignore
    )
    
    # Import the PyTorch model (keeping it simple)
    from p2pfl.examples.mnist.model.mlp_pytorch import model_build_fn
    model_fn = model_build_fn
    
    # Initialize the server
    server = CentralizedFedAvgServer(
        num_clients=num_clients,
        max_rounds=rounds
    )
    
    # Create clients with their local models and datasets
    clients = []
    for i in range(num_clients):
        # Build model for each client
        # Since model_fn() already returns a LightningModel (P2PFLModel), we can use it directly
        client_model = model_fn()
        # Update the model's sample count and contributors
        client_model.num_samples = partitions[i].get_num_samples()
        client_model.contributors = [f"client_{i}"]
        
        # Create client
        client = CentralizedFedAvgClient(
            client_id=f"client_{i}",
            model=client_model,
            local_epochs=epochs_per_round
        )
        
        # Assign local dataset to client
        client.set_local_dataset(partitions[i])
        clients.append(client)
    
    # Initialize the global model using the first client's model structure
    initial_model = clients[0].local_model
    server.initialize_global_model(initial_model)
    
    logger.info("Server init", f"Initialized server and {num_clients} clients")
    
    # Main federated learning loop
    for round_num in range(1, rounds + 1):
        logger.info("Server", f"\n--- Round {round_num}/{rounds} ---")
        
        # Server starts a new round
        current_round = server.start_training_round()
        logger.info("Server", f"Server started round {current_round}")
        
        # Server selects a fraction of clients for this round
        all_client_ids = [client.get_client_id() for client in clients]
        selected_clients = server.select_clients(all_client_ids, fraction_of_clients)
        logger.info("Server", f"Selected {len(selected_clients)} clients for training: {selected_clients}")
        
        # Get the current global model weights
        global_model = server.get_global_model()
        if global_model is None:
            logger.error("ERROR", "Error: No global model available")
            break
            
        global_weights = global_model.get_parameters()
        
        # Clients perform local training and send updates to server
        logger.info("Server", "Clients training locally...")
        import time
        start_time = time.time()
        for client in clients:
            if client.get_client_id() in selected_clients:
                client.send_update_to_server(server, current_round)
        training_time = time.time() - start_time
        logger.info("Server", f"Local training completed in {training_time:.2f} seconds")
        
        # Server waits for updates from selected clients
        logger.info("Server", "Server waiting for client updates...")
        server.wait_for_client_updates(timeout=120)  # 2 minute timeout
        
        # Server aggregates the updates
        logger.info("Server", "Server aggregating client updates...")
        aggregated_model = server.aggregate()
        
        # Update the global model on the server
        server.set_global_model(aggregated_model)
        
        # Print round summary
        logger.info("Server", f"Round {round_num} completed. Global model updated.")
        
        # Simple simulation of accuracy improvement
        # In a real implementation, you would evaluate the model on a validation set
        simulated_accuracy = min(0.95, 0.1 + 0.7 * (round_num / rounds))  # Simulated improvement
        logger.info("Server", f"Simulated validation accuracy after round {round_num}: {simulated_accuracy:.3f}")
        
        # Add metrics to server
        server.add_metrics(round_num, {
            'accuracy': simulated_accuracy,
            'loss': 1.0 - simulated_accuracy,
            'clients_participated': len(selected_clients)
        })
    
    # Training completed
    logger.info("Completed", f"\nTraining completed after {rounds} rounds!")
    
    # Get final metrics
    final_metrics = server.get_metrics_history()
    if final_metrics:
        latest_metrics = final_metrics[-1]
        logger.info("Completed", f"Final round metrics: {latest_metrics['metrics']}")
    
    logger.info("Completed", "Centralized FL experiment completed!")


def __parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Centralized FL MNIST experiment using FedAvg.")
    parser.add_argument("--clients", type=int, help="The number of clients.", default=5)
    parser.add_argument("--rounds", type=int, help="The number of rounds.", default=10)
    parser.add_argument("--epochs", type=int, help="The number of local epochs per round.", default=1)
    parser.add_argument("--fraction", type=float, help="Fraction of clients to use per round.", default=1)
    parser.add_argument("--framework", type=str, help="The framework to use.", default="pytorch", 
                       choices=["pytorch", "tensorflow", "flax"])
    args = parser.parse_args()
    return args

# python -m p2pfl.examples.centralized_mnist.centralized_mnist --client 10 --round 100
if __name__ == "__main__":
    # Parse arguments
    args = __parse_args()
    
    # Set standalone settings
    from p2pfl.utils.utils import set_standalone_settings
    set_standalone_settings()
    
    # Run the experiment
    centralized_mnist_experiment(
        num_clients=args.clients,
        rounds=args.rounds,
        epochs_per_round=args.epochs,
        fraction_of_clients=args.fraction
    )