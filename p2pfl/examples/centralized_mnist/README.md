# Centralized Federated Learning with FedAvg

This example demonstrates a centralized federated learning approach using the Federated Averaging (FedAvg) algorithm, contrasting with the peer-to-peer approach implemented in the main p2pfl library.

## Overview

In centralized federated learning:
- A central server coordinates the learning process
- Clients connect to the server to participate in training
- The server aggregates model updates from clients
- This differs from the P2P approach where nodes communicate directly with each other

## Components

### CentralizedFedAvgServer
- Manages the global model
- Coordinates training rounds
- Selects subsets of clients for each round
- Aggregates client model updates using FedAvg algorithm
- Tracks metrics and convergence

### CentralizedFedAvgClient  
- Represents individual participants in federated learning
- Performs local training on private data
- Sends model updates to the central server
- Receives updated global model from server

## Algorithm

The centralized FedAvg follows these steps:

1. Server initializes global model and broadcasts to selected clients
2. Each selected client trains locally for E epochs using their private data
3. Clients send updated model weights back to server
4. Server computes weighted average of received models based on data sizes
5. Server updates global model and repeats for R rounds

## Usage

```bash
python -m p2pfl.examples.centralized_mnist.centralized_mnist --clients 5 --rounds 10 --epochs 1 --fraction 1.0
```

## Training Performance Notes

The centralized FedAvg implementation includes realistic training simulation that adds computational delays to mimic actual model training. The training time scales with:
- Number of clients participating in each round
- Number of local epochs per round
- Size of local datasets

For experiments with many clients and rounds, the training will take proportionally longer, reflecting realistic federated learning scenarios.

## Key Differences from P2P Approach

| Aspect | Centralized FedAvg | P2P FedAvg |
|--------|-------------------|------------|
| Communication | Client-Server | Direct between nodes |
| Coordination | Central server manages rounds | Distributed consensus |
| Scalability | Limited by server capacity | Potentially unlimited |
| Privacy | Server sees all updates | Updates distributed |
| Fault Tolerance | Single point of failure | Robust to node failures |

## Benefits of Centralized Approach

- Simpler coordination and synchronization
- Better control over training process
- More predictable resource usage
- Easier monitoring and logging

## When to Use Centralized vs P2P

Use centralized FedAvg when:
- You have a reliable central server
- Network conditions allow client-server communication
- You need tight control over the learning process
- Privacy concerns are addressed by other means

Use P2P FedAvg when:
- Decentralization is critical
- Network infrastructure limits central server usage
- You want to avoid single points of failure
- Privacy is paramount