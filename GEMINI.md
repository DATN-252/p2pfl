# GEMINI.md - P2PFL Context (Main Branch - Legacy Design)

## Project Overview
**P2PFL** is a decentralized federated learning (DFL) framework designed for peer-to-peer networks. This version (Main branch) represents the core stable architecture before the introduction of advanced high-core optimizations and round-aware synchronization.

### Key Technologies
- **Language:** Python (>=3.10, <3.13)
- **Communication:** gRPC, Protobuf
- **Simulation:** Ray
- **ML Frameworks:** PyTorch, TensorFlow, JAX

---

## Core Architecture & Workflow (Legacy)

### 1. Node & State Management
- **Workflow:** Sequential execution of stages defined in `p2pfl/stages/`.
- **Synchronization:** Relies on standard `threading.Lock` and `threadate can lead to race conditions in high-latency or high-jitter environments.

### 3. Simulation & Resources
- **Ray Integration:** Uses Ray for actor-based simulation.
- **Resource Allocation:** Basic fraction-based allocation for CPU/GPU.

---

## Comparison: Legacy vs. Optimized Design

| Feature | Legacy Design (Main) | Optimized Design (Previous Branch) |
| :--- | :--- | :--- |
| **Aggregator Lock** | `threading.Lock` (Non-reentrant) | `threading.RLock` (Reentrant) |
| **Buffering** | Simple list-based `unhandled_models` | **Round-aware dictionary** buffering (`round_num -> list`) |
| **Wait Strategy** | Static timeout | **Dynamic Patience** (Wait longer if neighbors are still active) |
| **Round Awareness** | Implicit in workflow | **Explicitly tracked** in aggregator (`set_nodes_to_aggregate` takes `round_num`) |
| **Recovery** | Basic | **Proactive Recovery** (auto-injects buffered models when round starts) |
| **Reliability** | Susceptible to "late arrivals" from previous rounds | **Local Model Backup** and round-filtering to prevent stale data |
| **System Docs** | Missing `SYSTEM_SPECIFICATION.md` | Detailed 128-core optimization guide available |
| **Batch Tools** | Missing `run_batch_experiments.py` | Full batch execution and metrics plotting tools |

### Design Conclusion
The **Legacy Design** is functional for standard P2PFL scenarios but lacks the robustness required for large-scale, high-concurrency simulations. The **Optimized Design** (found in the previous branch) introduced critical synchronization primitives (RLock, Condition variables) and "Dynamic Patience" to handle the non-determinism of 100+ core environments where Ray scheduling and gRPC latency can cause significant round-drift between nodes.
