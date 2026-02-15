# System Setup & Execution Flow: P2PFL Optimized for 128-Cores

## 1. Environment & Parameter Setup

The following settings are configured to ensure maximum stability and resource utilization on high-core hardware.

| Parameter | Value | Location | Purpose |
| :--- | :--- | :--- | :--- |
| **RAY_ACTOR_POOL_SIZE** | 64 | `settings.py` | Total parallel training workers. |
| **Ray num_cpus** | 100 | `check_ray.py` | Cores dedicated to computation. |
| **OS CPU Reserve** | 28 | `check_ray.py` | Cores kept free for gRPC and System. |
| **grpc_timeout** | 60s | `.yaml` | Max wait for model transmission. |
| **wait_convergence** | 60s | `.yaml` | Handshake time for network startup. |
| **vote_timeout** | 120s | `.yaml` | Time allowed for network "election". |
| **heartbeat.timeout**| 99999s | `.yaml` | Prevent node removal during load. |
| **Torch threads** | 1 | `learner.py` | Force single thread per process. |

---

## 2. Detailed Execution Flow (Step-by-Step)

Each round in the decentralized learning process follows this strict sequence:

### Step 1: Pre-Round Stabilization
- System waits for `wait_convergence` (60s).
- gRPC Servers are initialized for all 50 nodes.
- Ray Actors are pre-allocated from the pool.

### Step 2: The "Election" Phase (Voting)
- Each node broadcasts a "Vote" to its neighbors.
- Nodes wait up to `vote_timeout` (120s) to collect votes.
- **Logic:** Each node determines the `train_set` (who will train this round).
- **Fix Applied:** If no votes arrive, the system now warns instead of crashing.

### Step 3: Local Training (Fit) Phase
- If selected in `train_set`, the node invokes `learner.fit()`.
- The task is sent to a Ray Actor.
- **ActorPool Control:** Node waits until Ray confirms the job has started (`future` assignment) before attempting to retrieve weights.
- Training runs for the configured number of `epochs`.

### Step 4: Immediate Model Protection
- **Force Add:** Immediately after training, the node calls `force_add_local_model()`.
- The model is stored in `self.__models` AND a separate `__local_model_backup`.
- **Address Normalization:** ID suffixes (like `-0`) are stripped to ensure the node recognizes its own model.

### Step 5: Communication (Gossip/Direct Send)
- The node sends its updated weights to all direct neighbors via gRPC.
- The receiving node's gRPC server handles the incoming stream and calls `add_model()`.
- **Concurrency Fix:** If two models arrive at the exact same millisecond, an `RLock` prevents data corruption.

### Step 6: The Aggregation "Wait"
- Node enters `wait_and_get_aggregation()` and waits for neighbors.
- **Fallback Logic:** If the timeout (300s) expires and no neighbor models arrived:
    1. System checks `unhandled_models` for late arrivals.
    2. System retrieves the `__local_model_backup`.
    3. The round continues using at least the local update.

### Step 7: Post-Round Cleanup
- Once `aggregate()` returns the new model, the node calls `self.clear()`.
- Aggregator internal states (train set, model list) are wiped clean.
- Node increases its round counter and loops back to Step 2.

---

## 3. Mandatory Maintenance Commands

Run these between every full experiment run to ensure a clean state:

```bash
# 1. Kill all hanging Python and Ray processes
pkill -9 python; pkill -9 raylet

# 2. Stop Ray service officially
ray stop --force

# 3. Clear previous logs (optional but recommended)
rm -rf experiments/*
```
