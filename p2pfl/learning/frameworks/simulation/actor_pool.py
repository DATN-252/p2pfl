#
# This file is part of the federated_learning_p2p (p2pfl) distribution
# (see https://github.com/pguijas/federated_learning_p2p).
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

"""Actor pool for distributed computing using Ray."""

import threading
from typing import Any

import ray
from ray.util.actor_pool import ActorPool

from p2pfl.learning.frameworks.learner import Learner
from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel
from p2pfl.management.logger import logger
from p2pfl.settings import Settings

###
# Inspired by the implementation of Flower. Thank you so much for taking FL to another level :)
#
# Original implementation: https://github.com/adap/flower/blob/main/src/py/flwr/simulation/ray_transport/ray_actor.py
###


@ray.remote
class VirtualLearnerActor:
    """Decorator for the learner to be used in the simulation."""

    def terminate(self) -> None:
        """Manually terminate Actor object."""
        logger.debug(self.__class__.__name__, f"Manually terminating {self.__class__.__name__}")
        ray.actor.exit_actor()

    def fit(self, addr: str, learner: Learner, apply_update: bool = True) -> tuple[str, P2PFLModel]:
        """Fit the model."""
        try:
            model = learner.fit(apply_update=apply_update)

        except Exception as ex:
            raise ex

        return addr, model

    def evaluate(self, addr: str, learner: Learner) -> tuple[str, dict[str, float]]:
        """Evaluate the model."""
        try:
            results = learner.evaluate()

        except Exception as ex:
            raise ex

        return addr, results


class SuperActorPool(ActorPool):
    """
    SuperActorPool extends ActorPool to manage a pool of VirtualLearnerActor instances for asynchronous distributed computing using Ray.

    Attributes:
        resources (dict): Resources for actor creation.
        _addr_to_future (dict): Mapping from actor address to future information.
        actor_to_remove (set): Set of actor IDs scheduled for removal.
        num_actors (int): Number of active actors in the pool.
        lock (threading.RLock): Reentrant lock for thread-safe operations.

    """

    def __init__(self, amount_actors: int = 1):
        """
        Initialize SuperActorPool.

        Args:
            amount_actors: Number of actors to initialize. Defaults to 1.

        """
        # Calculate resources per actor (Dynamic based on node's perspective)
        self.gpu_per_actor = self._calculate_gpu_per_actor(amount_actors)
        self.cpu_per_actor = self._calculate_cpu_per_actor(amount_actors)

        actors = [self.create_actor() for _ in range(amount_actors)]
        self.num_actors = len(actors)
        logger.debug("ActorPool", f"Initialized local pool with {self.num_actors} actors")
        super().__init__(actors)

        # A dict that maps addr to another dict containing: a reference to the remote job
        # and its status (i.e. whether it is ready or not)
        self._addr_to_future: dict[str, dict[str, Any]] = {}
        # a set of actor ids to be removed
        self.actor_to_remove: set[str] = set()

        self.lock = threading.RLock()

    def _calculate_gpu_per_actor(self, num_actors: int) -> float:
        """Calculate GPU fraction per actor."""
        # On n2d-highmem-8 without GPU, we force this to 0
        return 0.0 

    def _calculate_cpu_per_actor(self, num_actors: int) -> float:
        """Calculate CPU fraction per actor."""
        # 50 nodes * 1 actor/node * 0.1 CPU = 5 CPUs. 
        # This leaves 3 CPUs for gRPC and OS overhead on an 8 vCPU machine.
        return 0.1

    def create_actor(self) -> VirtualLearnerActor:
        """
        Create a new VirtualLearnerActor instance using provided resources.

        Returns:
            New actor instance.

        """
        # Create actor with resources if available
        options = {}
        if hasattr(self, "gpu_per_actor") and self.gpu_per_actor > 0:
            options["num_gpus"] = self.gpu_per_actor
        if hasattr(self, "cpu_per_actor") and self.cpu_per_actor > 0:
            options["num_cpus"] = self.cpu_per_actor

        if options:
            return VirtualLearnerActor.options(**options).remote()  # type: ignore
        else:
            return VirtualLearnerActor.options().remote()  # type: ignore

    def add_actor(self, num_actors: int) -> None:
        """
        Add a specified number of actors to the pool.

        Args:
            num_actors: Number of actors to add.

        """
        with self.lock:
            new_actors = [self.create_actor() for _ in range(num_actors)]
            self._idle_actors.extend(new_actors)
            self.num_actors += num_actors
            logger.info("ActorPool", f"Created {num_actors} actors")

    def submit(self, fn: Any, value: tuple[str, Learner]) -> None:
        """
        Submit a task to an idle actor in the pool.

        Args:
            fn: Function to be executed by the actor.
            value: Tuple containing address and learner information.

        """
        addr, learner = value
        actor = self._idle_actors.pop()

        if self._check_and_remove_actor_from_pool(actor):
            future = fn(actor, addr, learner)
            future_key = tuple(future) if isinstance(future, list) else future
            
            # Ensure future is assigned before releasing the lock
            # This prevents get_learner_result from seeing a None future
            self._future_to_actor[future_key] = (self._next_task_index, actor, addr)
            self._next_task_index += 1
            if addr not in self._addr_to_future:
                self._addr_to_future[addr] = {}
            self._addr_to_future[addr]["future"] = future_key
        else:
            logger.error("ActorPool", "Actor should have been removed from pool but wasn't")

    def submit_learner_job(self, actor_fn: Any, job: tuple[str, Learner]) -> None:
        """
        Submit a learner job to the pool, handling pending submits if no idle actors are available.

        Args:
            actor_fn: Function to be executed by the actor.
            job: Tuple containing address and learner information.

        """
        addr, _ = job
        # We use a combined key to distinguish between different jobs from the same address
        # (e.g., if a node submits evaluate immediately after fit)
        # However, the current framework uses addr as the primary lookup.
        # To minimize changes, we just ensure the reset is thread-safe and the job is tracked.
        with self.lock:
            self._reset_addr_to_future_dict(addr)
            if self._idle_actors:
                self.submit(actor_fn, job)
            else:
                self._pending_submits.append((actor_fn, job))
                logger.debug("ActorPool", f"Job for {addr} added to pending queue.")

    def _flag_future_as_ready(self, addr: str) -> None:
        """
        Flag the future associated with the given address as ready.

        Args:
            addr: Address of the future to flag.

        """
        self._addr_to_future[addr]["ready"] = True

    def _reset_addr_to_future_dict(self, addr: str) -> None:
        """
        Reset the future dictionary for a given address.

        Args:
            addr: Address to reset in the future dictionary.

        """
        if addr not in self._addr_to_future:
            self._addr_to_future[addr] = {}
        self._addr_to_future[addr]["future"] = None
        self._addr_to_future[addr]["ready"] = False

    def _is_future_ready(self, addr: str) -> bool:
        """
        Check if the future associated with the given address is ready.

        Args:
            addr: Address of the future to check.

        Returns:
            True if future is ready, False otherwise.

        """
        if addr not in self._addr_to_future:
            logger.error("ActorPool", "Future associated with the given address not found. This shouldn't be happening")
            return False
        return self._addr_to_future[addr]["ready"]  # type: ignore

    def _fetch_future_result(self, addr: str) -> tuple[Any, Any]:
        """
        Fetch the result of a future associated with the given address.

        Args:
            addr: Address of the future to fetch.

        Returns:
            Address and result fetched from the future.

        """
        try:
            future = self._addr_to_future[addr]["future"]
            if future is None:
                raise ValueError(f"No future job found for address {addr}. The job might have failed to submit or was already processed.")
            res_addr, result = ray.get(future)
        except ray.exceptions.RayActorError as ex:
            # print(ex)
            if hasattr(ex, "actor_id"):
                self._flag_actor_for_removal(ex.actor_id)
            raise ex
        assert res_addr == addr
        self._reset_addr_to_future_dict(addr)
        return res_addr, result

    def _flag_actor_for_removal(self, actor_id_hex: str) -> None:
        """
        Flag a specified actor for removal from the pool.

        Args:
            actor_id_hex: ID of the actor to be removed.

        """
        with self.lock:
            self.actor_to_remove.add(actor_id_hex)
            logger.debug("ActorPool", f"Actor({actor_id_hex}) will be removed from pool.")

    def _check_and_remove_actor_from_pool(self, actor: VirtualLearnerActor) -> bool:
        """
        Check if an actor should be removed from the pool based on flagged removals.

        Args:
            actor: Actor instance to check.

        Returns:
            True if the actor should not be removed, False otherwise.

        """
        with self.lock:
            actor_id = actor._actor_id.hex()  # type: ignore
            if actor_id in self.actor_to_remove:
                self.actor_to_remove.remove(actor_id)
                self.num_actors -= 1
                logger.debug("ActorPool", f"REMOVED actor {actor_id} from pool")
                return False
            return True

    def process_unordered_future(self, timeout: float | None = None) -> None:
        """
        Process the next unordered future result from the pool.

        Args:
            timeout: Timeout for processing the future. Defaults to None.

        Raises:
            StopIteration: If no more results are available.
            TimeoutError: If the future processing times out.

        """
        if not self.has_next():  # type: ignore
            raise StopIteration("No more results to get")
        res, _ = ray.wait(list(self._future_to_actor), num_returns=1, timeout=timeout)
        if res:
            [future] = res
        else:
            raise TimeoutError("Timed out waiting for result")
        with self.lock:
            _, actor, addr = self._future_to_actor.pop(future, (None, None, -1))
            if actor is not None:
                if self._check_and_remove_actor_from_pool(actor):
                    self._return_actor(actor)  # type: ignore
                self._flag_future_as_ready(addr)

    def get_learner_result(self, addr: str, timeout: float | None) -> tuple[Any, Any]:
        """
        Retrieve the learner result associated with the given address.

        Args:
            addr: Address of the learner result to retrieve.
            timeout: Timeout for retrieving the result. Defaults to None.

        Returns:
            Address and result of the learner job.

        """
        import time
        start_time = time.time()
        
        # Wait until the job is actually submitted (not in pending_submits anymore)
        # and has a valid future reference.
        while True:
            future = None
            with self.lock:
                future = self._addr_to_future.get(addr, {}).get("future")
            
            if future is not None:
                break
            
            # If we've been waiting too long for submission, something is wrong
            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(f"Timed out waiting for job {addr} to be submitted to Ray.")
            
            # Give Ray time to process submissions by releasing the lock and sleeping
            time.sleep(0.1)

        # Now that we have a future, wait for it to be ready
        while self.has_next() and not self._is_future_ready(addr):  # type: ignore
            try:
                self.process_unordered_future(timeout=timeout)
            except StopIteration:
                break
        return self._fetch_future_result(addr)
