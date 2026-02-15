from typing import Any
import numpy as np
from p2pfl.learning.aggregators.aggregator import Aggregator, NoModelsToAggregateError
from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel


class PSGD(Aggregator):
    """
    Decentralized Parallel SGD (D-PSGD)
    """

    SUPPORTS_PARTIAL_AGGREGATION = True
    requires_gradient_only: bool = False
    REQUIRED_INFO_KEYS = ["degrees"]

    def __init__(self, lr: float = 0.01):
        """
        Args:
            lr: step size γ (default 0.01)
        """
        super().__init__()
        self.lr = lr


    def aggregate(self, models: list[P2PFLModel]) -> P2PFLModel:
        if not models:
            raise NoModelsToAggregateError(f"({self.addr}) No models to aggregate")

        # 1. Find the model from the current node to access its info later
        self_idx = -1
        norm_self_addr = self.normalize_addr(self.addr)
        for i, m in enumerate(models):
            contributors = m.get_contributors()
            if contributors and norm_self_addr == self.normalize_addr(contributors[0]):
                self_idx = i
                break

        if self_idx == -1:
            raise NoModelsToAggregateError(f"Self model ({norm_self_addr}) not found in the aggregation list for PSGD.")
        
        self_model = models[self_idx]

        # 2. Calculate Metropolis-Hastings Weights (Topology-based)
        model_info = [self._get_and_validate_model_info(m) for m in models]
        degrees = [int(info["degrees"]) for info in model_info]
        my_degree = degrees[self_idx]
        
        weights = [0.0] * len(models)
        # Calculate neighbor weights
        for i in range(len(models)):
            if i == self_idx:
                continue
            weights[i] = 1.0 / (1.0 + max(my_degree, degrees[i]))
        
        # Calculate self-weight
        weights[self_idx] = 1.0 - sum(weights)
        
        # 3. Consensus Step: x_{k+1, i} = Σ_j w_ij * x_{k,j}
        # In P2PFL, models in 'models' list are already trained locally (x_j = x_old + delta_j).
        # The aggregation (mixing) of these trained models completes the D-PSGD step.
        x_params = self_model.get_parameters()
        final_params = [np.zeros_like(p, dtype=np.float64) for p in x_params]

        for i, m in enumerate(models):
            w_ij = weights[i]
            if w_ij > 0:
                for l, param in enumerate(m.get_parameters()):
                    final_params[l] += w_ij * param

        # 4. Return the final updated model
        return self_model.build_copy(
            params=final_params,
            num_samples=self_model.get_num_samples(),
            contributors=self_model.get_contributors(),
        )

    def _get_and_validate_model_info(self, model: P2PFLModel) -> dict[str, Any]:
        try:
            # Look for info attached by a specific callback if it exists
            info = model.get_info("gradient_delta_calculator")
        except KeyError:
            info = model.get_info()
        
        for key in self.REQUIRED_INFO_KEYS:
            if key not in info:
                raise ValueError(f"Model missing '{key}' information required for PSGD.")
        return info

    def get_required_callbacks(self) -> list[str]:
        return ["gradient_delta_calculator"]
