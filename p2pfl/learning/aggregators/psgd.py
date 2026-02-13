from typing import Any
import numpy as np
from p2pfl.learning.aggregators.aggregator import Aggregator, NoModelsToAggregateError
from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel


class PSGD(Aggregator):
    """
    Decentralized Parallel SGD (D-PSGD)
    """

    SUPPORTS_PARTIAL_AGGREGATION = True
    requires_gradient_only: bool = True
    REQUIRED_INFO_KEYS = ["delta", "degrees"]

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
        self_model = next((m for m in models if self.addr in m.get_contributors()), None)

        if self_model is None:
            # Fallback or raise error if the node's own model is not in the list
            raise NoModelsToAggregateError("Self model not found in the aggregation list.")

        # 2. Calculate Metropolis-Hastings Weights (Topology-based)
        # This assumes the self_model is the first in the list, which is the p2pfl convention.
        degrees = [int(self._get_and_validate_model_info(m)["degrees"]) for m in models]
        weights = [0.0] * len(models)
        my_degree = degrees[0]
        
        # Calculate neighbor weights
        for i in range(1, len(models)):
            weights[i] = 1.0 / (1.0 + max(my_degree, degrees[i]))
        
        # Calculate self-weight
        weights[0] = 1.0 - sum(weights[1:])
        
        # 3. Consensus Step: x_{k+1/2, i} = Σ_j w_ij * x_{k,j}
        x_params = self_model.get_parameters()
        mixed_params = [np.zeros_like(p) for p in x_params]

        for i, m in enumerate(models):
            w_ij = weights[i]
            if w_ij > 0:
                for l, param in enumerate(m.get_parameters()):
                    mixed_params[l] += w_ij * param

        # 3. Gradient Step: x_{k+1, i} = x_{k+1/2, i} + delta
        # The learner provides `delta` which is equal to `-γ * ∇F`.
        # So, the update is x_{k+1/2, i} + delta.
        
        info = self._get_and_validate_model_info(self_model)
        delta = info["delta"]

        final_params = [m_param + d_param for m_param, d_param in zip(mixed_params, delta)]

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
