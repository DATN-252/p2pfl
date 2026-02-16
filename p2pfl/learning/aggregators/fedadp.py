import numpy as np
from functools import reduce
import math
from collections import defaultdict
from typing import Any
from p2pfl.learning.aggregators.aggregator import Aggregator, NoModelsToAggregateError
from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel
from p2pfl.management.logger import logger


class FedAdp(Aggregator):

    SUPPORTS_PARTIAL_AGGREGATION: bool = False
    REQUIRED_INFO_KEYS = ["global_model"]

    def __init__(self, disable_partial_aggregation: bool = False) -> None:
        """Initialize the aggregator."""
        super().__init__(disable_partial_aggregation=disable_partial_aggregation)
        self.global_model_params: list[np.ndarray] = []
        self.node_correlation  = defaultdict(lambda : 0.0)

    def aggregate(self, models: list[P2PFLModel]) -> P2PFLModel:
        """
        Aggregate the models.

        Args:
            models: Dictionary with the models (node: model,num_samples).

        Returns:
            A P2PFLModel with the aggregated.

        Raises:
            NoModelsToAggregateError: If there are no models to aggregate.

        """
        # Check if there are models to aggregate
        if len(models) == 0:
            raise NoModelsToAggregateError(f"({self.addr}) Trying to aggregate models when there is no models")
        
        # Total Samples
        total_samples = sum([m.get_num_samples() for m in models])
        # Get contributors
        contributors: list[str] = []
        for m in models:
            contributors = contributors + m.get_contributors()

        # Set init global model
        if not self.global_model_params:
            # at initial model at index 0 is its self model
            self.global_model_params = [p.copy() for p in models[0].get_parameters()]
            return models[0].build_copy(params=self.global_model_params, num_samples=total_samples, contributors=contributors)

        # Loss function = Gradient of its
        list_lossvalues: list[list[np.ndarray]]  = []
        total_lossvalue = [np.zeros_like(i) for i in self.global_model_params]
        for m in models:
            pi = m.get_parameters()
            
            # Check
            assert len(pi) == len(self.global_model_params), "Layer count mismatch"

            li =  [-(pi - pg) / self.learning_rate for pi, pg in zip(pi, self.global_model_params)]
            list_lossvalues.append(li)
            total_lossvalue = [m.get_num_samples() / float(total_samples) * l + t for l, t in  zip(li, total_lossvalue)]


        # Get weighted models
        strategy_weights = []
        g_vec = np.concatenate([p.ravel() for p in total_lossvalue]) 
        g_norm = np.linalg.norm(g_vec)
        t = self.each_trained_round[self.addr]
        for index, m in enumerate(models):
            l_vec = np.concatenate([p.ravel() for p in list_lossvalues[index]])
            l_norm = np.linalg.norm(l_vec)

            if g_norm == 0 or l_norm == 0:
                cos_sim = 0.0
            else:
                cos_sim = float(np.dot(g_vec, l_vec) / (g_norm * l_norm))
                cos_sim = np.clip(cos_sim, -1.0, 1.0)
            angle = float(np.arccos(cos_sim))

            pre_arccos_i = self.node_correlation[index]
            arccos_i = angle if (t <= 1 or pre_arccos_i == 0.0) else (((t-1)/t)*pre_arccos_i + (1/t)*angle)
            self.node_correlation[index] = arccos_i

            gompertz_value = self._gompertz_function(arccos_i)
            strategy_weights.append(m.get_num_samples() * math.exp(gompertz_value))
            
        
        # Normalize weights
        total_strategy_weights = sum(strategy_weights)
        strategy_weights = [w/total_strategy_weights for w in strategy_weights]

        # Normalize accum
        accum = [np.zeros_like(p, dtype=np.float64) for p in self.global_model_params]
        for index, m in enumerate(models):
            for i, layer in enumerate(m.get_parameters()):
                accum[i] += (layer - self.global_model_params[i]) * strategy_weights[index]

        logger.info(self.addr, "Trained round value is " + str(t))

        # Change global parameter with newest
        self.global_model_params = [g + a for g, a in zip(self.global_model_params, accum)]

        # Return an aggregated p2pfl model
        return models[0].build_copy(params=self.global_model_params, num_samples=total_samples, contributors=contributors)


    
    def _gompertz_function(self, angle: float):
        return 5*(1-math.exp(-(math.exp(-5*(angle - 1)))))
    
    def _get_and_validate_model_info(self, model: P2PFLModel) -> dict[str, Any]:
        """
        Validate the model.

        Args:
            model: The model to validate.

        """
        info = model.get_info("fedadp")
        if not all(key in info for key in self.REQUIRED_INFO_KEYS):
            raise ValueError(f"Model is missing required info keys: {self.REQUIRED_INFO_KEYS}Model info keys: {info.keys()}")
        return info
    

    def get_required_callbacks(self) -> list[str]:
        """Retrieve the list of required callback keys for this aggregator."""
        return ["fedadp"]
    