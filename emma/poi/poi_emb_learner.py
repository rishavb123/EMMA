from typing import Any, Dict, Tuple
import abc
import numpy as np
import torch

from emma.poi.poi_field import POIFieldModel
from emma.components.state_samplers import StateSampler


class POIEmbLearner(abc.ABC):

    def __init__(self, poi_emb_size: int) -> None:
        super().__init__()
        self.poi_emb_size = poi_emb_size

    def set_poi_model(self, poi_model: POIFieldModel):
        self.poi_model = poi_model

    def set_device(self, device: str | None) -> None:
        self.device = device

    @abc.abstractmethod
    def generate_poi_emb(self, cur_obs: Any) -> np.ndarray:
        pass

    def train(self, inp: torch.Tensor) -> Dict[str, Any]:
        return {}


class RandomEmb(POIEmbLearner):

    def __init__(self, poi_emb_size: int) -> None:
        super().__init__(poi_emb_size)

    def generate_poi_emb(self, cur_obs: Any) -> np.ndarray:
        return np.random.normal(size=(self.poi_emb_size))


class SamplingPOILearner(POIEmbLearner):

    def __init__(
        self,
        state_sampler: StateSampler,
        poi_emb_size: int,
        device: str | None = None,
    ) -> None:
        super().__init__(poi_emb_size)
        self.state_sampler = state_sampler

    def set_device(self, device: str | None) -> None:
        super().set_device(device)
        self.state_sampler.set_device(device=device)

    def generate_poi_emb(self, cur_obs: Any) -> np.ndarray:
        return np.random.normal(size=(self.poi_emb_size))

    def train(self, inp: torch.Tensor | Dict[str, torch.Tensor]) -> Dict[str, Any]:
        if isinstance(inp, dict):
            inp = inp["state"]
        return self.state_sampler.train(inp)
