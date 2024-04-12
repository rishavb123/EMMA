from typing import Dict, Tuple

import abc
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.vec_env import VecEnv
import numpy as np
import torch

from emma.external_model import ExternalModelTrainer


class POIFieldModel(abc.ABC):

    def __init__(self, external_model_trainer: ExternalModelTrainer) -> None:
        super().__init__()
        self.external_model_trainer = external_model_trainer

    @abc.abstractmethod
    def calculate_poi_values(
        self,
        model_inp: torch.Tensor,
        poi_shape: Tuple,
    ) -> np.ndarray:
        pass


class ZeroPOIField(POIFieldModel):

    def __init__(self, external_model_trainer: ExternalModelTrainer) -> None:
        super().__init__(external_model_trainer)

    def calculate_poi_values(
        self,
        model_inp: torch.Tensor,
        poi_shape: Tuple,
    ) -> np.ndarray:
        return np.zeros_like(poi_shape)


class DisagreementPOIField(POIFieldModel):

    def __init__(
        self, external_model_trainer: ExternalModelTrainer, num_samples: int = 30
    ) -> None:
        super().__init__(external_model_trainer)
        self.num_samples = num_samples

    def calculate_poi_values(
        self,
        model_inp: torch.Tensor,
        poi_shape: Tuple,
    ) -> np.ndarray:
        with torch.no_grad():
            uncertainty: torch.Tensor = (
                self.external_model_trainer.model.uncertainty_estimate(model_inp)
            )
            return uncertainty.cpu().numpy().reshape(poi_shape)


class ModelGradientPOIField(POIFieldModel):

    def __init__(self, external_model_trainer: ExternalModelTrainer) -> None:
        super().__init__(external_model_trainer)

    def calculate_poi_values(
        self,
        model_inp: torch.Tensor,
        poi_shape: Tuple,
    ) -> np.ndarray:
        self.external_model_trainer.model.train(mode=False)
        model_out = self.external_model_trainer.model(model_inp)

        grad_lst = []

        for i in range(model_out.shape[0]):
            out = model_out[i].mean()
            param_grads = torch.autograd.grad(
                out,
                list(self.external_model_trainer.model.parameters()),
                retain_graph=True,
            )
            grad_mean = np.concatenate(
                [grad.cpu().numpy().flatten() for grad in param_grads]
            ).mean()
            grad_lst.append(grad_mean)

        return np.array(grad_lst).reshape(poi_shape)
