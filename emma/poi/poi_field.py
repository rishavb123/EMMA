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

    def reset(self) -> None:
        pass

    @abc.abstractmethod
    def _calculate_poi_values(
        self,
        model_inp: (
            torch.Tensor | Tuple[torch.Tensor, torch.Tensor] | Dict[str, torch.Tensor]
        ),
    ) -> np.ndarray:
        pass

    def calculate_poi_values(
        self,
        model_inp: (
            torch.Tensor | Tuple[torch.Tensor, torch.Tensor] | Dict[str, torch.Tensor]
        ),
    ) -> np.ndarray:
        if self.external_model_trainer.model is None:
            if type(model_inp) == dict:
                size = model_inp["state"].shape[0]
            elif type(model_inp) == tuple:
                size = model_inp[0].shape[0]
            else:
                size = model_inp.shape[0]
            return np.zeros(size)
        else:
            return self._calculate_poi_values(model_inp=model_inp)


class ZeroPOIField(POIFieldModel):

    def __init__(self, external_model_trainer: ExternalModelTrainer) -> None:
        super().__init__(external_model_trainer)

    def _calculate_poi_values(
        self,
        model_inp: (
            torch.Tensor | Tuple[torch.Tensor, torch.Tensor] | Dict[str, torch.Tensor]
        ),
    ) -> np.ndarray:
        if type(model_inp) == tuple:
            n_examples = model_inp[0].shape[0]
        elif type(model_inp) == dict:
            n_examples = model_inp["state"].shape[0]
        else:
            n_examples = model_inp.shape[0]
        return np.zeros(n_examples)


class DisagreementPOIField(POIFieldModel):

    def __init__(
        self, external_model_trainer: ExternalModelTrainer, num_samples: int = 30
    ) -> None:
        super().__init__(external_model_trainer)
        self.num_samples = num_samples

    def _calculate_poi_values(
        self,
        model_inp: (
            torch.Tensor | Tuple[torch.Tensor, torch.Tensor] | Dict[str, torch.Tensor]
        ),
    ) -> np.ndarray:
        with torch.no_grad():
            uncertainty: torch.Tensor = (
                self.external_model_trainer.model.uncertainty_estimate(model_inp)
            )
            return uncertainty.cpu().numpy()


class ModelGradientPOIField(POIFieldModel):

    def __init__(self, external_model_trainer: ExternalModelTrainer) -> None:
        super().__init__(external_model_trainer)

    def _calculate_poi_values(
        self,
        model_inp: (
            torch.Tensor | Tuple[torch.Tensor, torch.Tensor] | Dict[str, torch.Tensor]
        ),
    ) -> np.ndarray:
        self.external_model_trainer.model.train(mode=False)
        model_out = self.external_model_trainer.predict(model_inp)

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

        return np.array(grad_lst)
