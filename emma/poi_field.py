from typing import Dict

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
        env: VecEnv,
        rollout_buffer: RolloutBuffer,
        info_buffer: Dict[str, np.ndarray],
    ) -> np.ndarray:
        pass


class ZeroPOIField(POIFieldModel):

    def __init__(self, external_model_trainer: ExternalModelTrainer) -> None:
        super().__init__(external_model_trainer)

    def calculate_poi_values(
        self,
        env: VecEnv,
        rollout_buffer: RolloutBuffer,
        info_buffer: Dict[str, np.ndarray],
    ) -> np.ndarray:
        return np.zeros_like(rollout_buffer.rewards)


class LossPOIField(POIFieldModel):  # beta: 0.001 for correct key distance

    def __init__(self, external_model_trainer: ExternalModelTrainer) -> None:
        super().__init__(external_model_trainer)
        self.unaggregated_loss_f = self.external_model_trainer.loss_type(
            reduction="none"
        )

    def calculate_poi_values(
        self,
        env: VecEnv,
        rollout_buffer: RolloutBuffer,
        info_buffer: Dict[str, np.ndarray],
    ) -> np.ndarray:
        with torch.no_grad():
            self.external_model_trainer.model.train(mode=False)
            model_inp = self.external_model_trainer.rollout_to_model_input(
                env=env, rollout_buffer=rollout_buffer, info_buffer=info_buffer
            )
            gt_out = self.external_model_trainer.rollout_to_model_output(
                env=env, rollout_buffer=rollout_buffer, info_buffer=info_buffer
            )
            model_out = self.external_model_trainer.predict(model_inp)

            return (
                self.unaggregated_loss_f(gt_out, model_out)
                .cpu()
                .numpy()
                .reshape(rollout_buffer.rewards.shape)
            )


class MCDropoutPOIField(POIFieldModel):  # beta: 0.1 for correct key distance

    def __init__(
        self, external_model_trainer: ExternalModelTrainer, num_samples: int = 30
    ) -> None:
        super().__init__(external_model_trainer)
        self.num_samples = num_samples

    def calculate_poi_values(
        self,
        env: VecEnv,
        rollout_buffer: RolloutBuffer,
        info_buffer: Dict[str, np.ndarray],
    ) -> np.ndarray:
        with torch.no_grad():
            model_inp = self.external_model_trainer.rollout_to_model_input(
                env=env, rollout_buffer=rollout_buffer, info_buffer=info_buffer
            )
            samples = []
            self.external_model_trainer.model.train(mode=True)
            for _ in range(self.num_samples):
                model_out = self.external_model_trainer.predict(model_inp)
                samples.append(model_out.cpu().numpy())
            samples = np.array(samples)
            return (
                samples.std(axis=0)
                .mean(axis=tuple(range(len(samples.shape) - 1))[1:])
                .reshape(rollout_buffer.rewards.shape)
            )


class ModelGradientPOIField(POIFieldModel):  # beta: 10 for correct key distance

    def __init__(self, external_model_trainer: ExternalModelTrainer) -> None:
        super().__init__(external_model_trainer)

    def calculate_poi_values(
        self,
        env: VecEnv,
        rollout_buffer: RolloutBuffer,
        info_buffer: Dict[str, np.ndarray],
    ) -> np.ndarray:
        model_inp = self.external_model_trainer.rollout_to_model_input(
            env=env, rollout_buffer=rollout_buffer, info_buffer=info_buffer
        )
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

        return np.array(grad_lst).reshape(rollout_buffer.rewards.shape)
