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
        self, env: VecEnv, rollout_buffer: RolloutBuffer
    ) -> np.ndarray:
        pass


class ZeroPOIField(POIFieldModel):

    def __init__(self, external_model_trainer: ExternalModelTrainer) -> None:
        super().__init__(external_model_trainer)

    def calculate_poi_values(
        self, env: VecEnv, rollout_buffer: RolloutBuffer
    ) -> np.ndarray:
        return np.zeros_like(rollout_buffer.rewards)


class LossPOIField(POIFieldModel):

    def __init__(self, external_model_trainer: ExternalModelTrainer) -> None:
        super().__init__(external_model_trainer)
        self.unaggregated_loss_f = self.external_model_trainer.loss_type(
            reduction="none"
        )

    def calculate_poi_values(
        self, env: VecEnv, rollout_buffer: RolloutBuffer
    ) -> np.ndarray:
        with torch.no_grad():
            model_inp = self.external_model_trainer.rollout_to_model_input(
                env=env, rollout_buffer=rollout_buffer
            )
            gt_out = self.external_model_trainer.rollout_to_model_output(
                env=env, rollout_buffer=rollout_buffer
            )
            model_out = self.external_model_trainer.predict(model_inp)

            return (
                self.unaggregated_loss_f(gt_out, model_out)
                .cpu()
                .numpy()
                .reshape(rollout_buffer.rewards.shape)
            )
