from typing import Any, Dict, SupportsFloat

import abc
import gymnasium as gym
import numpy as np
import torch
import logging
import wandb
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.type_aliases import RolloutBufferSamples
from stable_baselines3.common.vec_env import VecEnv


logger = logging.getLogger(__name__)


class ExternalModelTrainer(abc.ABC):

    model: torch.nn.Module | None = None

    def __init__(
        self, model: torch.nn.Module, device: str, loss_f: torch.nn.Module, lr: float = 0.001
    ) -> None:
        self.model = model.to(device=device)
        self.loss_f = loss_f
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    @abc.abstractmethod
    def rollout_to_model_input(
        self,
        env: VecEnv,
        rollout_buffer_samples: RolloutBufferSamples,
    ) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def rollout_to_model_output(
        self,
        env: VecEnv,
        rollout_buffer_samples: RolloutBufferSamples,
    ) -> torch.Tensor:
        pass

    def receive_rollout(
        self, env: VecEnv, rollout_buffer_samples: RolloutBufferSamples
    ):
        inp = self.rollout_to_model_input(
            env=env, rollout_buffer_samples=rollout_buffer_samples
        )
        out = self.rollout_to_model_output(
            env=env, rollout_buffer_samples=rollout_buffer_samples
        )

        self.optimizer.zero_grad()

        pred_out = self.model(inp)
        loss = self.loss_f(out, pred_out)
        loss.backward()

        self.optimizer.step()

        return loss.item()

    def predict(self, obs: Any) -> Any:
        return ExternalModelTrainer.model(self.obs_to_model_inp(obs=obs))


class ExternalModelTrainerCallback(BaseCallback):

    def __init__(
        self,
        model_trainer: ExternalModelTrainer,
        batch_size: int = 32,
        wandb_mode: str | None = None,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.model_trainer = model_trainer
        self.batch_size = batch_size
        self.wandb_mode = wandb_mode

    def _on_step(self) -> bool:
        return super()._on_step()

    def _on_training_start(self) -> None:
        return super()._on_training_start()

    def _on_training_end(self) -> None:
        return super()._on_training_end()

    def _on_rollout_start(self) -> None:
        return super()._on_rollout_start()

    def _on_rollout_end(self) -> None:
        super()._on_rollout_end()
        env = self.model.get_env()

        count = 0
        total_loss = 0

        for rollout_data in self.model.rollout_buffer.get(self.batch_size):
            total_loss += self.model_trainer.receive_rollout(
                env=env, rollout_buffer_samples=rollout_data
            )
            count += 1

        av_loss = total_loss / count
        logger.info(f"Loss: {av_loss}")
        if self.wandb_mode != "disabled":
            wandb.log(
                {
                    "external_model_loss": av_loss,
                }
            )
