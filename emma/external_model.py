from typing import Any, Dict, SupportsFloat, Type

import abc
import gymnasium as gym
import numpy as np
import torch
import logging
import wandb
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.vec_env import VecEnv


logger = logging.getLogger(__name__)


class ExternalModelTrainer(abc.ABC):

    model: torch.nn.Module | None = None

    def __init__(
        self,
        model: torch.nn.Module,
        device: str,
        loss_type: Type[torch.nn.Module],
        lr: float = 0.001,
    ) -> None:
        self.model = model.to(device=device)
        self.loss_type = loss_type
        self.loss_f = self.loss_type()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    @abc.abstractmethod
    def rollout_to_model_input(
        self,
        env: VecEnv,
        rollout_buffer: RolloutBuffer,
    ) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def rollout_to_model_output(
        self,
        env: VecEnv,
        rollout_buffer: RolloutBuffer,
    ) -> torch.Tensor:
        pass

    def receive_rollout(self, env: VecEnv, rollout_buffer: RolloutBuffer):
        inp = self.rollout_to_model_input(env=env, rollout_buffer=rollout_buffer)
        out = self.rollout_to_model_output(env=env, rollout_buffer=rollout_buffer)

        self.optimizer.zero_grad()

        pred_out = self.predict(inp=inp)
        loss = self.loss_f(out, pred_out)
        loss.backward()

        self.optimizer.step()

        return loss.item()

    def predict(self, inp: torch.Tensor) -> torch.Tensor:
        return self.model(inp)


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

        av_loss = self.model_trainer.receive_rollout(
            env=env, rollout_buffer=self.model.rollout_buffer
        )

        logger.info(f"Loss: {av_loss}")
        if self.wandb_mode != "disabled":
            wandb.log(
                {
                    "external_model_loss": av_loss,
                }
            )
