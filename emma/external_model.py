from typing import Dict, Type

import abc
import torch
from torch import nn
import logging
import numpy as np

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.vec_env import VecEnv


logger = logging.getLogger(__name__)


class MCModule(nn.Module):

    def __init__(
        self, inner_module: nn.Module, *args, num_samples: int = 30, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.inner_module = inner_module
        self.num_samples = num_samples

    def generate_samples(self, x: torch.Tensor):
        self.inner_module.train(mode=True)
        outputs = []

        for _ in range(self.num_samples):
            outputs.append(self.inner_module.forward(x))

        return torch.stack(outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.generate_samples(x).mean(dim=0)

    def uncertainty_estimate(self, x: torch.Tensor) -> torch.Tensor:
        stddevs = self.generate_samples(x).std(dim=0)
        return stddevs.mean(dim=tuple(range(len(stddevs.shape)))[1:])

    def forward_and_uncertainty_estimate(self, x: torch.Tensor) -> torch.Tensor:
        samples = self.generate_samples(x)
        stddevs = samples.std(dim=0)
        return samples.mean(dim=0), stddevs.mean(
            dim=tuple(range(len(stddevs.shape)))[1:]
        )


class ExternalModelTrainer(abc.ABC):

    model: torch.nn.Module | None = None

    def __init__(
        self,
        model: torch.nn.Module,
        device: str,
        loss_type: Type[torch.nn.Module],
        lr: float = 0.001,
        dtype=torch.float32,
    ) -> None:
        self.device = device
        self.dtype = dtype
        self.model = model.to(device=device, dtype=dtype)
        self.loss_type = loss_type
        self.loss_f = self.loss_type().to(device=device, dtype=dtype)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    @abc.abstractmethod
    def rollout_to_model_input(
        self,
        env: VecEnv,
        rollout_buffer: RolloutBuffer,
        info_buffer: Dict[str, np.ndarray],
    ) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def rollout_to_model_output(
        self,
        env: VecEnv,
        rollout_buffer: RolloutBuffer,
        info_buffer: Dict[str, np.ndarray],
    ) -> torch.Tensor:
        pass

    def receive_rollout(
        self,
        env: VecEnv,
        rollout_buffer: RolloutBuffer,
        info_buffer: Dict[str, np.ndarray],
    ) -> None:
        inp = self.rollout_to_model_input(
            env=env, rollout_buffer=rollout_buffer, info_buffer=info_buffer
        ).to(device=self.device, dtype=self.dtype)
        out = self.rollout_to_model_output(
            env=env, rollout_buffer=rollout_buffer, info_buffer=info_buffer
        ).to(device=self.device, dtype=self.dtype)

        self.optimizer.zero_grad()

        self.model.train(mode=True)
        pred_out = self.model(inp)
        loss = self.loss_f(out, pred_out)
        loss.backward()

        self.optimizer.step()

        return loss.item()
