from typing import Any, Dict, Tuple

import abc
import torch
import hydra
import numpy as np

from emma.components.networks import VAE


class StateSampler(abc.ABC):

    def __init__(self, device: str | None) -> None:
        self.device = device

    @abc.abstractmethod
    def sample(self, cur_obs: Any, batch_size: int = 1) -> torch.Tensor:
        pass

    def train(self, inp: torch.Tensor) -> np.ndarray:
        return 0.0


class NoiseSampler(StateSampler):

    def __init__(
        self,
        device: str | None,
        low: float = 0.0,
        high: float = 1.0,
    ) -> None:
        super().__init__(device)
        self.low = low
        self.high = high

    def sample(self, cur_obs: Any, batch_size: int = 1) -> torch.Tensor:
        return torch.rand((batch_size, *cur_obs.shape))


class VAESampler(StateSampler):

    def __init__(
        self,
        vae: VAE,
        device: str | None,
        optimizer_cls: str = "torch.optim.Adam",
        optimizer_kwargs: Dict[str, Any] | None = None,
        vae_train_epochs: int = 1,
        vae_train_batch_size: int = 128,
    ) -> None:
        super().__init__(device)
        self.vae = vae
        self.vae.set_device(device=device)
        self.optimizer: torch.optim.Optimizer = hydra.utils.instantiate(
            {
                "_target_": optimizer_cls,
                **({} if optimizer_kwargs is None else optimizer_kwargs),
            },
            params=self.vae.parameters(),
        )
        self.vae_train_epochs = vae_train_epochs
        self.vae_train_batch_size = vae_train_batch_size

    def sample(self, cur_obs: Any, batch_size: int = 1) -> torch.Tensor:
        return self.vae.sample(batch_size=batch_size, condition=cur_obs)

    def train(self, inp: torch.Tensor) -> np.ndarray:
        self.vae.train(mode=True)
        n_examples = inp.shape[0]
        eff_batch_size = min(n_examples, self.vae_train_batch_size)
        if eff_batch_size == -1:
            eff_batch_size = n_examples

        total_loss = 0.0

        for _ in range(self.vae_train_epochs):
            indices = torch.randperm(n_examples)
            for start_idx in range(0, n_examples, eff_batch_size):
                end_idx = min(start_idx + eff_batch_size, n_examples)
                data = inp[indices[start_idx:end_idx]]
                self.optimizer.zero_grad()

                loss = self.vae.loss(data)

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item() * (end_idx - start_idx)

        return total_loss / (n_examples * self.vae_train_epochs)
