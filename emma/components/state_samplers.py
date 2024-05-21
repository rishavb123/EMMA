from typing import Any, Dict, Tuple

import abc
import torch
import hydra
import numpy as np

from emma.components.networks import VAE, reset_weights


class StateSampler(abc.ABC):

    def __init__(self) -> None:
        pass

    def reset(self) -> None:
        pass

    def set_device(self, device: str | None) -> None:
        self.device = device

    @abc.abstractmethod
    def sample(self, cur_obs: Any, batch_size: int = 1) -> torch.Tensor:
        pass

    def train(self, inp: torch.Tensor) -> Dict[str, Any]:
        return np.array(0.0)


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
        optimizer_cls: str = "torch.optim.Adam",
        optimizer_kwargs: Dict[str, Any] | None = None,
        vae_train_epochs: int = 1,
        vae_train_batch_size: int = 128,
    ) -> None:
        super().__init__()
        self.vae = vae
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = optimizer_kwargs
        self.optimizer: torch.optim.Optimizer = hydra.utils.instantiate(
            {
                "_target_": self.optimizer_cls,
                **({} if self.optimizer_kwargs is None else self.optimizer_kwargs),
            },
            params=self.vae.parameters(),
        )
        self.vae_train_epochs = vae_train_epochs
        self.vae_train_batch_size = vae_train_batch_size

    def reset(self) -> None:
        super().reset()
        reset_weights(self.vae)
        self.optimizer: torch.optim.Optimizer = hydra.utils.instantiate(
            {
                "_target_": self.optimizer_cls,
                **({} if self.optimizer_kwargs is None else self.optimizer_kwargs),
            },
            params=self.vae.parameters(),
        )

    def set_device(self, device: str | None) -> None:
        super().set_device(device)
        self.vae.set_device(device=device)

    def sample(self, cur_obs: Any, batch_size: int = 1) -> torch.Tensor:
        return self.vae.sample(batch_size=batch_size, condition=cur_obs)

    def train(self, inp: torch.Tensor) -> Dict[str, Any]:
        self.vae.train(mode=True)
        if type(inp) == dict:
            inp = inp["state"]
        elif type(inp) == tuple:
            inp = inp[0]
        n_examples = inp.shape[0]
        eff_batch_size = min(n_examples, self.vae_train_batch_size)
        if eff_batch_size == -1:
            eff_batch_size = n_examples

        total_loss = 0.0

        for epoch in range(self.vae_train_epochs):
            indices = torch.randperm(n_examples)
            for start_idx in range(0, n_examples, eff_batch_size):
                end_idx = min(start_idx + eff_batch_size, n_examples)
                data = inp[indices[start_idx:end_idx]]
                self.optimizer.zero_grad()

                loss = self.vae.loss(data)

                loss.backward()
                self.optimizer.step()

                if epoch == self.vae_train_epochs - 1:
                    total_loss += loss.item() * (end_idx - start_idx)

        return {"vae_loss": total_loss / n_examples}
