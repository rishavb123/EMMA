from typing import Any, Callable, Tuple

import torch
from torch import nn


class VAE(nn.Module):

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        latent_dim: int,
        reconstruction_loss_f: nn.Module | None = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = latent_dim

        self.reconstruction_loss_f = (
            nn.BCELoss() if reconstruction_loss_f is None else reconstruction_loss_f
        )

    def set_device(self, device: str | None):
        self.device = None if device is None else torch.device(device=device)
        self.encoder = self.encoder.to(self.device)
        self.decoder = self.decoder.to(self.device)
        self.reconstruction_loss_f = self.reconstruction_loss_f.to(self.device)

    def reparameterization(self, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        epsilon = torch.randn_like(mean, device=self.device)
        return mean + std * epsilon

    def sample_noise(self, batch_size: int = 1, condition: Any = None):
        return torch.randn((batch_size, self.latent_dim), device=self.device)

    def sample(self, batch_size: int = 1, condition: Any = None):
        with torch.no_grad():
            noise = self.sample_noise(batch_size=batch_size, condition=condition)
            return self.decoder(noise)

    def loss(self, x: torch.Tensor) -> torch.Tensor:
        x_hat, mean, logvar = self.forward(x=x)
        reconstruction_loss = self.reconstruction_loss_f(x_hat, x)
        kl_divergence_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        return reconstruction_loss + kl_divergence_loss

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, logvar = self.encoder(x)
        std = torch.exp(0.5 * logvar)
        z = self.reparameterization(mean=mean, std=std)
        x_hat = self.decoder(z)
        return x_hat, mean, logvar


class PermutationInvariantNetwork(nn.Module):

    def __init__(
        self,
        phi: nn.Module,
        rho: nn.Module,
        mixer: Callable[[torch.Tensor, int], torch.Tensor] | None = torch.mean,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.phi = phi
        self.mixer = mixer
        self.rho = rho

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # x: (batch_size, set_size, *element_shape) or (set_size, *element_shape)
        z = self.phi(x)
        # z: (batch_size, set_size, hidden_size) or (set_size, hidden_size)
        
        mixdim = len(z.shape) - 2
        if self.mixer:
            z = self.mixer(z, dim=mixdim)

        return self.rho(z)

