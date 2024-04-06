from typing import Any, Callable, Dict, List, Tuple, Type

import abc
from gymnasium import Space
import torch
from torch import nn
import logging
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import (
    MlpExtractor,
    FlattenExtractor,
    BaseFeaturesExtractor,
)

from torch.optim.adam import Adam as Adam
from torch.optim.optimizer import Optimizer as Optimizer


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

        return torch.stack(outputs).to(dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.generate_samples(x).mean(dim=0)

    def uncertainty_estimate(self, x: torch.Tensor) -> torch.Tensor:
        stddevs = self.generate_samples(x).std(dim=0)
        mean_dims = tuple(range(len(stddevs.shape)))[1:]
        if mean_dims == ():
            return stddevs
        else:
            return stddevs.mean(dim=tuple(range(len(stddevs.shape)))[1:])

    def forward_and_uncertainty_estimate(self, x: torch.Tensor) -> torch.Tensor:
        samples = self.generate_samples(x)
        stddevs = samples.std(dim=0)
        return samples.mean(dim=0), stddevs.mean(
            dim=tuple(range(len(stddevs.shape)))[1:]
        )


class DropoutMlpExtractor(MlpExtractor):

    def __init__(
        self,
        feature_dim: int,
        net_arch: List[int] | Dict[str, List[int]],
        activation_fn: nn.Module,
        device: torch.device | str = "auto",
        dropout_p: float = 0.5,
        num_samples: int = 30,
    ) -> None:
        super().__init__(feature_dim, net_arch, activation_fn, device)

        n = len(list(self.policy_net.children()))
        self.policy_net.insert(n - 2, module=torch.nn.Dropout(p=dropout_p))
        self.policy_net = MCModule(
            inner_module=self.policy_net, num_samples=num_samples
        ).to(device=device, dtype=torch.float32)


class MCActorCriticPolicy(ActorCriticPolicy):

    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
        lr_schedule: Callable[[float], float],
        net_arch: List[int] | Dict[str, List[int]] | None = None,
        activation_fn: nn.Module = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Dict[str, Any] | None = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: type[Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Dict[str, Any] | None = None,
        mlp_extractor_dropout_p: float = 0.5,
        mlp_extractor_num_samples: int = 30,
    ):
        self.mlp_extractor_dropout_p = mlp_extractor_dropout_p
        self.mlp_extractor_num_samples = mlp_extractor_num_samples
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            share_features_extractor,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = DropoutMlpExtractor(
            self.features_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
            dropout_p=self.mlp_extractor_dropout_p,
            num_samples=self.mlp_extractor_num_samples,
        )


class PolicyNetworkFromAgent(nn.Module):

    def __init__(self, agent: PPO, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.policy = agent.policy

    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        features = self.policy.extract_features(obs)
        if self.policy.share_features_extractor:
            pi_features = features
        else:
            pi_features, _vf_features = features
        latent_pi = self.policy.mlp_extractor.policy_net.inner_module(pi_features)
        distribution = self.policy._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        actions = actions.reshape((-1, *self.policy.action_space.shape))  # type: ignore[misc]
        return actions


class ExternalModelTrainer(abc.ABC):

    def __init__(
        self,
        model: torch.nn.Module | None,
        device: str,
        loss_type: Type[torch.nn.Module],
        lr: float = 0.001,
        dtype=torch.float32,
    ) -> None:
        self.device = device
        self.dtype = dtype
        self.lr = lr
        self.set_model(model)
        self.loss_type = loss_type
        self.loss_f = self.loss_type().to(device=device, dtype=dtype)

    def set_model(self, model: torch.nn.Module | None):
        if model is None:
            self.model = None
        else:
            self.model = model.to(device=self.device, dtype=self.dtype)
            self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

    def set_agent(self, agent: PPO) -> None:
        # self.agent = agent
        pass

    def process_observations(self, observations: torch.Tensor) -> torch.Tensor:
        # Converts observations from shape (batch_size, n_envs, ...) to (batch_size * n_envs, ...)
        return observations.flatten(end_dim=1)

    def rollout_to_model_input(
        self,
        env: VecEnv,
        rollout_buffer: RolloutBuffer,
        info_buffer: Dict[str, np.ndarray],
    ) -> torch.Tensor:
        return self.process_observations(
            rollout_buffer.to_torch(rollout_buffer.observations)
        )

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
    ) -> np.ndarray:
        if self.model is None:
            raise Exception("Model is not set!")

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

    def calc_loss(
        self,
        env: VecEnv,
        rollout_buffer: RolloutBuffer,
        info_buffer: Dict[str, np.ndarray],
    ):
        if self.model is None:
            raise Exception("Model is not set!")

        inp = self.rollout_to_model_input(
            env=env, rollout_buffer=rollout_buffer, info_buffer=info_buffer
        ).to(device=self.device, dtype=self.dtype)
        out = self.rollout_to_model_output(
            env=env, rollout_buffer=rollout_buffer, info_buffer=info_buffer
        ).to(device=self.device, dtype=self.dtype)

        with torch.no_grad():
            self.model.train(mode=False)
            pred_out = self.model(inp)

            loss = self.loss_f(out, pred_out)

        return loss.item()


class PolicyTrainer(ExternalModelTrainer):

    def __init__(
        self,
        device: str,
        lr: float = 0.001,
        dtype=torch.float32,
    ) -> None:
        super().__init__(None, device, torch.nn.MSELoss, lr, dtype)

    def set_agent(self, agent: PPO) -> None:
        if self.model is None:
            self.set_model(
                MCModule(
                    inner_module=PolicyNetworkFromAgent(agent=agent),
                    num_samples=agent.policy.mlp_extractor.policy_net.num_samples,
                )
            )

    def set_model(self, model: torch.nn.Module | None):
        if model is None:
            self.model = None
        else:
            self.model = model.to(device=self.device, dtype=self.dtype)

    def rollout_to_model_output(
        self,
        env: VecEnv,
        rollout_buffer: RolloutBuffer,
        info_buffer: Dict[str, np.ndarray],
    ) -> torch.Tensor:
        # Note: this is just a placeholder function that will never be used
        return torch.tensor([0], device=self.device, dtype=self.dtype)

    def receive_rollout(
        self,
        env: VecEnv,
        rollout_buffer: RolloutBuffer,
        info_buffer: Dict[str, np.ndarray],
    ) -> np.ndarray:
        # Policy is already being trained by stable baselines
        # Don't need to do it here
        return np.array(0)

    def calc_loss(
        self,
        env: VecEnv,
        rollout_buffer: RolloutBuffer,
        info_buffer: Dict[str, np.ndarray],
    ):
        # Policy is handled by stable baselines
        # Further, better eval is the mean rewards that are already logged
        return np.array(0)
