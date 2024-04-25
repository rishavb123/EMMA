import abc
import logging
from typing import Any, Dict, List, Tuple
import hydra
import numpy as np
import torch
import torch.nn as nn

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.buffers import RolloutBuffer

from emma.poi.poi_field import POIFieldModel


logger = logging.getLogger(__name__)


class POIPPO(PPO, abc.ABC):

    def __init__(
        self,
        *args,
        poi_model: POIFieldModel | None = None,
        infos_to_save: Dict[str, Tuple] | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.poi_model = poi_model
        self.infos_to_save = [] if infos_to_save is None else infos_to_save
        self._reset_info_buffer()

    def _reset_info_buffer(self):
        self.info_buffer = {
            k: np.zeros(
                shape=(self.n_steps, self.n_envs, *self.infos_to_save[k]),
                dtype=np.float32,
            )
            for k in self.infos_to_save
        }

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        self._reset_info_buffer()
        return super().collect_rollouts(env, callback, rollout_buffer, n_rollout_steps)

    def _update_info_buffer(
        self, infos: List[Dict[str, Any]], dones: np.ndarray | None = None
    ) -> None:
        super()._update_info_buffer(infos, dones)
        for k in self.info_buffer:
            self.info_buffer[k][self.rollout_buffer.pos] = [
                info.get(k, 0) for info in infos
            ]


class POIAgnosticPPO(POIPPO):

    def __init__(self, *args, poi_model: POIFieldModel | None = None, **kwargs):
        super().__init__(*args, poi_model=poi_model, **kwargs)


class POIInstrinsicRewardPPO(POIPPO):

    def __init__(
        self, *args, poi_model: POIFieldModel | None = None, beta: float = 1.0, **kwargs
    ):
        super().__init__(*args, poi_model=poi_model, **kwargs)
        self.beta = beta

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        result = super().collect_rollouts(
            env=env,
            callback=callback,
            rollout_buffer=rollout_buffer,
            n_rollout_steps=n_rollout_steps,
        )
        if result:
            if self.poi_model is not None:
                model_inp = (
                    self.poi_model.external_model_trainer.rollout_to_model_input(
                        env=env,
                        rollout_buffer=rollout_buffer,
                        info_buffer=self.info_buffer,
                    )
                )
                intrinsic_rewards = self.beta * self.poi_model.calculate_poi_values(
                    model_inp=model_inp
                ).reshape(self.rollout_buffer.rewards.shape)

                logger.debug(
                    f"Extrinsic Reward Mean: {self.rollout_buffer.rewards.mean()}"
                )
                logger.debug(
                    f"Intrinsic Reward Mean: {self.beta} * {intrinsic_rewards.mean() / self.beta}"
                )

                self.rollout_buffer.rewards += intrinsic_rewards
            return True
        else:
            return False


class POISkillSamplingDiaynPPO(POIPPO):

    def __init__(
        self,
        *args,
        poi_model: POIFieldModel | None = None,
        discriminator: nn.Module | None = None,
        infos_to_save: Dict[str, Tuple] | None = None,
        beta: float = 1.0,
        discriminator_optimizer_cls: str = "torch.optim.Adam",
        discriminator_optimizer_kwargs: Dict[str, Any] | None = None,
        discriminator_batch_size: int = 128,
        **kwargs,
    ):
        super().__init__(
            *args, poi_model=poi_model, infos_to_save=infos_to_save, **kwargs
        )
        self.beta = beta

        self.cross_entropy_loss = nn.CrossEntropyLoss()

        self.discriminator_optimizer_cls = discriminator_optimizer_cls
        self.discriminator_optimizer_kwargs = discriminator_optimizer_kwargs
        self.discriminator_batch_size = discriminator_batch_size

        self.skill_size = self.observation_space["poi_emb"].shape
        assert len(self.skill_size) == 1
        self.skill_size = self.skill_size[0]

        self.set_discriminator(discriminator)

    def set_discriminator(self, discriminator: nn.Module | None):
        if discriminator is None:
            self.discriminator = None
        else:
            self.discriminator = discriminator.to(device=self.device)
            self.discriminator_optimizer: torch.optim.Optimizer = (
                hydra.utils.instantiate(
                    {
                        "_target_": self.discriminator_optimizer_cls,
                        **(
                            {}
                            if self.discriminator_optimizer_kwargs is None
                            else self.discriminator_optimizer_kwargs
                        ),
                    },
                    params=self.discriminator.parameters(),
                )
            )

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        result = super().collect_rollouts(
            env, callback, rollout_buffer, n_rollout_steps
        )
        if result:
            if self.poi_model is not None and self.discriminator is not None:
                # Calculate Instrinsic Rewards
                observations = torch.tensor(
                    rollout_buffer.observations["state"],
                    dtype=torch.float32,
                    device=self.device,
                )  # (batch_size, n_envs, *obs_shape)
                skills = torch.tensor(
                    rollout_buffer.observations["poi_emb"],
                    dtype=torch.float32,
                    device=self.device,
                )  # (batch_size, n_envs, skill_size: one hot)

                with torch.no_grad():
                    pred_skills: torch.Tensor = self.discriminator(
                        observations
                    )  # (batch_size, n_envs, skill_size: prob dist)

                intrinsic_rewards = self.beta * (
                    np.log(torch.sum(skills * pred_skills, dim=-1).cpu().numpy())
                    - np.log(
                        1 / self.skill_size
                    )  # This normalization assumes a uniform skill prior which isn't always true
                )  # (batch_size, n_envs)

                # Update Discriminator
                observations = observations.flatten(end_dim=1)
                skills = skills.flatten(end_dim=1)

                n_examples = observations.shape[0]
                indices = torch.randperm(n_examples)

                for start_idx in range(0, n_examples, self.discriminator_batch_size):
                    self.discriminator_optimizer.zero_grad()

                    end_idx = start_idx + self.discriminator_batch_size
                    batch_obs = observations[indices[start_idx:end_idx]]
                    batch_skills = skills[indices[start_idx:end_idx]]
                    batch_pred_skills = self.discriminator(batch_obs)
                    loss: torch.Tensor = self.cross_entropy_loss(
                        batch_pred_skills, batch_skills
                    )

                    loss.backward()
                    self.discriminator_optimizer.step()

                logger.debug(
                    f"Extrinsic Reward Mean: {self.rollout_buffer.rewards.mean()}"
                )
                logger.debug(
                    f"Intrinsic Reward Mean: {self.beta} * {intrinsic_rewards.mean() / self.beta}"
                )

                self.rollout_buffer.rewards += intrinsic_rewards
            return True
        else:
            return False
