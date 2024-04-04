import abc
import logging
from typing import Any, Dict, List, Tuple
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.buffers import RolloutBuffer

from emma.poi_field import POIFieldModel


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
        self.infos_to_save = infos_to_save
        self._reset_info_buffer()

    def _reset_info_buffer(self):
        self.info_buffer = {
            k: np.zeros(shape=(self.n_steps, self.n_envs, *self.infos_to_save[k]))
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
            self.info_buffer[k][self.rollout_buffer.pos, :] = [
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
                intrinsic_rewards = self.beta * self.poi_model.calculate_poi_values(
                    env=env,
                    rollout_buffer=rollout_buffer,
                )

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
