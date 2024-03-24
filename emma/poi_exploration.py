import abc
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.buffers import RolloutBuffer

from emma.poi_field import POIFieldModel


class POIPPO(PPO, abc.ABC):

    def __init__(self, *args, poi_model: POIFieldModel | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.poi_model = poi_model


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
                    rollout_buffer
                )
                self.rollout_buffer.rewards += intrinsic_rewards
            return True
        else:
            return False
