from typing import Any

import gymnasium as gym
import numpy as np

from emma.poi_field import POIFieldModel
from emma.state_samplers import StateSampler


class EMMAWrapper(gym.ObservationWrapper):

    def __init__(
        self,
        env: gym.Env,
        poi_model: POIFieldModel,
        state_sampler: StateSampler | None,
        poi_emb_size: int = 64,
        per_step_poi_emb: bool = False,
    ):
        super().__init__(env)
        self.poi_model = poi_model
        self.state_sampler = state_sampler
        self.poi_emb_size = poi_emb_size
        self.use_poi_emb = self.poi_emb_size > 0
        self.per_step_poi_emb = per_step_poi_emb

        self.state_space = self.observation_space

        if self.use_poi_emb:
            self.poi_emb_space = gym.spaces.Box(0, 1, (self.poi_emb_size,), np.float32)
            self.observation_space = gym.spaces.Dict(
                {"state": self.state_space, "poi_emb": self.poi_emb_size}
            )

    def generate_poi_emb(self) -> None:
        if self.use_poi_emb:
            assert (
                self.state_sampler is not None
            ), "No state sampler specified to generate poi embedding."
            self.current_poi_emb = np.random.normal(size=(self.poi_emb_size))

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        self.generate_poi_emb()
        return super().reset(seed=seed, options=options)

    def observation(self, observation: Any) -> Any:
        if self.per_step_poi_emb:
            self.generate_poi_emb()
        if self.use_poi_emb:
            return {
                "state": observation,
                "poi_emb": self.current_poi_emb,
            }
        else:
            return observation
