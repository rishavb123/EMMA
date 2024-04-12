from typing import Any

import gymnasium as gym
import numpy as np

from emma.poi.poi_field import POIFieldModel
from emma.poi.poi_emb_learner import POIEmbLearner


class EMMAWrapper(gym.ObservationWrapper):

    def __init__(
        self,
        env: gym.Env,
        poi_model: POIFieldModel,
        poi_emb_learner: POIEmbLearner,
        per_step_poi_emb: bool = False,
    ):
        super().__init__(env)
        self.poi_model = poi_model
        self.poi_emb_learner = poi_emb_learner
        self.poi_emb_learner.set_poi_model(self.poi_model)
        self.use_poi_emb = self.poi_emb_learner.poi_emb_size > 0
        self.per_step_poi_emb = per_step_poi_emb

        self.state_space = self.observation_space
        self.current_poi_emb = None

        if self.use_poi_emb:
            self.poi_emb_space = gym.spaces.Box(
                0, 1, (self.poi_emb_learner.poi_emb_size,), np.float32
            )
            self.observation_space = gym.spaces.Dict(
                {"state": self.state_space, "poi_emb": self.poi_emb_space}
            )

    def generate_poi_emb(self, cur_obs: Any) -> None:
        if self.use_poi_emb:
            self.current_poi_emb = self.poi_emb_learner.generate_poi_emb(
                cur_obs=cur_obs
            ).astype(np.float32)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)
        self.generate_poi_emb(cur_obs=obs)
        return obs, info

    def observation(self, observation: Any) -> Any:
        if self.per_step_poi_emb:
            if self.use_poi_emb:
                if self.use_poi_emb:
                    if self.current_poi_emb is None:
                        self.current_poi_emb = np.zeros(
                            self.poi_emb_learner.poi_emb_size, dtype=np.float32
                        )
                    self.generate_poi_emb(
                        cur_obs={
                            "state": observation,
                            "poi_emb": self.current_poi_emb,
                        }
                    )
                else:
                    self.generate_poi_emb(cur_obs=observation)
        if self.use_poi_emb:
            assert self.current_poi_emb is not None, "Current POI embedding is None!"
            return {
                "state": observation,
                "poi_emb": self.current_poi_emb,
            }
        else:
            return observation
