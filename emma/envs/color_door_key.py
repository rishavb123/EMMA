from typing import Optional, Any, Dict, List, SupportsFloat

import logging
import gymnasium as gym
from stable_baselines3.common.type_aliases import RolloutBufferSamples
from stable_baselines3.common.vec_env import VecEnv
import torch
from minigrid.core.constants import COLOR_TO_IDX, OBJECT_TO_IDX
from minigrid.core.grid import Grid
from minigrid.core.world_object import Door, Goal, Key
from minigrid.core.mission import MissionSpace
from minigrid.minigrid_env import MiniGridEnv

from emma.external_model import ExternalModelTrainer


logger = logging.getLogger(__name__)


class ColorDoor(Door):
    """
    A Door instance where the key color can be specified and doesn't have to match the door
    """

    def __init__(self, color, is_open=False, is_locked=False, key_color=None):
        super().__init__(color, is_open, is_locked)
        self.is_open = is_open
        self.is_locked = is_locked
        if key_color:
            self.key_color = key_color
        else:
            self.key_color = color

    def toggle(self, env, pos):
        # If the player has the right key to open the door
        if self.is_locked:
            if isinstance(env.carrying, Key) and env.carrying.color == self.key_color:
                self.is_locked = False
                self.is_open = True
                return True
            return False

        self.is_open = not self.is_open
        return True


class ColoredDoorKeyEnv(MiniGridEnv):

    def __init__(
        self,
        door_color: str = "yellow",
        key_colors: Optional[List[str]] = None,
        correct_key_color: str = "yellow",
        size: int = 8,
        max_steps: Optional[int] = None,
        **kwargs: Dict[str, Any],
    ):
        self.door_color = door_color
        self.key_colors = key_colors if key_colors is not None else [correct_key_color]
        self.correct_key_color = correct_key_color
        if max_steps is None:
            max_steps = 10 * size**2
        mission_space = MissionSpace(mission_func=self._gen_mission)
        super().__init__(
            mission_space=mission_space, grid_size=size, max_steps=max_steps, **kwargs
        )

    @staticmethod
    def _gen_mission():
        return "use the correct key to open the door and get to the goal"

    def step(self, action):
        return super().step(action)

    def _gen_grid(self, width: int, height: int):
        # Create an empty grid
        self.grid = Grid(width=width, height=height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal in the bottom right corner
        self.put_obj(Goal(), width - 2, height - 2)

        # Create a vertical splitting wall
        splitIdx = self._rand_int(2, width - 2)
        self.grid.vert_wall(splitIdx, 0)

        # Place the agent at a random position and orientation on the left side
        self.place_agent(size=(splitIdx, height))

        # Place a door in the wall
        doorIdx = self._rand_int(1, width - 2)
        self.put_obj(
            ColorDoor(
                self.door_color, is_locked=True, key_color=self.correct_key_color
            ),
            splitIdx,
            doorIdx,
        )

        # Place a yellow key on the left side
        for color in self.key_colors:
            self.place_obj(obj=Key(color=color), top=(0, 0), size=(splitIdx, height))

        self.mission = self._gen_mission()


class CorrectKeyDistancePredictor(ExternalModelTrainer):

    def __init__(self, model: torch.nn.Module, device: str, lr: float = 0.001) -> None:
        super().__init__(model=model, device=device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.loss_f = torch.nn.MSELoss()

    def obs_to_model_inp(self, obs: Any) -> Any:
        return obs

    def receive_rollout(
        self, env: VecEnv, rollout_buffer_samples: RolloutBufferSamples
    ):
        correct_key_color = env.get_attr("unwrapped")[0].correct_key_color
        batch_size, channels, width, height = rollout_buffer_samples.observations.shape
        obj_idxs = rollout_buffer_samples.observations[:, 0, :, :]
        colors = rollout_buffer_samples.observations[:, 1, :, :]

        device = rollout_buffer_samples.observations.device

        min_dists = []

        for batch_idx in range(batch_size):
            mask = torch.logical_and(
                obj_idxs[batch_idx] == OBJECT_TO_IDX["key"],
                colors[batch_idx] == COLOR_TO_IDX[correct_key_color],
            )
            indices = mask.nonzero().to(dtype=torch.float32)
            agent_pos = torch.tensor(
                [width // 2, height - 1], dtype=torch.float32, device=device
            )

            if len(indices) > 0:
                min_dist = min(torch.linalg.norm(idx - agent_pos) for idx in indices)
            else:
                min_dist = torch.tensor(width + height, device=device)

            min_dists.append(min_dist)

        inp = rollout_buffer_samples.observations.flatten(start_dim=1)
        out = torch.tensor(min_dists, device=device)[:, None]

        self.optimizer.zero_grad()

        pred_out = self.model(inp)
        loss = self.loss_f(out, pred_out)
        loss.backward()

        self.optimizer.step()

        return loss.item()
