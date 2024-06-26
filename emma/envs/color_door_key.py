from typing import Optional, Any, Dict, List

import logging
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.vec_env import VecEnv
import torch
import torch.nn.functional as F
import numpy as np

from minigrid.core.constants import COLOR_TO_IDX, OBJECT_TO_IDX
from minigrid.core.grid import Grid
from minigrid.core.world_object import Door, Goal, Key
from minigrid.core.mission import MissionSpace
from minigrid.minigrid_env import MiniGridEnv
from torch.nn.modules import Module

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
        obs, reward, terminated, truncated, info = super().step(action)
        info["correct_key_color_idx"] = COLOR_TO_IDX[self.correct_key_color]
        info["agent_x"] = self.agent_pos[0]
        info["agent_y"] = self.agent_pos[1]
        info["agent_dir"] = self.agent_dir
        return obs, reward, terminated, truncated, info

    def _gen_grid(self, width: int, height: int):
        # Create an empty grid
        self.grid = Grid(width=width, height=height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal in the bottom right corner
        self.put_obj(Goal(), width - 2, height - 2)

        # Create a vertical splitting wall
        splitIdx = self._rand_int(3, width - 2)
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

    def __init__(
        self,
        model: Module | None,
        device: str,
        loss_type: type[Module] = torch.nn.MSELoss,
        optimizer_cls: str = "torch.optim.Adam",
        optimizer_kwargs: Dict[str, Any] | None = None,
        batch_size: int = 128,
        epochs_per_rollout: int = 1,
        dtype=torch.float32,
    ) -> None:
        super().__init__(
            model,
            device,
            loss_type,
            optimizer_cls,
            optimizer_kwargs,
            batch_size,
            epochs_per_rollout,
            dtype,
        )

    def rollout_to_model_output(
        self,
        env: VecEnv,
        rollout_buffer: RolloutBuffer,
        info_buffer: Dict[str, np.ndarray],
    ) -> torch.Tensor:
        observations = self.rollout_to_model_input(
            env=env, rollout_buffer=rollout_buffer, info_buffer=info_buffer
        )
        correct_key_color_idx = rollout_buffer.to_torch(
            info_buffer["correct_key_color_idx"]
        ).flatten(end_dim=1)
        if len(observations.shape) == 2:
            batch_size = observations.shape[0]
            channels = (
                20 if set(torch.unique(observations).cpu().numpy()) == {0.0, 1.0} else 3
            )
            width = int((observations.shape[1] // channels) ** 0.5)
            height = width
            observations = observations.reshape(
                (batch_size, width, height, channels)
            ).moveaxis(3, 1)
        elif len(observations.shape) == 4:
            batch_size, channels, width, height = observations.shape
        else:
            raise ValueError(f"Unknown observation shape: {observations.shape}")

        if channels == 20:
            obj_elements = len(OBJECT_TO_IDX)
            colors_elements = len(COLOR_TO_IDX)

            one_hot_encoded_objs = observations[:, :obj_elements, :, :]
            one_hot_encoded_colors = observations[
                :, obj_elements : obj_elements + colors_elements, :, :
            ]
            one_hot_encoded_states = observations[
                :, obj_elements + colors_elements :, :, :
            ]

            obj_idxs = torch.argmax(one_hot_encoded_objs, dim=1)
            color_idxs = torch.argmax(one_hot_encoded_colors, dim=1)
            state_idxs = torch.argmax(one_hot_encoded_states, dim=1)

            observations = torch.stack([obj_idxs, color_idxs, state_idxs], dim=1)

        obj_idxs = observations[:, 0, :, :]
        color_idxs = observations[:, 1, :, :]

        min_dists = []

        for batch_idx in range(batch_size):
            mask = torch.logical_and(
                obj_idxs[batch_idx] == OBJECT_TO_IDX["key"],
                color_idxs[batch_idx] == correct_key_color_idx[batch_idx],
            )
            indices = mask.nonzero().to(device=self.device, dtype=self.dtype)
            agent_pos = torch.tensor(
                [width // 2, height - 1], device=self.device, dtype=self.dtype
            )

            if len(indices) > 0:
                min_dist = min(torch.linalg.norm(idx - agent_pos) for idx in indices)
            else:
                min_dist = torch.tensor(
                    width + height, device=self.device, dtype=self.dtype
                )

            min_dists.append(min_dist)

        return torch.tensor(min_dists, device=self.device, dtype=self.dtype)[:, None]


class DirectionPredictor(ExternalModelTrainer):

    def __init__(
        self,
        model: Module | None,
        device: str,
        loss_type: type[Module] = torch.nn.MSELoss,
        optimizer_cls: str = "torch.optim.Adam",
        optimizer_kwargs: Dict[str, Any] | None = None,
        batch_size: int = 128,
        epochs_per_rollout: int = 1,
        dtype=torch.float32,
    ) -> None:
        super().__init__(
            model,
            device,
            loss_type,
            optimizer_cls,
            optimizer_kwargs,
            batch_size,
            epochs_per_rollout,
            dtype,
        )

    def rollout_to_model_output(
        self,
        env: VecEnv,
        rollout_buffer: RolloutBuffer,
        info_buffer: Dict[str, np.ndarray],
    ) -> torch.Tensor:
        agent_dir = (
            rollout_buffer.to_torch(info_buffer["agent_dir"])
            .flatten(end_dim=1)
            .to(dtype=torch.long)
        )
        return F.one_hot(agent_dir, num_classes=7)
