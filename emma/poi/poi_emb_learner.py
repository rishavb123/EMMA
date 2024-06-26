from typing import Any, Dict, Tuple
import abc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import hydra

from emma.poi.poi_field import POIFieldModel
from emma.poi.poi_exploration import POIPPO, POISkillSamplingDiaynPPO
from emma.components.networks import PermutationInvariantNetwork, reset_weights
from emma.components.state_samplers import StateSampler


class POIEmbLearner(abc.ABC):

    def __init__(self, poi_emb_size: int) -> None:
        super().__init__()
        self.poi_emb_size = poi_emb_size
        self.device = None

    def reset(self) -> None:
        pass

    def set_poi_model(self, poi_model: POIFieldModel):
        self.poi_model = poi_model

    def set_agent(self, agent: POIPPO):
        self.agent_policy = agent.policy

    def calculate_policy_actions(self, obs: torch.Tensor):
        features = self.agent_policy.extract_features(obs)
        if self.agent_policy.share_features_extractor:
            pi_features = features
        else:
            pi_features, _vf_features = features
        latent_pi = self.agent_policy.mlp_extractor.policy_net(pi_features)
        distribution = self.agent_policy._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=True)
        actions = actions.reshape((-1, *self.agent_policy.action_space.shape))  # type: ignore[misc]
        return actions

    def set_device(self, device: str | None) -> None:
        self.device = device

    @abc.abstractmethod
    def generate_poi_emb(self, cur_obs: Any, timestep: int) -> np.ndarray:
        pass

    def train(self, inp: torch.Tensor) -> Dict[str, Any]:
        return {}


class RandomEmb(POIEmbLearner):

    def __init__(self, poi_emb_size: int) -> None:
        super().__init__(poi_emb_size)

    def generate_poi_emb(self, cur_obs: Any, timestep: int) -> np.ndarray:
        return np.random.uniform(low=0, high=1, size=(self.poi_emb_size))


class SamplingPOILearner(POIEmbLearner):

    def __init__(
        self,
        state_sampler: StateSampler,
        emb_update_model: PermutationInvariantNetwork,
        frozen_poi_pred_model: torch.nn.Module,
        poi_emb_size: int,
        num_poi_samples: int,
        num_eval_poi_samples: int | None = None,
        optimizer_cls: str = "torch.optim.Adam",
        optimizer_kwargs: Dict[str, Any] | None = None,
        loss_fn: torch.nn.Module | None = None,
        poi_learner_epochs: int = 1,
        poi_learner_obs_subset: float = 1.0,
        poi_learner_batch_size: int = 256,
        poi_emb_updates_per_generate: int = 5,
        poi_emb_num_projections: int = 5,
    ) -> None:
        super().__init__(poi_emb_size)
        self.state_sampler = state_sampler
        self.emb_update_model = emb_update_model
        self.frozen_poi_pred_model = frozen_poi_pred_model
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = optimizer_kwargs
        self.optimizer: torch.optim.Optimizer = hydra.utils.instantiate(
            {
                "_target_": self.optimizer_cls,
                **({} if self.optimizer_kwargs is None else self.optimizer_kwargs),
            },
            params=self.emb_update_model.parameters(),
        )
        self.loss_fn = torch.nn.MSELoss() if loss_fn is None else loss_fn

        self.num_poi_samples = num_poi_samples
        self.num_eval_poi_samples = (
            int(self.num_poi_samples * 0.2)
            if num_eval_poi_samples is None
            else num_eval_poi_samples
        )
        self.poi_learner_epochs = poi_learner_epochs
        self.poi_learner_batch_size = poi_learner_batch_size
        self.poi_emb_updates_per_generate = poi_emb_updates_per_generate
        self.poi_emb_num_projections = poi_emb_num_projections
        self.poi_learner_obs_subset = poi_learner_obs_subset
        self.reset_emb()

    def reset_emb(self):
        self.emb = torch.randn((self.poi_emb_size,), dtype=torch.float32)
        self.emb = self.emb / self.emb.norm()
        if self.device is not None:
            self.emb = self.emb.to(self.device)
        self.random_basis = [
            u / np.linalg.norm(u)
            for u in [
                np.random.random((self.poi_emb_size))
                for _ in range(self.poi_emb_num_projections)
            ]
        ]

    def reset(self) -> None:
        super().reset()
        self.reset_emb()
        reset_weights(self.frozen_poi_pred_model)
        reset_weights(self.emb_update_model)
        reset_weights(self.loss_fn)
        self.optimizer: torch.optim.Optimizer = hydra.utils.instantiate(
            {
                "_target_": self.optimizer_cls,
                **({} if self.optimizer_kwargs is None else self.optimizer_kwargs),
            },
            params=self.emb_update_model.parameters(),
        )
        self.state_sampler.reset()

    def set_device(self, device: str | None) -> None:
        super().set_device(device)
        self.state_sampler.set_device(device=device)
        self.emb_update_model = self.emb_update_model.to(device)
        self.frozen_poi_pred_model = self.frozen_poi_pred_model.to(device)
        self.emb = self.emb.to(self.device)

    def set_agent(self, agent: POIPPO):
        return super().set_agent(agent)

    def generate_poi_emb(self, cur_obs: Any, timestep: int) -> np.ndarray:
        if self.poi_emb_size == 0:
            return np.array([])

        if not hasattr(self, "agent_policy"):
            return np.zeros(self.poi_emb_size)

        old_emb = self.emb.detach().clone().repeat(self.num_poi_samples, 1)

        with torch.no_grad():
            for _ in range(self.poi_emb_updates_per_generate):
                samples = self.state_sampler.sample(
                    cur_obs=cur_obs, batch_size=self.num_poi_samples
                )  # (set_size, *obs_shape)

                if self.poi_model.external_model_trainer.keep_conditions:
                    inp = {"state": samples, "poi_emb": old_emb}
                elif self.poi_model.external_model_trainer.action_to_model:
                    inp = {"state": samples, "poi_emb": old_emb}
                    actions = self.calculate_policy_actions(inp)
                    actions = F.one_hot(
                        actions,
                        num_classes=self.poi_model.external_model_trainer.n_actions,
                    )
                    inp = (inp["state"], actions)
                else:
                    inp = samples

                poi_data = torch.tensor(
                    self.poi_model.calculate_poi_values(inp),
                    device=self.device,
                    dtype=torch.float32,
                ).unsqueeze(
                    1
                )  # (set_size, 1)

                emb_update_inp = torch.cat(
                    (samples, poi_data, self.emb.repeat(self.num_poi_samples, 1)), dim=1
                ).unsqueeze(
                    0
                )  # (1, set_size, 1 + obs_shape)

                self.emb: torch.Tensor = self.emb + self.emb_update_model(
                    emb_update_inp
                ).squeeze(0)
                self.emb = self.emb / self.emb.norm()

        return self.emb.detach().cpu().numpy()

    def train(self, inp: torch.Tensor | Dict[str, torch.Tensor]) -> Dict[str, Any]:
        if self.poi_emb_size == 0:
            return {}

        if not hasattr(self, "agent_policy"):
            return {}

        if isinstance(inp, dict):
            inp = inp["state"]

        m = self.state_sampler.train(inp)

        # inp: (batch_size, *obs_shape)

        if type(inp) == tuple:
            idx = torch.randperm(inp[0].shape[0])[
                : int(inp[0].shape[0] * self.poi_learner_obs_subset)
            ]
            inp = tuple(sub_inp[idx] for sub_inp in inp)
        else:
            inp = inp[
                torch.randperm(inp.shape[0])[
                    : int(inp.shape[0] * self.poi_learner_obs_subset)
                ]
            ]

        n_steps = inp[0].shape[0] if type(inp) == tuple else inp.shape[0]

        samples = []
        eval_samples = []

        for _ in range(self.poi_learner_epochs):
            for i in range(n_steps):
                if type(inp) == tuple:
                    obs = tuple(sub_inp[i] for sub_inp in inp)
                else:
                    obs = inp[i]
                samples.append(
                    self.state_sampler.sample(
                        cur_obs=obs, batch_size=self.num_poi_samples
                    )
                )
                eval_samples.append(
                    self.state_sampler.sample(
                        cur_obs=obs, batch_size=self.num_eval_poi_samples
                    )
                )
        samples = torch.stack(samples).to(
            device=self.device
        )  # (n_examples, set_size, *obs_shape)
        eval_samples = torch.stack(eval_samples).to(
            device=self.device
        )  # (n_examples, eval_set_size, *obs_shape)


        n_examples = self.poi_learner_epochs * n_steps
        indices = torch.randperm(n_examples)

        eff_batch_size = min(n_examples, self.poi_learner_batch_size)
        if eff_batch_size == -1:
            eff_batch_size = n_examples

        total_loss = 0.0

        for start_idx in range(0, n_examples, eff_batch_size):
            end_idx = min(start_idx + eff_batch_size, n_examples)
            obs = samples[
                indices[start_idx:end_idx]
            ]  # (batch_size, set_size, *obs_shape)
            eval_obs = eval_samples[
                indices[start_idx:end_idx]
            ]  # (batch_size, eval_set_size, *obs_shape)

            (batch_size, set_size, *obs_shape) = obs.shape
            (_batch_size, eval_set_size, *_obs_shape) = eval_obs.shape

            old_emb = self.emb.detach().clone().repeat(batch_size * set_size, 1)
            old_emb_eval = (
                self.emb.detach().clone().repeat(batch_size * eval_set_size, 1)
            )

            with torch.no_grad():
                if self.poi_model.external_model_trainer.keep_conditions:
                    inp = {
                        "state": obs.reshape(batch_size * set_size, *obs_shape),
                        "poi_emb": old_emb,
                    }
                elif self.poi_model.external_model_trainer.action_to_model:
                    inp = {
                        "state": obs.reshape(batch_size * set_size, *obs_shape),
                        "poi_emb": old_emb,
                    }
                    actions = self.calculate_policy_actions(inp)
                    actions = F.one_hot(
                        actions,
                        num_classes=self.poi_model.external_model_trainer.n_actions,
                    )
                    inp = (inp["state"], actions)
                else:
                    inp = obs.reshape(batch_size * set_size, *obs_shape)
                if self.poi_model.external_model_trainer.keep_conditions:
                    eval_inp = {
                        "state": eval_obs.reshape(
                            batch_size * eval_set_size, *obs_shape
                        ),
                        "poi_emb": old_emb_eval,
                    }
                elif self.poi_model.external_model_trainer.action_to_model:
                    eval_inp = {
                        "state": eval_obs.reshape(
                            batch_size * eval_set_size, *obs_shape
                        ),
                        "poi_emb": old_emb_eval,
                    }
                    actions = self.calculate_policy_actions(eval_inp)
                    actions = F.one_hot(
                        actions,
                        num_classes=self.poi_model.external_model_trainer.n_actions,
                    )
                    eval_inp = (eval_inp["state"], actions)
                else:
                    eval_inp = eval_obs.reshape(batch_size * eval_set_size, *obs_shape)

                poi_data = torch.tensor(
                    self.poi_model.calculate_poi_values(inp).reshape(
                        (batch_size, set_size, 1)
                    ),
                    device=self.device,
                    dtype=torch.float32,
                )
                eval_poi_data = torch.tensor(
                    self.poi_model.calculate_poi_values(eval_inp).reshape(
                        (batch_size, eval_set_size, 1)
                    ),
                    device=self.device,
                    dtype=torch.float32,
                )

            emb_update_inp = torch.cat(
                (obs, poi_data, self.emb.repeat((batch_size, set_size, 1))),
                dim=2,
            )  # (batch_size, num_samples, obs_shape + 1 + emb_size)

            new_embs: torch.Tensor = self.emb.repeat(
                (batch_size, 1)
            ) + self.emb_update_model(
                emb_update_inp
            )  # (batch_size, emb_size)
            new_embs = new_embs / new_embs.norm(dim=1).unsqueeze(dim=1)

            all_obs = torch.cat(
                (obs, eval_obs), dim=1
            )  # (batch_size, set_size + eval_set_size, *obs_shape)
            all_poi = torch.cat(
                (poi_data, eval_poi_data), dim=1
            )  # (batch_size, set_size + eval_set_size, 1)

            poi_pred_inp = torch.cat(
                (all_obs, new_embs.unsqueeze(1).repeat(1, set_size + eval_set_size, 1)),
                dim=2,
            )  # (batch_size, set_size + eval_set_size, obs_shape + emb_size)

            all_poi_pred: torch.Tensor = self.frozen_poi_pred_model(
                poi_pred_inp
            )  # (batch_size, set_size + eval_set_size, 1)

            self.optimizer.zero_grad()
            loss: torch.Tensor = self.loss_fn(all_poi_pred, all_poi)
            loss.backward(retain_graph=True)
            self.optimizer.step()

            total_loss += loss.item()

        m["emb_update_model_loss"] = total_loss / n_examples

        m["cur_emb/norm"] = float(self.emb.norm())
        emb_np = self.emb.detach().cpu().numpy()
        for i in range(len(self.random_basis)):
            m[f"cur_emb/projection_{i}"] = np.dot(emb_np, self.random_basis[i])

        return m


class POISkillManager(POIEmbLearner):

    def __init__(
        self,
        state_sampler: StateSampler,
        n_samples: int,
        poi_emb_size: int,
        uniform_skill_prior_warmstart: int = 0,
        uniform_skill_prior_weight: float = 0.0,
    ) -> None:
        super().__init__(poi_emb_size)
        self.state_sampler = state_sampler
        self.n_samples = n_samples
        self.uniform_skill_prior_warmstart = uniform_skill_prior_warmstart
        self.uniform_skill_prior_weight = uniform_skill_prior_weight
        self.discriminator = None

    def reset(self) -> None:
        super().reset()
        self.state_sampler.reset()

    def set_agent(self, agent: POISkillSamplingDiaynPPO):
        super().set_agent(agent=agent)
        self.discriminator = agent.discriminator

    def set_device(self, device: str | None) -> None:
        super().set_device(device)
        self.state_sampler.set_device(device=device)

    def generate_poi_emb(self, cur_obs: Any, timestep: int) -> np.ndarray:
        if timestep < self.uniform_skill_prior_warmstart or self.discriminator is None:
            skill_idx = torch.randint(low=0, high=self.poi_emb_size, size=(1,))
            sampled_skill = torch.zeros(size=(self.poi_emb_size,), dtype=torch.float32)
            sampled_skill[skill_idx] = 1.0
            return sampled_skill.cpu().numpy()
        else:
            samples = self.state_sampler.sample(
                cur_obs=cur_obs, batch_size=self.n_samples
            )  # (n_samples, *obs_shape)

            skills: torch.Tensor = self.discriminator(
                samples
            )  # (n_samples, skill_size)

            if self.poi_model.external_model_trainer.keep_conditions:
                inp = {"state": samples, "poi_emb": skills}
            elif self.poi_model.external_model_trainer.action_to_model:
                inp = {"state": samples, "poi_emb": skills}
                actions = self.calculate_policy_actions(inp)
                actions = F.one_hot(
                    actions, num_classes=self.poi_model.external_model_trainer.n_actions
                )
                inp = (inp["state"], actions)
            else:
                inp = samples

            pois = torch.tensor(
                self.poi_model.calculate_poi_values(inp),
                device=self.device,
                dtype=torch.float32,
            )  # (n_samples, )

            counts = skills.sum(dim=0)  # (skill_size, )
            counts[counts == 0] = 1.0

            average_poi_per_skill = (
                torch.sum(skills * pois.unsqueeze(dim=1), dim=0) / counts
            )  # (skill_size, )
            skill_probs = F.softmax(average_poi_per_skill, dim=0)  # (skill_size, )
            skill_probs = (
                skill_probs * (1 - self.uniform_skill_prior_weight)
                + self.uniform_skill_prior_weight / self.poi_emb_size
            )

            skill_idx = torch.multinomial(input=skill_probs, num_samples=1)
            sampled_skill = torch.zeros_like(skill_probs)
            sampled_skill[skill_idx] = 1

            return sampled_skill.cpu().numpy()

    def train(self, inp: torch.Tensor) -> Dict[str, Any]:
        return self.state_sampler.train(inp=inp)
