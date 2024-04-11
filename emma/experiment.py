from typing import Any, Dict
from dataclasses import dataclass, field
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
import logging
import pandas as pd
import wandb
import matplotlib.pyplot as plt
import copy

from stable_baselines3.common.callbacks import BaseCallback

from experiment_lab.experiments.rl import RLConfig, RLExperiment
from experiment_lab.core import run_experiment, BaseAnalysis

from emma.poi_field import POIFieldModel
from emma.poi_exploration import POIPPO
from emma.external_model import ExternalModelTrainer


logger = logging.getLogger(__name__)


class ExternalModelTrainerCallback(BaseCallback):

    def __init__(
        self,
        model_trainer: ExternalModelTrainer,
        poi_field_model: POIFieldModel,
        batch_size: int = 32,
        n_eval_steps: int = 0,
        wandb_mode: str | None = None,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.model_trainer = model_trainer
        self.poi_field_model = poi_field_model
        self.batch_size = batch_size
        self.n_eval_steps = n_eval_steps
        self.random_eval_model: POIPPO | None = None
        self.random_eval_callback: BaseCallback | None = None
        self.wandb_mode = wandb_mode

    def _on_step(self) -> bool:
        return super()._on_step()

    def _on_training_start(self) -> None:
        return super()._on_training_start()

    def _on_training_end(self) -> None:
        return super()._on_training_end()

    def _on_rollout_start(self) -> None:
        return super()._on_rollout_start()

    def _on_rollout_end(self) -> None:
        super()._on_rollout_end()
        self.model: POIPPO  # Just adding a type annotation

        env = self.model.get_env()

        self.model_trainer.set_agent(self.model)

        av_loss = self.model_trainer.receive_rollout(
            env=env,
            rollout_buffer=self.model.rollout_buffer,
            info_buffer=self.model.info_buffer,
        )

        av_poi_value = self.poi_field_model.calculate_poi_values(
            env=env,
            rollout_buffer=self.model.rollout_buffer,
            info_buffer=self.model.info_buffer,
        ).mean()

        logger.info(f"Train - Av Loss: {av_loss}; Av POI per step: {av_poi_value}")

        eval_av_loss = None
        eval_av_poi_value = None

        if (
            self.n_eval_steps > 0
        ):  # TODO: this code messes some things up for the videos, and the transfer timesteps count as it calls step in the env

            if self.random_eval_model is None:

                self.cur_idx = env.get_attr("env_idx")[0]
                self.eval_env = copy.deepcopy(env.get_attr("cur_env")[0])

                self.random_eval_model = POIPPO(
                    policy=self.model.policy_class,
                    env=self.eval_env,
                    learning_rate=0,
                    n_steps=self.n_eval_steps,
                    gamma=self.model.gamma,
                    gae_lambda=self.model.gae_lambda,
                    clip_range=self.model.clip_range,
                    clip_range_vf=self.model.clip_range_vf,
                    normalize_advantage=self.model.normalize_advantage,
                    ent_coef=self.model.ent_coef,
                    vf_coef=self.model.vf_coef,
                    max_grad_norm=self.model.max_grad_norm,
                    poi_model=self.model.poi_model,
                    infos_to_save=self.model.infos_to_save,
                    device=self.model.device,
                )
            else:
                self.random_eval_model._setup_model()

            cur_idx = env.get_attr("env_idx")[0]
            if cur_idx != self.cur_idx:
                self.cur_idx = cur_idx
                self.eval_env = copy.deepcopy(env.get_attr("cur_env")[0])
                self.random_eval_model.set_env(self.eval_env)

            self.random_eval_model._last_obs = None
            _, random_eval_callback = self.random_eval_model._setup_learn(
                self.n_eval_steps, None, True
            )

            self.random_eval_model.collect_rollouts(
                self.random_eval_model.env,
                random_eval_callback,
                self.random_eval_model.rollout_buffer,
                self.n_eval_steps,
            )

            eval_av_loss = self.model_trainer.calc_loss(
                env=env,
                rollout_buffer=self.random_eval_model.rollout_buffer,
                info_buffer=self.random_eval_model.info_buffer,
            )
            eval_av_poi_value = self.poi_field_model.calculate_poi_values(
                env=env,
                rollout_buffer=self.random_eval_model.rollout_buffer,
                info_buffer=self.random_eval_model.info_buffer,
            ).mean()

            logger.info(
                f"Eval - Av Loss: {eval_av_loss}; Av POI per step: {eval_av_poi_value}"
            )

        if self.wandb_mode != "disabled":
            wandb_logs = {
                "external_model_train/av_external_model_loss": av_loss,
                "external_model_train/av_poi_per_step": av_poi_value,
            }
            if eval_av_loss is not None:
                wandb_logs["external_model_eval/av_external_model_loss"] = eval_av_loss
            if eval_av_poi_value is not None:
                wandb_logs["external_model_eval/av_poi_per_step"] = eval_av_poi_value
            wandb.log(wandb_logs)


@dataclass
class EMMAConfig(RLConfig):

    model_trainer: Dict[str, Any] = MISSING
    external_model_batch_size: int = 64
    n_eval_steps: int = 1280

    poi_model: Dict[str, Any] = field(
        default_factory=lambda: {"_target_": "emma.poi_field.ZeroPOIField"}
    )

    def __post_init__(self) -> None:
        super().__post_init__()

        model_trainer = hydra.utils.instantiate(
            self.model_trainer,
            device=self.device,
        )
        instantiated_poi_model = hydra.utils.instantiate(
            self.poi_model, external_model_trainer=model_trainer
        )

        if self.callback_cls_lst is None:
            self.callback_cls_lst = []
        self.callback_cls_lst.insert(0, "emma.experiment.ExternalModelTrainerCallback")

        if self.callback_kwargs_lst is None:
            self.callback_kwargs_lst = []
        self.callback_kwargs_lst.insert(
            0,
            {
                "model_trainer": model_trainer,
                "poi_field_model": instantiated_poi_model,
                "batch_size": self.external_model_batch_size,
                "n_eval_steps": self.n_eval_steps,
                "wandb_mode": (
                    self.wandb["mode"]
                    if self.wandb is not None and "mode" in self.wandb
                    else (None if self.wandb is not None else "disabled")
                ),
            },
        )

        if self.model_kwargs is None:
            self.model_kwargs = {"poi_model": instantiated_poi_model}
        else:
            self.model_kwargs["poi_model"] = instantiated_poi_model


class EMMAAnalysis(BaseAnalysis):

    def __init__(self, cfg: EMMAConfig) -> None:
        self.cfg = cfg

    def analyze(self, df: pd.DataFrame, configs: Dict[str, Dict[str, Any]]) -> Any:

        results_idx = []
        results_values = {
            f"convergence_efficiency_{env_idx}": []
            for env_idx in range(1 + len(self.cfg.transfer_steps))
        }

        for experiment_id in df.index.get_level_values(0).unique():
            run_config = configs[experiment_id]
            for run_id in df.loc[experiment_id].index.get_level_values(0).unique():
                run_df = df.loc[experiment_id, run_id]

                objective = (
                    run_df["external_model_train/av_external_model_loss"]
                    .ewm(span=30)
                    .mean()
                )
                reverse_cum_min_objective = objective[::-1].cummin()[::-1]
                max_objective = objective.max()

                transfer_steps = run_config["transfer_steps"]
                idx_transfers = [
                    (run_df["global_step"] > transfer_step).idxmax()
                    for transfer_step in transfer_steps
                ]
                idx_transfers.insert(0, 0)
                idx_transfers.append(run_df.shape[0])

                converged_signal = (objective - reverse_cum_min_objective) / (
                    max_objective - reverse_cum_min_objective
                ) < 0.02

                results_idx.append((experiment_id, run_id))
                for i in range(len(idx_transfers) - 1):
                    cur_idx = idx_transfers[i]
                    next_idx = idx_transfers[i + 1]
                    first_true = converged_signal[cur_idx:next_idx].idxmax()
                    converged_signal[first_true:next_idx] = True
                    results_values[f"convergence_efficiency_{i}"].append(
                        run_df.loc[first_true, "global_step"]
                        - run_df.loc[cur_idx, "global_step"]
                    )

                # run_df["converged_signal"] = converged_signal

        q = 0.25

        def iqm(series: pd.Series):
            return series[
                (series.quantile(q) <= series) & (series <= series.quantile(1 - q))
            ].mean()

        def iqstd(series: pd.Series):
            return series[
                (series.quantile(q) <= series) & (series <= series.quantile(1 - q))
            ].std()

        aggregators = ["mean", "std", iqm, iqstd]
        results_df = (
            pd.DataFrame(
                results_values,
                index=pd.MultiIndex.from_tuples(results_idx, names=df.index.names[:2]),
            )
            .groupby("experiment_id")
            .agg({col: aggregators for col in results_values})
        )

        results_df.columns = [f"{col[0]}_{col[1]}" for col in results_df.columns]

        results_df.to_json(f"{self.output_directory}/results.json", indent=4)

        logger.info(f"\n{results_df}")

        return f"Saved Results to {self.output_directory}/results.json"


def register_configs():
    cs = ConfigStore.instance()
    cs.store(name="emma_config", node=EMMAConfig)


if __name__ == "__main__":
    run_experiment(
        experiment_cls=RLExperiment,
        config_cls=EMMAConfig,
        analysis_cls=EMMAAnalysis,
        register_configs=register_configs,
        config_name="key_prediction",
        config_path="./configs",
    )
