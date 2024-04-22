from typing import Any, Dict
from dataclasses import dataclass, field
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, OmegaConf
import logging
import pandas as pd
import wandb
import json

from stable_baselines3.common.callbacks import BaseCallback

from experiment_lab.experiments.rl.environment import GeneralVecEnv
from experiment_lab.experiments.rl import RLConfig, RLExperiment, register_configs
from experiment_lab.core import run_experiment, BaseAnalysis

from emma.poi.poi_field import POIFieldModel
from emma.poi.poi_exploration import POIPPO
from emma.external_model import ExternalModelTrainer
from emma.poi.poi_emb_learner import POIEmbLearner


logger = logging.getLogger(__name__)


@dataclass
class EMMAConfig(RLConfig):

    model_cls: str = "emma.poi.poi_exploration.POIAgnosticPPO"

    model_trainer: Dict[str, Any] = MISSING
    n_eval_steps: int = 1280

    obs_length: int = 147

    poi_model: Dict[str, Any] = field(
        default_factory=lambda: {"_target_": "emma.poi.poi_field.ZeroPOIField"}
    )
    poi_emb_learner: Dict[str, Any] | None = field(
        default_factory=lambda: {
            "_target_": "emma.poi.poi_emb_learner.RandomEmb",
            "poi_emb_size": 64,
        }
    )
    emma_wrapper_kwargs: Dict[str, Any] = field(default_factory=lambda: {})

    def __post_init__(self) -> None:
        super().__post_init__()

        instantiated_model_trainer: ExternalModelTrainer = hydra.utils.instantiate(
            self.model_trainer,
            device=self.device,
        )
        instantiated_poi_model: POIFieldModel = hydra.utils.instantiate(
            self.poi_model, external_model_trainer=instantiated_model_trainer
        )
        instantiated_poi_emb_learner: POIEmbLearner = hydra.utils.instantiate(
            self.poi_emb_learner,
        )
        instantiated_poi_emb_learner.set_device(self.device)
        instantiated_poi_emb_learner.set_poi_model(instantiated_poi_model)

        if self.callback_cls_lst is None:
            self.callback_cls_lst = []
        self.callback_cls_lst.insert(0, "emma.experiment.EMMATrainerCallback")

        if self.callback_kwargs_lst is None:
            self.callback_kwargs_lst = []
        self.callback_kwargs_lst.insert(
            0,
            {
                "model_trainer": instantiated_model_trainer,
                "poi_field_model": instantiated_poi_model,
                "poi_emb_learner": instantiated_poi_emb_learner,
                "n_eval_steps": self.n_eval_steps,
                "wandb_mode": (
                    self.wandb["mode"]
                    if self.wandb is not None and "mode" in self.wandb
                    else (None if self.wandb is not None else "disabled")
                ),
            },
        )

        if self.wrapper_cls_lst is None:
            self.wrapper_cls_lst = []
        if self.wrapper_kwargs_lst is None:
            self.wrapper_kwargs_lst = []
        for _ in range(len(self.wrapper_kwargs_lst), len(self.wrapper_cls_lst)):
            self.wrapper_kwargs_lst.append({})
        self.wrapper_cls_lst.append("emma.poi.poi_wrapper.EMMAWrapper")
        self.wrapper_kwargs_lst.append(
            {
                "poi_model": instantiated_poi_model,
                "poi_emb_learner": instantiated_poi_emb_learner,
                **self.emma_wrapper_kwargs,
            }
        )

        if self.model_kwargs is None:
            self.model_kwargs = {"poi_model": instantiated_poi_model}
        else:
            self.model_kwargs["poi_model"] = instantiated_poi_model


class EMMATrainerCallback(BaseCallback):

    def __init__(
        self,
        model_trainer: ExternalModelTrainer,
        poi_field_model: POIFieldModel,
        poi_emb_learner: POIEmbLearner,
        batch_size: int = 32,
        n_eval_steps: int = 0,
        wandb_mode: str | None = None,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.model_trainer = model_trainer
        self.poi_field_model = poi_field_model
        self.poi_emb_learner = poi_emb_learner
        self.batch_size = batch_size
        self.n_eval_steps = n_eval_steps
        self.random_eval_model: POIPPO | None = None
        self.random_eval_callback: BaseCallback | None = None
        self.wandb_mode = wandb_mode

    def _on_step(self) -> bool:
        return super()._on_step()

    def _on_training_start(self) -> None:
        super()._on_training_start()
        self.model_trainer.reset()
        self.poi_field_model.reset()
        self.poi_emb_learner.reset()

    def _on_training_end(self) -> None:
        return super()._on_training_end()

    def _on_rollout_start(self) -> None:
        return super()._on_rollout_start()

    def _on_rollout_end(self) -> None:
        super()._on_rollout_end()
        self.model: POIPPO  # Just adding a type annotation

        env: GeneralVecEnv = self.model.get_env()

        self.model_trainer.set_agent(self.model)

        rollout_buffer = self.model.rollout_buffer
        info_buffer = self.model.info_buffer

        inp = self.model_trainer.rollout_to_model_input(
            env=env, rollout_buffer=rollout_buffer, info_buffer=info_buffer
        )
        out = self.model_trainer.rollout_to_model_output(
            env=env, rollout_buffer=rollout_buffer, info_buffer=info_buffer
        ).to(device=self.model_trainer.device, dtype=self.model_trainer.dtype)

        av_loss = self.model_trainer.receive_rollout(
            inp=inp,
            out=out,
        )

        additional_metrics = self.poi_emb_learner.train(inp=inp)

        av_poi_value = self.poi_field_model.calculate_poi_values(model_inp=inp).mean()

        logger.info(f"Train - Av Loss: {av_loss}; Av POI per step: {av_poi_value}")
        logger.info(json.dumps(additional_metrics))

        eval_av_loss = None
        eval_av_poi_value = None

        if self.n_eval_steps > 0:
            if self.random_eval_model is None:
                self.random_eval_model = POIPPO(
                    policy=self.model.policy_class,
                    env=env.eval_env,
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

            eval_inp = self.model_trainer.rollout_to_model_input(
                env=env,
                rollout_buffer=self.random_eval_model.rollout_buffer,
                info_buffer=info_buffer,
            ).to(device=self.model_trainer.device, dtype=self.model_trainer.dtype)
            eval_out = self.model_trainer.rollout_to_model_output(
                env=env,
                rollout_buffer=self.random_eval_model.rollout_buffer,
                info_buffer=self.random_eval_model.info_buffer,
            ).to(device=self.model_trainer.device, dtype=self.model_trainer.dtype)

            eval_av_loss = self.model_trainer.calc_loss(
                inp=eval_inp,
                out=eval_out,
            )
            eval_av_poi_value = self.poi_field_model.calculate_poi_values(
                model_inp=eval_inp,
            ).mean()

            logger.info(
                f"Eval - Av Loss: {eval_av_loss}; Av POI per step: {eval_av_poi_value}"
            )

        if self.wandb_mode != "disabled":
            wandb_logs = {
                "external_model_train/av_external_model_loss": av_loss,
                "external_model_train/av_poi_per_step": av_poi_value,
                **additional_metrics,
            }
            if eval_av_loss is not None:
                wandb_logs["external_model_eval/av_external_model_loss"] = eval_av_loss
            if eval_av_poi_value is not None:
                wandb_logs["external_model_eval/av_poi_per_step"] = eval_av_poi_value
            wandb.log(wandb_logs)


class EMMAAnalysis(BaseAnalysis):

    def __init__(self, cfg: EMMAConfig) -> None:
        self.cfg = cfg

    def analyze(self, df: pd.DataFrame, configs: Dict[str, Dict[str, Any]]) -> Any:

        results_idx = []
        results_values = {
            f"convergence_efficiency_{env_idx}": []
            for env_idx in range(1 + len(self.cfg.transfer_steps))
        } | {
            f"asymptotic_performance_{env_idx}": []
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
                    results_values[f"asymptotic_performance_{i}"].append(
                        objective[cur_idx:next_idx].min()
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
