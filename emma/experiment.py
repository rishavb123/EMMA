from typing import Any, Dict
from dataclasses import dataclass, field
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
import logging
import wandb

from stable_baselines3.common.callbacks import BaseCallback

from experiment_lab.experiments.rl import RLConfig, RLExperiment
from experiment_lab.core import run_experiment

from emma.poi_field import POIFieldModel
from emma.external_model import ExternalModelTrainer


logger = logging.getLogger(__name__)


class ExternalModelTrainerCallback(BaseCallback):

    def __init__(
        self,
        model_trainer: ExternalModelTrainer,
        poi_field_model: POIFieldModel,
        batch_size: int = 32,
        wandb_mode: str | None = None,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.model_trainer = model_trainer
        self.poi_field_model = poi_field_model
        self.batch_size = batch_size
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
        env = self.model.get_env()

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

        logger.info(f"Loss: {av_loss}; Av POI per step: {av_poi_value}")
        if self.wandb_mode != "disabled":
            wandb.log(
                {
                    "external_model_loss": av_loss,
                    "av_poi_per_step": av_poi_value,
                }
            )


@dataclass
class EMMAConfig(RLConfig):

    model_trainer: Dict[str, Any] = MISSING
    external_model_batch_size: int = 64

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
        self.callback_cls_lst.insert(
            0, "emma.external_model.ExternalModelTrainerCallback"
        )

        if self.callback_kwargs_lst is None:
            self.callback_kwargs_lst = []
        self.callback_kwargs_lst.insert(
            0,
            {
                "model_trainer": model_trainer,
                "poi_field_model": instantiated_poi_model,
                "batch_size": self.external_model_batch_size,
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


def register_configs():
    cs = ConfigStore.instance()
    cs.store(name="emma_config", node=EMMAConfig)


if __name__ == "__main__":
    run_experiment(
        experiment_cls=RLExperiment,
        config_cls=EMMAConfig,
        register_configs=register_configs,
        config_name="door_key_change",
        config_path="./configs",
    )
