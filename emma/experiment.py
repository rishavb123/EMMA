from typing import Any, Dict
from dataclasses import dataclass, field
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from experiment_lab.experiments.rl import RLConfig, RLExperiment
from experiment_lab.core import run_experiment


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
            self.callback_cls_lst = ["emma.external_model.ExternalModelTrainerCallback"]
        else:
            self.callback_cls_lst.insert(
                0, "emma.external_model.ExternalModelTrainerCallback"
            )

        if self.callback_kwargs_lst is None:
            self.callback_kwargs_lst = [
                {
                    "model_trainer": model_trainer,
                    "batch_size": self.external_model_batch_size,
                }
            ]
        else:
            self.callback_kwargs_lst.insert(
                0,
                {
                    "model_trainer": model_trainer,
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
