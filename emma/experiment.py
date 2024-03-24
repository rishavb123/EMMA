from dataclasses import dataclass

from experiment_lab.experiments.rl.config import RLConfig


@dataclass
class EMMAConfig(RLConfig):

    poi_

    def __post_init__(self) -> None:
        super().__post_init__()