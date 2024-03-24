import abc
from stable_baselines3.common.buffers import RolloutBuffer

from emma.external_model import ExternalModelTrainer


class POIFieldModel(abc.ABC):

    def __init__(self, external_model_trainer: ExternalModelTrainer) -> None:
        super().__init__()
        self.external_model_trainer = external_model_trainer

    @abc.abstractmethod
    def calculate_poi_values(self, rollout_buffer: RolloutBuffer) -> float:
        return 0.0

class ZeroPOIField(POIFieldModel):

    def __init__(self, external_model_trainer: ExternalModelTrainer) -> None:
        super().__init__(external_model_trainer)

    def calculate_poi_values(self, rollout_buffer: RolloutBuffer) -> float:
        return 0.0
