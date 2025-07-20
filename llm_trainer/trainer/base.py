from abc import ABC, abstractmethod
from typing import Any

from ..utils.logger import Logger


class BaseTrainer(ABC):
    def __init__(
        self,
        model_cfgs: dict[str, Any],
        data_cfgs: dict[str, Any],
        training_cfgs: dict[str, Any],
    ):
        self.model_cfgs = model_cfgs
        self.data_cfgs = data_cfgs
        self.training_cfgs = training_cfgs
        self.logger = Logger(name=f"{self.__class__.__name__.lower()}_logger")

    @abstractmethod
    def train(self) -> None:
        pass
