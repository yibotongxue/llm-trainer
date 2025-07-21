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
        self.init_model()
        self.logger.print("模型初始化成功")
        self.init_datasets()
        self.logger.print("数据集初始化成功")
        self.init_trainer()
        self.logger.print("训练器初始化成功")

    @abstractmethod
    def init_model(self) -> None:
        pass

    @abstractmethod
    def init_datasets(self) -> None:
        pass

    @abstractmethod
    def init_trainer(self) -> None:
        pass

    @abstractmethod
    def train(self) -> None:
        pass
