from abc import abstractmethod
from typing import Any

from transformers import Trainer

from ..base import BaseTrainer


class BaseTRLTrainer(BaseTrainer):
    def __init__(
        self,
        model_cfgs: dict[str, Any],
        data_cfgs: dict[str, Any],
        training_cfgs: dict[str, Any],
    ):
        super().__init__(model_cfgs, data_cfgs, training_cfgs)
        self.trainer: Trainer | None = None
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

    def train(self) -> None:
        if self.trainer is None:
            raise ValueError("The trainer isn't initialized yet")
        self.trainer.train()

    def evaluate(self) -> dict[str, float] | None:
        if self.trainer is None:
            raise ValueError("The trainer isn't initialized yet")
        if not hasattr(self, "evaluate_dataset"):
            return None
        return self.trainer.evaluate()  # type: ignore [no-any-return]
