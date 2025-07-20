from typing import Any

from lightning.pytorch.strategies import DeepSpeedStrategy
from pytorch_lightning import Trainer
from transformers.tokenization_utils import PreTrainedTokenizer

from ..base import BaseTrainer
from .data import *
from .model import *


class BaseLightningTrainer(BaseTrainer):
    LightningModuleClass: type[BaseCustomLightningModule] = BaseCustomLightningModule
    DataModuleClass: type[BaseCustomLightningDataModule] = BaseCustomLightningDataModule

    def __init__(
        self,
        model_cfgs: dict[str, Any],
        data_cfgs: dict[str, Any],
        training_cfgs: dict[str, Any],
    ):
        super().__init__(model_cfgs, data_cfgs, training_cfgs)
        self.tokenizer: PreTrainedTokenizer | None = None
        self.init_model()
        self.logger.print("模型初始化成功")
        self.init_datasets()
        self.logger.print("数据集初始化成功")
        self.init_trainer()
        self.logger.print("训练器初始化成功")

    def init_model(self) -> None:
        self.module = self.LightningModuleClass(self.model_cfgs)

    def init_datasets(self) -> None:
        self.data_module = self.DataModuleClass(self.data_cfgs, self.tokenizer)

    def init_trainer(self) -> None:
        deepspeed_config = self.training_cfgs["deepspeed_config"]
        self.trainer = Trainer(
            strategy=DeepSpeedStrategy(config=deepspeed_config),
            **self.training_cfgs.get("trainer_args", {}),
        )

    def train(self) -> None:
        if self.trainer is None:
            raise ValueError("The trainer isn't initialized yet")
        self.trainer.fit(self.module, datamodule=self.data_module)


class LightningSftTrainer(BaseLightningTrainer):
    LightningModuleClass: type[BaseCustomLightningModule] = LightningSftModule
    DataModuleClass: type[BaseCustomLightningDataModule] = LightningSftDataModule
