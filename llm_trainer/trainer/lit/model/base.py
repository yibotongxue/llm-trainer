from typing import Any

from pytorch_lightning import LightningModule


class BaseCustomLightningModule(LightningModule):  # type: ignore [misc]
    def __init__(self, model_cfgs: dict[str, Any]):
        super().__init__()
        self.save_hyperparameters(model_cfgs)
        self.model_cfgs = model_cfgs
        self.init_model()

    def init_model(self) -> None:
        pass
