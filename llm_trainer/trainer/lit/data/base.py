from typing import Any

from pytorch_lightning import LightningDataModule


class BaseCustomLightningDataModule(LightningDataModule):  # type: ignore [misc]
    def __init__(self, data_cfgs: dict[str, Any]):
        super().__init__()
        self.data_cfgs = data_cfgs
        self.init_datasets()

    def init_datasets(self) -> None:
        pass
