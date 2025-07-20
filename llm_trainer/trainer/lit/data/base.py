from typing import Any

from pytorch_lightning import LightningDataModule
from transformers.tokenization_utils import PreTrainedTokenizer


class BaseCustomLightningDataModule(LightningDataModule):  # type: ignore [misc]
    def __init__(
        self, data_cfgs: dict[str, Any], tokenizer: PreTrainedTokenizer | None = None
    ):
        super().__init__()
        self.data_cfgs = data_cfgs
        if tokenizer is None:
            raise ValueError("Tokenizer must be provided")
        self.tokenizer = tokenizer
