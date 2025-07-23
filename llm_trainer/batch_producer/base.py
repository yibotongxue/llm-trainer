from abc import ABC, abstractmethod
from typing import Any

from ..utils.type_utils import BatchExample, ConversationalFormatSample


class BaseBatchProducer(ABC):
    def __init__(self, batch_cfgs: dict[str, Any]) -> None:
        self.batch_cfgs = batch_cfgs

    @abstractmethod
    def generate_batch(
        self, example: list[BatchExample]
    ) -> list[ConversationalFormatSample]:
        pass
