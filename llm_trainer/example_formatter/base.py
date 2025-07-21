from abc import ABC, abstractmethod
from typing import Any

from ..utils.type_utils import BatchExample


class BaseExampleFormatter(ABC):
    @abstractmethod
    def format_example(self, raw_sample: dict[str, Any]) -> BatchExample:
        """将原始样本转换为BatchExample格式"""
