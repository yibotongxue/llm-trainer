from abc import ABC, abstractmethod
from typing import Any

from ....utils.type_utils import ReasonData


class BaseDataFilter(ABC):
    def __init__(self, filter_cfgs: dict[str, Any]) -> None:
        self.filter_cfgs = filter_cfgs

    @abstractmethod
    def filter_data(self, data: list[ReasonData]) -> list[ReasonData]:
        pass
