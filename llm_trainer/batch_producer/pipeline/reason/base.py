from abc import ABC, abstractmethod
from typing import Any

from ....utils.type_utils import InstructionData, ReasonData


class BaseReasonGenerator(ABC):
    def __init__(self, reason_cfgs: dict[str, Any]) -> None:
        self.reason_cfgs = reason_cfgs

    @abstractmethod
    def generate_reasons(self, instructions: list[InstructionData]) -> list[ReasonData]:
        pass
