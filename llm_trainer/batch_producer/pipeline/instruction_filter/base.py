from abc import ABC, abstractmethod
from typing import Any

from ....utils.type_utils import InstructionData


class BaseInstructionFilter(ABC):
    def __init__(self, instruction_filter_cfgs: dict[str, Any]) -> None:
        self.instruction_filter_cfgs = instruction_filter_cfgs

    @abstractmethod
    def filter_instructions(
        self, instructions: list[InstructionData]
    ) -> list[InstructionData]:
        pass
