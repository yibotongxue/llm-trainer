from abc import ABC, abstractmethod
from typing import Any

from ....utils.type_utils import BatchExample, InstructionData


class BaseInstructionGenerator(ABC):
    def __init__(self, instruction_cfgs: dict[str, Any]) -> None:
        self.instruction_cfgs = instruction_cfgs

    @abstractmethod
    def generate_instructions(
        self, example: list[BatchExample]
    ) -> list[InstructionData]:
        pass
