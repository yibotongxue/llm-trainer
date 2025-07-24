from typing import Any

from .base import BaseInstructionFilter


class AllPassInstructionFilter(BaseInstructionFilter):
    def filter_instructions(self, instructions: list[Any]) -> list[Any]:
        return instructions
