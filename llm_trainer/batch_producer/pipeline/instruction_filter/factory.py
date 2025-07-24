from typing import Any

from .base import BaseInstructionFilter


def get_instruction_filter(
    instruction_filter_cfgs: dict[str, Any]
) -> BaseInstructionFilter:
    instruction_filter_type = instruction_filter_cfgs["instruction_filter_type"]
    if instruction_filter_type == "all_pass":
        from .all_pass import AllPassInstructionFilter

        return AllPassInstructionFilter(instruction_filter_cfgs)
    else:
        raise ValueError(f"Unknown instruction filter type: {instruction_filter_type}")
