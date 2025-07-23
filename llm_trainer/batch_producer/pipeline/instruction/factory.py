from typing import Any

from .base import BaseInstructionGenerator


def get_instruction_generator(
    instruction_cfgs: dict[str, Any]
) -> BaseInstructionGenerator:
    instruction_generator_type = instruction_cfgs["instruction_generator_type"]
    if instruction_generator_type == "llm":
        from .llm import LLMInstructionGenerator

        return LLMInstructionGenerator(instruction_cfgs)
    else:
        raise ValueError(
            f"Unknown instruction generator type: {instruction_generator_type}"
        )
