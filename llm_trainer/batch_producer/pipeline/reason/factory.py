from typing import Any

from .base import BaseReasonGenerator


def get_reason_generator(reason_cfgs: dict[str, Any]) -> BaseReasonGenerator:
    reason_generator_type = reason_cfgs["reason_generator_type"]
    if reason_generator_type == "llm":
        from .llm import LLMReasonGenerator

        return LLMReasonGenerator(reason_cfgs)
    else:
        raise ValueError(f"Unknown reason generator type: {reason_generator_type}")
