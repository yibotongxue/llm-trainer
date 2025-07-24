from typing import Any

from .base import BaseInstructionClassifier


def get_instruction_classifier(
    classifier_cfgs: dict[str, Any]
) -> BaseInstructionClassifier:
    classifier_type = classifier_cfgs["classifier_type"]
    if classifier_type == "llm":
        from .llm import LLMInstructionClassifier

        return LLMInstructionClassifier(classifier_cfgs)
    else:
        raise ValueError(f"Unknown instruction classifier type: {classifier_type}")
