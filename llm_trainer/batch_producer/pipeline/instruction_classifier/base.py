from abc import ABC, abstractmethod
from typing import Any


class BaseInstructionClassifier(ABC):
    def __init__(self, classifier_cfgs: dict[str, Any]) -> None:
        self.classifier_cfgs = classifier_cfgs

    @abstractmethod
    def classify_instruction(self, instructions: list[str]) -> list[list[str]]:
        pass
