from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from ..utils.type_utils import InferenceInput, InferenceOutput

AnswerType = TypeVar("AnswerType", bound=object, covariant=True)


class BasePromptBuilder(ABC, Generic[AnswerType]):
    @abstractmethod
    def build_prompt(self, raw_input: InferenceInput) -> InferenceInput:
        """
        Build a prompt from the raw input data.

        Args:
            raw_input (InferenceInput): The raw input data.

        Returns:
            InferenceInput: The constructed prompt.
        """

    @abstractmethod
    def extract_answer(self, raw_output: InferenceOutput) -> AnswerType | None:
        """
        Extract the answer from the raw output data.

        Args:
            raw_output (InferenceOutput): The raw output data.

        Returns:
            AnswerType | None: The extracted answer or None if not applicable.
        """
