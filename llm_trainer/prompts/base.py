from abc import ABC, abstractmethod


class BasePromptBuilder(ABC):
    @abstractmethod
    def build_prompt(self, raw_prompt: str) -> str:
        """
        Build a prompt from the raw prompt string.

        Args:
            raw_prompt (str): The raw prompt string.

        Returns:
            str: The built prompt.
        """

    @abstractmethod
    def extract_answer(self, raw_output: str) -> str | None:
        """
        Extract the answer from the raw output string.

        Args:
            raw_output (str): The raw output string.

        Returns:
            str: The extracted answer.
        """
