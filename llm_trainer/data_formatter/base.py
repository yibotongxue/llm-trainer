from abc import ABC, abstractmethod
from typing import Any

from ..utils.type_utils import ConversationalFormatSample


class BaseDataFormatter(ABC):
    @abstractmethod
    def format_conversation(
        self, raw_sample: dict[str, Any]
    ) -> ConversationalFormatSample:
        """
        Convert a raw sample into a ConversationalFormatSample.

        Args:
            raw_sample (dict[str, Any]): The raw sample to format.

        Returns:
            ConversationalFormatSample: The formatted conversational sample.
        """
