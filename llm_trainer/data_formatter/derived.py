from typing import Any

from ..utils.type_utils import ConversationalFormatSample
from .base import BaseDataFormatter
from .registry import DataFormatterRegistry


@DataFormatterRegistry.register("default")
class DefaultDataFormatter(BaseDataFormatter):
    def format_conversation(
        self, raw_sample: dict[str, Any]
    ) -> ConversationalFormatSample:
        return ConversationalFormatSample(
            messages=[
                {"role": message["role"], "content": message["content"]}
                for message in raw_sample["messages"]
            ],
            meta_data=raw_sample.copy(),
        )


@DataFormatterRegistry.register("star1")
class Star1DataFormatter(BaseDataFormatter):
    def format_conversation(
        self, raw_sample: dict[str, Any]
    ) -> ConversationalFormatSample:
        return ConversationalFormatSample(
            messages=[
                {
                    "role": "user",
                    "content": raw_sample["question"],
                },
                {
                    "role": "assistant",
                    "content": raw_sample["response"],
                },
            ],
            meta_data=raw_sample.copy(),
        )
