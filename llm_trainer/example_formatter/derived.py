from typing import Any

from ..utils.type_utils import BatchExample
from .base import BaseExampleFormatter
from .registry import ExampleFormatterRegistry


@ExampleFormatterRegistry.register("default")
class DefaultExampleFormatter(BaseExampleFormatter):
    def format_example(self, raw_sample: dict[str, Any]) -> BatchExample:
        """
        将原始样本转换为BatchExample格式。
        假设raw_sample包含'prompt'和'expected_completion'字段。
        """
        return BatchExample(
            prompt=raw_sample["prompt"],
            expected_completion=raw_sample.get("expected_completion", None),
            failure_completion=raw_sample.get("failure_completion", None),
            meta_data=raw_sample.get("meta_data", {}),
        )


@ExampleFormatterRegistry.register("STAR-1")
class Star1ExampleFormatter(BaseExampleFormatter):
    def format_example(self, raw_sample: dict[str, Any]) -> BatchExample:
        return BatchExample(
            prompt=raw_sample["question"],
            expected_completion=raw_sample["response"],
            failure_completion=None,
            meta_data=raw_sample.copy(),
        )
