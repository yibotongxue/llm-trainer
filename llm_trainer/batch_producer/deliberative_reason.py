from typing import Any

from ..utils.type_utils import BatchExample, ConversationalFormatSample
from .base import BaseBatchProducer
from .pipeline.filter import get_data_filter
from .pipeline.instruction import get_instruction_generator
from .pipeline.instruction_classifier import get_instruction_classifier
from .pipeline.instruction_filter import get_instruction_filter
from .pipeline.reason import get_reason_generator


class DeliberativeReasonBatchProducer(BaseBatchProducer):
    def __init__(self, batch_cfgs: dict[str, Any]) -> None:
        super().__init__(batch_cfgs)
        self.instruction_classifier_cfgs: dict[str, Any] = batch_cfgs.get(
            "instruction_classifier_cfgs", {}
        )
        self.instruction_cfgs: dict[str, Any] = batch_cfgs.get("instruction_cfgs", {})
        self.instruction_filter_cfgs: dict[str, Any] = batch_cfgs.get(
            "instruction_filter_cfgs", {}
        )
        self.reason_cfgs: dict[str, Any] = batch_cfgs.get("reason_cfgs", {})
        self.filter_cfgs: dict[str, Any] = batch_cfgs.get("filter_cfgs", {})
        self.instruction_classifier = get_instruction_classifier(
            self.instruction_classifier_cfgs
        )
        self.instruction_generator = get_instruction_generator(self.instruction_cfgs)
        self.reason_generator = get_reason_generator(self.reason_cfgs)
        self.data_filter = get_data_filter(self.filter_cfgs)
        self.instruction_filter = get_instruction_filter(self.instruction_filter_cfgs)

    def generate_batch(
        self, example: list[BatchExample]
    ) -> list[ConversationalFormatSample]:
        example_instructions = [exp.prompt for exp in example]
        categorys = self.instruction_classifier.classify_instruction(
            example_instructions
        )
        examples_with_category = [
            exp.with_category(category) for exp, category in zip(example, categorys)
        ]
        instructions = self.instruction_generator.generate_instructions(
            examples_with_category
        )
        filtered_instructions = self.instruction_filter.filter_instructions(
            instructions
        )
        reasons = self.reason_generator.generate_reasons(filtered_instructions)
        filtered_data = self.data_filter.filter_data(reasons)

        batch_samples = [
            ConversationalFormatSample(
                messages=[
                    {"role": "user", "content": data.instruction},
                    {"role": "assistant", "content": data.response},
                ],
                meta_data=data.model_dump(),
            )
            for data in filtered_data
        ]

        return batch_samples
