from typing import Any

from ....inference import InferenceFactory
from ....prompts import PromptBuilderRegistry
from ....prompts.reason_generate import ReasonGeneratePromptBuilder
from ....utils.logger import Logger
from ....utils.type_utils import InferenceInput, InstructionData, ReasonData
from .base import BaseReasonGenerator


class LLMReasonGenerator(BaseReasonGenerator):
    def __init__(self, reason_cfgs: dict[str, Any]):
        super().__init__(reason_cfgs)
        model_cfgs: dict[str, Any] = reason_cfgs["model_cfgs"]
        inference_cfgs: dict[str, Any] = reason_cfgs["inference_cfgs"]
        cache_cfgs: dict[str, Any] = reason_cfgs.get("cache_cfgs", None)
        self.inference = InferenceFactory.get_inference_instance(
            model_cfgs=model_cfgs,
            inference_cfgs=inference_cfgs,
            cache_cfgs=cache_cfgs,
        )
        self.logger = Logger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        self.prompt_builder_type = reason_cfgs["prompt_builder_type"] + "ReasonGenerate"
        prompt_builder = PromptBuilderRegistry.get_by_name(self.prompt_builder_type)()
        if not isinstance(prompt_builder, ReasonGeneratePromptBuilder):
            raise TypeError(
                f"Expected ReasonGeneratePromptBuilder, got {type(prompt_builder).__name__}"
            )

    def generate_reasons(self, instructions: list[InstructionData]) -> list[ReasonData]:
        inputs = [
            InferenceInput.from_prompts(
                prompt=instruction.instruction,
                system_prompt="",
            ).with_meta_data({"category": instruction.category})
            for instruction in instructions
        ]
        outputs = self.inference.generate(
            inputs=inputs,
            prompt_template=self.prompt_builder_type,
            enable_tqdm=True,
            tqdm_args={"desc": "Generating reasons"},
        )
        for i, output in enumerate(outputs):
            if output.extracted_answer is None:
                self.logger.warning(
                    f"The output {i}: {output} get None extracted answer."
                )
        results: list[ReasonData] = []
        for i, output in enumerate(outputs):
            if output.extracted_answer is None:
                self.logger.warning(
                    f"Skipping example {i} due to None extracted answer: {output}"
                )
                continue
            if not isinstance(output.extracted_answer, ReasonData):
                self.logger.warning(
                    f"Expected ReasonData, got {type(output.extracted_answer).__name__} for example {i}: {output.extracted_answer}"
                )
                continue
            results.append(output.extracted_answer)
        return results
