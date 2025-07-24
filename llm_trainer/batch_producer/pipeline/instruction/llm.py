from typing import Any

from ....inference import InferenceFactory
from ....prompts import PromptBuilderRegistry
from ....prompts.instruction_generate import InstructionGeneratePromptBuilder
from ....utils.logger import Logger
from ....utils.type_utils import BatchExample, InferenceInput, InstructionData
from .base import BaseInstructionGenerator


class LLMInstructionGenerator(BaseInstructionGenerator):
    def __init__(self, instruction_cfgs: dict[str, Any]):
        super().__init__(instruction_cfgs)
        model_cfgs: dict[str, Any] = instruction_cfgs["model_cfgs"]
        inference_cfgs: dict[str, Any] = instruction_cfgs["inference_cfgs"]
        cache_cfgs: dict[str, Any] = instruction_cfgs.get("cache_cfgs", None)
        self.inference = InferenceFactory.get_inference_instance(
            model_cfgs=model_cfgs,
            inference_cfgs=inference_cfgs,
            cache_cfgs=cache_cfgs,
        )
        self.logger = Logger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        self.prompt_builder_type = (
            instruction_cfgs["prompt_builder_type"] + "InstructionGenerate"
        )
        self.prompt_builder = PromptBuilderRegistry.get_by_name(
            self.prompt_builder_type
        )()
        if not isinstance(self.prompt_builder, InstructionGeneratePromptBuilder):
            raise TypeError(
                f"Expected InstructionGeneratePromptBuilder, got {type(self.prompt_builder).__name__}"
            )

    def generate_instructions(
        self, example: list[BatchExample]
    ) -> list[InstructionData]:
        inputs = [
            InferenceInput.from_prompts(
                prompt=exp.prompt,
                system_prompt="",
            ).with_meta_data(
                {
                    "category": exp.category,
                }
            )
            for exp in example
        ]
        outputs = self.inference.generate(
            inputs=inputs,
            prompt_template=self.prompt_builder_type,
            enable_tqdm=True,
            tqdm_args={"desc": "Generating instructions"},
        )
        instructions: list[InstructionData] = []
        for i, output in enumerate(outputs):
            extracted_answer = output.extracted_answer
            if extracted_answer is None:
                continue
            if not isinstance(extracted_answer, list):
                self.logger.warning(
                    f"Expected list, got {type(extracted_answer).__name__} for example {i}: {extracted_answer}"
                )
                continue
            for i, item in enumerate(extracted_answer):
                if not isinstance(item, InstructionData):
                    self.logger.warning(
                        f"Expected InstructionData, got {type(item).__name__} for example {i}: {item}"
                    )
                    continue
                instructions.append(item)
        return instructions
