from typing import Any

from ....inference import InferenceFactory
from ....prompts import PromptBuilderRegistry
from ....prompts.instruction_classifier import InstructionClassifierPromptBuilder
from ....utils.logger import Logger
from ....utils.type_utils import InferenceInput
from .base import BaseInstructionClassifier


class LLMInstructionClassifier(BaseInstructionClassifier):
    def __init__(self, classifier_cfgs: dict[str, Any]):
        super().__init__(classifier_cfgs)
        model_cfgs: dict[str, Any] = classifier_cfgs["model_cfgs"]
        inference_cfgs: dict[str, Any] = classifier_cfgs["inference_cfgs"]
        cache_cfgs: dict[str, Any] = classifier_cfgs.get("cache_cfgs", None)
        self.inference = InferenceFactory.get_inference_instance(
            model_cfgs=model_cfgs,
            inference_cfgs=inference_cfgs,
            cache_cfgs=cache_cfgs,
        )
        self.logger = Logger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        self.prompt_builder_type = classifier_cfgs["prompt_builder_type"]
        prompt_builder = PromptBuilderRegistry.get_by_name(self.prompt_builder_type)()
        if not isinstance(prompt_builder, InstructionClassifierPromptBuilder):
            raise TypeError(
                f"Expected InstructionClassifierPromptBuilder, got {type(prompt_builder).__name__}"
            )

    def classify_instruction(self, instructions: list[str]) -> list[list[str]]:
        inputs = [
            InferenceInput.from_prompts(
                prompt=instruction,
                system_prompt="",
            )
            for instruction in instructions
        ]
        outputs = self.inference.generate(
            inputs=inputs,
            prompt_template=self.prompt_builder_type,
            enable_tqdm=True,
            tqdm_args={"desc": "Classifying instructions"},
        )
        flatten_outputs = [output[0] for output in outputs]
        results: list[list[str]] = []
        for i, output in enumerate(flatten_outputs):
            if output.extracted_answer is None:
                self.logger.warning(
                    f"The output {i}: {output} get None extracted answer."
                )
                continue
            if not isinstance(output.extracted_answer, list):
                self.logger.warning(
                    f"Expected list, got {type(output.extracted_answer).__name__} for example {i}: {output.extracted_answer}"
                )
                continue
            verified_extracted_answer: list[str] = []
            for i, item in enumerate(output.extracted_answer):
                if not isinstance(item, str):
                    self.logger.warning(
                        f"Expected str, got {type(item).__name__} for example {i}: {item}"
                    )
                    continue
                verified_extracted_answer.append(item)
            results.append(verified_extracted_answer)
        return results
