from ast import literal_eval
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
        self.prompt_builder_type = instruction_cfgs["prompt_builder_type"]
        prompt_builder = PromptBuilderRegistry.get_by_name(self.prompt_builder_type)()
        if not isinstance(prompt_builder, InstructionGeneratePromptBuilder):
            raise TypeError(
                f"Expected InstructionGeneratePromptBuilder, got {type(prompt_builder).__name__}"
            )

    def generate_instructions(
        self, example: list[BatchExample]
    ) -> list[InstructionData]:
        inputs = [
            InferenceInput.from_prompts(
                prompt=exp.prompt,
                system_prompt="",
            )
            for exp in example
        ]
        outputs = self.inference.generate(
            inputs=inputs,
            prompt_template=self.prompt_builder_type,
            enable_tqdm=True,
            tqdm_args={"desc": "Generating instructions"},
        )
        flatten_outputs = [output[0] for output in outputs]
        instructions: list[InstructionData] = []
        for i, output in enumerate(flatten_outputs):
            extracted_answer = output.extracted_answer
            if extracted_answer is None:
                continue
            evaled_answer = literal_eval(extracted_answer)
            if isinstance(evaled_answer, list):
                for item in evaled_answer:
                    instruction = self._parse_dict(item)
                    if instruction is not None:
                        instructions.append(instruction)
            elif isinstance(evaled_answer, dict):
                instruction = self._parse_dict(evaled_answer)
                if instruction is not None:
                    instructions.append(instruction)
                    continue
                for item in evaled_answer.values():
                    instruction = self._parse_dict(item)
                    if instruction is not None:
                        instructions.append(instruction)
            else:
                self.logger.warning(
                    f"Expected list or dict, got {type(evaled_answer).__name__} for example {i}: {extracted_answer}"
                )
        return instructions

    def _parse_dict(self, data: Any) -> InstructionData | None:
        """
        解析字典数据为InstructionData对象

        参数
        ----
        data : Any
            输入数据，可以是字典或其他类型

        返回
        ----
        InstructionData | None
            如果解析成功，返回InstructionData对象，否则返回None
        """
        if not isinstance(data, dict):
            self.logger.warning(f"Expected {data} as dict, got {type(data).__name__}")
            return None
        if "instruction" not in data:
            self.logger.warning(f"Missing 'instruction' key in data: {data}")
            return None
        try:
            return InstructionData(
                instruction=data["instruction"],
                meta_data=data,
            )
        except Exception as e:
            self.logger.warning(f"Error parsing data to InstructionData: {e}")
            return None
