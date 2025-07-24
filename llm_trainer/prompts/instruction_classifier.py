from ast import literal_eval

from ..utils.extract_tools import extract_last_tag_content
from ..utils.logger import Logger
from ..utils.type_utils import InferenceInput, InferenceOutput
from .base import BasePromptBuilder
from .registry import PromptBuilderRegistry
from .templates.policy import CATEGORY_DEFINITION


class InstructionClassifierPromptBuilder(BasePromptBuilder[list[str]]):
    pass


@PromptBuilderRegistry.register("STAR-1InstructionClassifier")
class Star1InstructionClassifierPromptBuilder(InstructionClassifierPromptBuilder):
    def __init__(self) -> None:
        super().__init__()
        self.logger = Logger(f"{self.__class__.__module__}.{self.__class__.__name__}")

    def build_prompt(self, raw_input: InferenceInput) -> InferenceInput:
        conversation = raw_input.conversation.copy()
        last_message = conversation[-1].copy()
        raw_instruction: str = last_message["content"]
        built_prompt = f"""The following is an instruction from a user to an assistant: {raw_instruction}
Please determine whether this instruction applies to the following safety categories:
{CATEGORY_DEFINITION}

----
FINAL INSTRUCTIONS:
If it applies to one or more categories, please output the category name, if it does not apply to any category, classify it as "Other" and output ["Other"].
The category names should be within the following list:
["Harassment / Hate / Discrimination", "Sexual / Adult", "Violence / Physical Harm", "Self-Harm", "Illicit / Criminal Behavior", "Misinformation / Disinformation", "Privacy / Personal Data", "Intellectual Property", "Other"]

You should thoughtfully consider the instruction and verify each category detailedly. Finally, you should output a list in the last line of the response, containing the category names within the <answer></answer> tag.
For example, if the instruction is related to "Harassment / Hate / Discrimination" and "Violence / Physical Harm", you should output:
<answer>["Harassment / Hate / Discrimination", "Violence / Physical Harm"]
If the instruction does not apply to any category, you should output:
<answer>["Other"]</answer>
Please ensure that the answer within answer tag can be parsed by the Python code `literal_eval` and is a list of strings.
""".strip()
        last_message["content"] = built_prompt
        conversation[-1] = last_message
        return InferenceInput(
            conversation=conversation,
            system_prompt="",
            meta_data={
                "raw_meta_data": raw_input.meta_data,
                "raw_instruction": raw_instruction,
                "template": "STAR-1-InstructionClassifier",
            },
        )

    def extract_answer(self, raw_output: InferenceOutput) -> list[str] | None:
        answer = extract_last_tag_content(raw_output.response, "answer")
        if answer is None:
            return None
        try:
            answer_list = literal_eval(answer)
            if not isinstance(answer_list, list):
                self.logger.error(
                    f"Expected list, got {type(answer_list).__name__} for output: {answer}"
                )
                return None

            verified_answer = []
            for i, item in enumerate(answer_list):
                if not isinstance(item, str):
                    self.logger.warning(
                        f"Expected str, got {type(item).__name__} for item {i}: {item}"
                    )
                    continue
                verified_answer.append(item)
            return verified_answer
        except Exception as e:
            self.logger.error(f"Failed to parse answer: {e}")
            return None
