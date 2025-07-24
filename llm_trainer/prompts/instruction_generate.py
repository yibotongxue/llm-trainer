from ..utils.extract_tools import extract_json_dict
from ..utils.logger import Logger
from ..utils.type_utils import InferenceInput, InferenceOutput, InstructionData
from .base import BasePromptBuilder
from .registry import PromptBuilderRegistry
from .templates.policy import CATEGORY_LIST, POLICY_MAPPER


class InstructionGeneratePromptBuilder(BasePromptBuilder[list[InstructionData]]):
    pass


@PromptBuilderRegistry.register("SimpleInstructionGenerate")
class SimpleInstructionGeneratePromptBuilder(InstructionGeneratePromptBuilder):
    def __init__(self) -> None:
        super().__init__()
        self.logger = Logger(f"{self.__class__.__module__}.{self.__class__.__name__}")

    def build_prompt(self, raw_input: InferenceInput) -> InferenceInput:
        conversation = raw_input.conversation.copy()
        last_message = conversation[-1].copy()
        raw_example: str = last_message["content"]
        category: list[str] | None = raw_input.meta_data.get("category", None)
        if category is None:
            self.logger.warning("Get none category")
            category = list(POLICY_MAPPER.keys())
        for c in category:
            if c not in POLICY_MAPPER:
                self.logger.warning(f"Get unknown category {c}, will ignore it")
                category.remove(c)
        policy = "\n".join(POLICY_MAPPER[c] for c in category)
        built_prompt = f"""You are an exceptionally intelligent AI assistant who has a thorough grasp of safety policies.
Now you will need to help researcher to generate some safety alignment training data. The researchers will offer you an example harmful instruction along with the safety policy it probably breaches.
First you should examine all the provided safety category and policy to filter that the example instruction doesn't breach.
Next, for each safety policy that the example instruction breaches, you need to generate five diverse training prompts that violate the same policy in different ways. Make sure the new prompts are as varied as possible while still breaching the same safety policy.
Finally you should format all the generated training data in a JSON list, each JSON object contains the "instruction" and "category" field. The "instruction" field refer to the generated prompts and the "category" field refers to the category of the safety policy the prompt breach.
Note that the value of "category" field should be one of the element of the {CATEGORY_LIST}
An example final answer format is
```json
[
    {{
        "instruction": <THE HARMFUL INSTRUCTION GENERATED>,
        "category": <CATEGORY>,
    }},
    {{
        "instruction": <THE HARMFUL INSTRUCTION GENERATED>,
        "category": <CATEGORY>,
    }},
    ...
]
```
The example prompt is

{raw_example}

The related safety policy is/are:

{policy}

Please make sure that the last ```json ``` code block is your final answer and do not output any other code block after that.
""".strip()
        last_message["content"] = built_prompt
        conversation[-1] = last_message
        return InferenceInput(
            conversation=conversation,
            system_prompt="",
            meta_data={
                "raw_meta_data": raw_input.meta_data,
                "raw_example": raw_example,
                "template": "STAR-1-InstructionGenerate",
            },
        )

    def extract_answer(
        self, raw_output: InferenceOutput
    ) -> list[InstructionData] | None:
        json_dict = extract_json_dict(raw_output.response)
        if json_dict is None:
            self.logger.error("Fail to extract json dict from response")
            return None
        if isinstance(json_dict, dict):
            self.logger.error(
                f"Get dict from the output, expected list instead. The output is {raw_output.response}"
            )
            return None
        result: list[InstructionData] = []
        for i, item in enumerate(json_dict):
            try:
                instruction_data = InstructionData(
                    **item, meta_data={"template": "SimpleInstructionGenerate"}
                )
            except Exception as e:
                self.logger.warning(
                    f"Fail to parse {i}: {item} to InstructionData, the exception is {e}"
                )
            result.append(instruction_data)
        return result
