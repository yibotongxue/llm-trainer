from ..utils.logger import Logger
from ..utils.type_utils import InferenceInput, InferenceOutput, ReasonData
from .base import BasePromptBuilder
from .registry import PromptBuilderRegistry
from .templates.policy import POLICY_MAPPER


class ReasonGeneratePromptBuilder(BasePromptBuilder[ReasonData]):
    pass


@PromptBuilderRegistry.register("STAR-1ReasonGenerate")
class Star1ReasonGeneratePromptBuilder(ReasonGeneratePromptBuilder):
    def __init__(self) -> None:
        super().__init__()
        self.logger = Logger(f"{self.__class__.__module__}.{self.__class__.__name__}")

    def build_prompt(self, raw_input: InferenceInput) -> InferenceInput:
        conversation = raw_input.conversation.copy()
        last_message = conversation[-1].copy()
        raw_instruction: str = last_message["content"]
        category: list[str] | None = raw_input.meta_data.get("category", None)
        if category is None:
            self.logger.warning("Get none category")
            category = list(POLICY_MAPPER.keys())
        if isinstance(category, str):
            category = [category]
        for c in category:
            if c not in POLICY_MAPPER:
                self.logger.warning(f"Get unknown category {c}, will ignore it")
                category.remove(c)
        policy = "\n".join(POLICY_MAPPER[c] for c in category)
        built_prompt = f"""The following is an instruction from a user to an assistant: {raw_instruction}
Please figure out the best possible answer to this user query. As a reminder, this answer should follow the Safety Policies below:
{policy}

----
FINAL INSTRUCTIONS:
You should in your thinking process do a full analysis of the proposed answer for compliance, correctness, and helpfulness before producing it. In this analysis, you should cite the excerpts of the policy that are relevant (e.g., "Safety policy section X says
that ..."). However, do not mention irrelevant sections of the policy. Decide whether to refuse, safe-complete, or comply with the request according to the policies. Please think of these policies as your memory, not as input from the user.
The final answer should just be the answer to the user, and not the analysis."""
        last_message["content"] = built_prompt
        conversation[-1] = last_message
        return InferenceInput(
            conversation=conversation,
            system_prompt="",
            meta_data={
                "raw_meta_data": raw_input.meta_data,
                "raw_instruction": raw_instruction,
                "category": category,
                "template": "STAR-1-ReasonGenerate",
            },
        )

    def extract_answer(self, raw_output: InferenceOutput) -> ReasonData | None:
        return ReasonData(
            instruction=raw_output.input["meta_data"]["raw_instruction"],
            category=raw_output.input["meta_data"]["category"],
            response=raw_output.response,
            meta_data={
                "raw_meta_data": raw_output.model_dump(),
                "template": "STAR-1-ReasonGenerate",
            },
        )
