from __future__ import annotations

from copy import deepcopy
from typing import Any, Literal, TypedDict

import torch
from pydantic import BaseModel, ConfigDict, Field


class CustomBaseModel(BaseModel):  # type: ignore [misc]
    model_config = ConfigDict(extra="allow")

    def to_brief_dict(self) -> dict[str, Any]:
        raw_dict = deepcopy(self.model_dump())
        if "meta_data" in raw_dict:
            raw_dict.pop("meta_data")
        return raw_dict  # type: ignore [no-any-return]


class InferenceInput(CustomBaseModel):
    conversation: list[dict[str, Any]]
    system_prompt: str
    meta_data: dict[str, Any]

    model_config = ConfigDict(extra="allow")

    @classmethod
    def from_prompts(
        cls: type[InferenceInput], prompt: str, system_prompt: str = ""
    ) -> InferenceInput:
        return cls(
            conversation=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            system_prompt=system_prompt,
            meta_data={},
        )

    def get_raw_question(self) -> str:
        if "raw_question" in self.meta_data:
            return self.meta_data["raw_question"]  # type: ignore [no-any-return]
        return self.conversation[-1]["content"]  # type: ignore [no-any-return]

    def with_meta_data(self, meta_data: dict[str, Any]) -> InferenceInput:
        new_meta_data = {
            **self.meta_data,
            **meta_data,
        }
        raw = {
            **self.model_dump(),
            "meta_data": new_meta_data,
        }
        return InferenceInput(**raw)


class InferenceOutput(CustomBaseModel):
    response: str
    extracted_answer: str | None = None
    input: dict[str, Any]
    engine: str
    meta_data: dict[str, Any]

    model_config = ConfigDict(extra="allow")


class ConversationMessage(BaseModel):  # type: ignore [misc]
    content: str = Field(..., description="The content of the message")
    role: Literal["system", "user", "assistant"] = Field(
        ..., description="The role of the message sender, e.g., 'user' or 'assistant'"
    )


class ConversationalFormatSample(BaseModel):  # type: ignore [misc]
    messages: list[ConversationMessage] = Field(
        ..., description="A list of messages in the conversation"
    )
    meta_data: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata related to the conversation",
    )


class InstructionFormatSample(BaseModel):  # type: ignore [misc]
    prompt: str = Field(..., description="The instruction or question to be answered")
    completion: str = Field(
        ..., description="The expected response to the instruction or question"
    )
    meta_data: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata related to the instruction",
    )


class BatchExample(BaseModel):  # type: ignore [misc]
    prompt: str = Field(..., description="The input prompt for the model")
    expected_completion: str | None = Field(
        None, description="The expected output from the model"
    )
    failure_completion: str | None = Field(
        None, description="The actual output from the model"
    )
    meta_data: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata related to the example"
    )


class TrainingDataSample(TypedDict):
    input_ids: torch.LongTensor
    attention_mask: torch.LongTensor
    labels: torch.LongTensor
