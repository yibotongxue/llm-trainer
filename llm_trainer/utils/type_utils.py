from typing import Any, Literal, TypedDict

import torch
from pydantic import BaseModel, Field


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
