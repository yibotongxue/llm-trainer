from typing import Any, Literal

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
