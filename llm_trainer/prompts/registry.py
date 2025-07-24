from typing import Any

from ..utils.registry import BaseRegistry
from .base import BasePromptBuilder


class PromptBuilderRegistry(BaseRegistry[BasePromptBuilder[Any]]):
    pass
