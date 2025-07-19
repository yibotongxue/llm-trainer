from collections.abc import Callable
from typing import ClassVar, Generic, TypeVar

T = TypeVar("T", bound=object)


class BaseRegistry(Generic[T]):
    """通用注册表基类"""

    _registry: ClassVar[dict[str, type[T]]] = {}  # type: ignore [misc]

    @classmethod
    def register(cls, name: str) -> Callable[[type[T]], type[T]]:
        def decorator(subclass: type[T]) -> type[T]:
            cls._registry[name] = subclass
            return subclass

        return decorator

    @classmethod
    def get_by_name(cls, name: str) -> type[T]:
        if name not in cls._registry:
            raise KeyError(f"No class registered under '{name}'")
        return cls._registry[name]
