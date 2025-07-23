import json
from typing import Any

import redis  # type: ignore [import-untyped]

from .base import BaseCacheManager
from .registry import CacheManagerRegistry


@CacheManagerRegistry.register("redis")
class RedisCacheManager(BaseCacheManager):
    def __init__(self, cache_cfgs: dict[str, Any]) -> None:
        super().__init__(cache_cfgs=cache_cfgs)
        self.redis_client = redis.Redis(**cache_cfgs)

    def _load_cache(self, key: str) -> dict[str, Any] | None:
        value = self.redis_client.get(key)
        if value is not None:
            # TODO 需要更仔细地检查加载的内容
            return json.loads(value)  # type: ignore [no-any-return]
        return None

    def save_cache(self, key: str, value: dict[str, Any]) -> None:
        self.redis_client.set(key, json.dumps(value))
