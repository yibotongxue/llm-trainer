from typing import Any

from .base import BaseBatchProducer


def get_batch_producer(batch_cfgs: dict[str, Any]) -> BaseBatchProducer:
    batch_producer_type = batch_cfgs.get("batch_producer_type", "default")
    raise ValueError(f"Unknown batch producer type: {batch_producer_type}")
