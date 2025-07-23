from typing import Any

from .base import BaseBatchProducer


def get_batch_producer(batch_cfgs: dict[str, Any]) -> BaseBatchProducer:
    batch_producer_type = batch_cfgs["batch_producer_type"]
    if batch_producer_type == "deliberative_reason":
        from .deliberative_reason import DeliberativeReasonBatchProducer

        return DeliberativeReasonBatchProducer(batch_cfgs)
    else:
        raise ValueError(f"Unknown batch producer type: {batch_producer_type}")
