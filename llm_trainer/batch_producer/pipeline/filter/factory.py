from typing import Any

from .base import BaseDataFilter


def get_data_filter(filter_cfgs: dict[str, Any]) -> BaseDataFilter:
    filter_type = filter_cfgs["filter_type"]
    if filter_type == "all_pass":
        from .all_pass import AllPassFilter

        return AllPassFilter(filter_cfgs)
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")
