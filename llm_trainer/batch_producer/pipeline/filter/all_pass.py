from ....utils.type_utils import ReasonData
from .base import BaseDataFilter


class AllPassFilter(BaseDataFilter):
    def filter_data(self, data: list[ReasonData]) -> list[ReasonData]:
        return data
