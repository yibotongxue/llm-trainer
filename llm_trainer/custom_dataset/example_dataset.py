from torch.utils.data import Dataset

from ..example_formatter import BaseExampleFormatter
from ..utils.type_utils import BatchExample


class ExampleDataset(Dataset):  # type: ignore [misc]
    def __init__(
        self, raw_dataset: Dataset, example_formatter: BaseExampleFormatter
    ) -> None:
        self.raw_dataset = raw_dataset
        self.example_formatter = example_formatter

    def __len__(self) -> int:
        return len(self.raw_dataset)

    def __getitem__(self, idx: int | slice) -> BatchExample | list[BatchExample]:
        if isinstance(idx, slice):
            return [self.__getitem__(i) for i in range(*idx.indices(len(self)))]  # type: ignore [misc]
        raw_sample = self.raw_dataset[idx]
        return self.example_formatter.format_example(raw_sample)
