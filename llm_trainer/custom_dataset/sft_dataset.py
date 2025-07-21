from torch.utils.data import Dataset
from transformers.tokenization_utils import PreTrainedTokenizer

from ..data_formatter import BaseDataFormatter
from ..utils.type_utils import TrainingDataSample


class SftDataset(Dataset):  # type: ignore [misc]
    def __init__(
        self,
        raw_dataset: Dataset,
        data_formatter: BaseDataFormatter,
        tokenizer: PreTrainedTokenizer,
    ) -> None:
        self.raw_dataset = raw_dataset
        self.data_formatter = data_formatter
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.raw_dataset)

    def __getitem__(self, idx: int) -> TrainingDataSample:
        item = self.raw_dataset[idx]
        formatted_item = self.data_formatter.format_conversation(item)
        encoded = self.tokenizer.apply_chat_template(
            conversation=formatted_item.messages,
            add_generation_prompt=False,
            return_tensors="pt",
            return_dict=True,
        )
        query_ids = self.tokenizer.apply_chat_template(
            conversation=formatted_item.messages[:-1],
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=False,
        )
        labels = encoded["input_ids"].clone()
        labels[:, : query_ids.size(-1)] = -100
        return TrainingDataSample(
            input_ids=encoded["input_ids"].squeeze(0),
            attention_mask=encoded["attention_mask"].squeeze(0),
            labels=labels.squeeze(0),
        )
