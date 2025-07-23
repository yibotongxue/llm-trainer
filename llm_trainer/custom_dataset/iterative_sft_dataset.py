from torch.utils.data import Dataset
from transformers.tokenization_utils import PreTrainedTokenizer

from ..utils.type_utils import ConversationalFormatSample, TrainingDataSample


class IterativeSftDataset(Dataset):  # type: ignore [misc]
    def __init__(
        self,
        sample_list: list[ConversationalFormatSample],
        tokenizer: PreTrainedTokenizer,
    ):
        self.sample_list = sample_list
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.sample_list)

    def __getitem__(self, idx: int) -> TrainingDataSample:
        item = self.sample_list[idx]
        encoded = self.tokenizer.apply_chat_template(
            conversation=item.messages,
            add_generation_prompt=False,
            return_tensors="pt",
            return_dict=True,
        )
        query_ids = self.tokenizer.apply_chat_template(
            conversation=item.messages[:-1],
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
