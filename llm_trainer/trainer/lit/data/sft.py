from typing import TypedDict

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset

from ....data_formatter import BaseDataFormatter, DataFormatterRegistry
from ....utils.type_utils import ConversationalFormatSample
from .base import BaseCustomLightningDataModule


class _SftDataset(Dataset):  # type: ignore [misc]
    def __init__(self, raw_dataset: Dataset, data_formatter: BaseDataFormatter) -> None:
        self.raw_dataset = raw_dataset
        self.data_formatter = data_formatter

    def __len__(self) -> int:
        return len(self.raw_dataset)

    def __getitem__(self, idx: int) -> ConversationalFormatSample:
        item = self.raw_dataset[idx]
        formatted_item = self.data_formatter.format_conversation(item)
        return formatted_item


class SftBatchSample(TypedDict):
    input_ids: torch.LongTensor
    attention_mask: torch.BoolTensor
    assistant_mask: torch.BoolTensor


class LightningSftDataModule(BaseCustomLightningDataModule):
    def setup(self, stage: str) -> None:
        data_path = self.data_cfgs["data_path"]
        load_cfgs = self.data_cfgs.get("load_configs", {})
        raw_dataset = load_dataset(data_path, **load_cfgs)
        data_size = self.data_cfgs.get("data_size", None)
        if data_size is not None:
            raw_dataset = raw_dataset.select(range(int(data_size)))
        data_template = self.data_cfgs.get("data_template", "default")
        data_formatter = DataFormatterRegistry.get_by_name(data_template)()
        self.dataset = _SftDataset(raw_dataset, data_formatter)

    def collate_fn(self, batch: list[ConversationalFormatSample]) -> SftBatchSample:
        texts = []

        for sample in batch:
            messages = sample.messages
            # 使用 tokenizer 的 chat 模板拼接文本
            prompt_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,  # 不添加空 assistant 开头
            )
            texts.append(prompt_text)

        # 编码整个 batch
        tokenized = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.data_cfgs.get("max_length", 2048),
            return_tensors="pt",
        )

        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]

        # assistant_mask 构造
        # 注意：这里使用特殊 token 区分哪些 token 属于 assistant 的回复
        assistant_mask = torch.zeros_like(input_ids, dtype=torch.bool)

        for i, sample in enumerate(batch):
            message_list = sample.messages
            # 找出 assistant 回复段落在消息序列中的 index
            assistant_content = ""
            for msg in message_list:
                if msg["role"] == "assistant":
                    assistant_content += msg["content"] + "\n"

            if assistant_content.strip() == "":
                continue  # 没有 assistant 回复就跳过

            # Encode only assistant content
            assistant_encoded = self.tokenizer(
                assistant_content,
                truncation=True,
                max_length=input_ids.size(1),
                add_special_tokens=False,
            )

            # 在 input_ids 中搜索该段编码是否出现
            target_seq = torch.tensor(assistant_encoded["input_ids"], dtype=torch.long)
            seq_len = target_seq.size(0)
            for j in range(input_ids.size(1) - seq_len + 1):
                if torch.equal(input_ids[i, j : j + seq_len], target_seq):
                    assistant_mask[i, j : j + seq_len] = 1
                    break  # Assume first match is correct

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "assistant_mask": assistant_mask,
        }

    def train_dataloader(self) -> DataLoader:
        data_loader_cfgs = self.data_cfgs.get("train_dataloader", {})
        return DataLoader(
            dataset=self.dataset, collate_fn=self.collate_fn, **data_loader_cfgs
        )
