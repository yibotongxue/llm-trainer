import os
from typing import TypedDict

import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments

from ..data_formatter import BaseDataFormatter, DataFormatterRegistry
from ..utils.type_utils import ConversationalFormatSample
from .base import BaseTrainer


class _SftBatchSample(TypedDict):
    input_ids: torch.LongTensor
    attention_mask: torch.BoolTensor
    labels: torch.LongTensor


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


class SftTrainer(BaseTrainer):
    def init_model(self) -> None:
        model_name_or_path = self.model_cfgs["model_path"]
        model_args = self.model_cfgs.get("model_args", {})
        self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            **model_args,
        )
        self.model.gradient_checkpointing_enable()
        self.model.config.use_cache = (
            False  # Disable cache for generation during training
        )
        tokenizer_args = self.model_cfgs.get("tokenizer_args", {})
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            use_fast=True,
            **tokenizer_args,
        )

    def init_datasets(self) -> None:
        data_path = self.data_cfgs["data_path"]
        load_cfgs = self.data_cfgs.get("load_configs", {})
        raw_dataset = load_dataset(data_path, **load_cfgs)
        data_size = self.data_cfgs.get("data_size", None)
        if data_size is not None:
            raw_dataset = raw_dataset.select(range(int(data_size)))
        data_template = self.data_cfgs.get("data_template", "default")
        data_formatter = DataFormatterRegistry.get_by_name(data_template)()
        self.dataset = _SftDataset(
            raw_dataset=raw_dataset,
            data_formatter=data_formatter,
        )

    def collate_fn(self, batch: list[ConversationalFormatSample]) -> _SftBatchSample:
        query_len_list: list[int] = []
        input_len_list: list[int] = []

        for sample in batch:
            input_ids = self.tokenizer.apply_chat_template(
                conversation=sample.messages,
                add_generation_prompt=False,
                return_tensors="pt",
                return_dict=False,
            )
            query_ids = self.tokenizer.apply_chat_template(
                conversation=sample.messages[:-1],
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=False,
            )
            query_len_list.append(query_ids.size(-1))
            input_len_list.append(input_ids.size(-1))

        batched_input_ids = self.tokenizer.apply_chat_template(
            conversation=[sample.messages for sample in batch],
            add_generation_prompt=False,
            return_tensors="pt",
            return_dict=True,
            padding=True,
        )

        labels = batched_input_ids.input_ids.clone()
        for i in range(len(batch)):
            query_len = query_len_list[i]
            input_len = input_len_list[i]
            labels[i, -input_len : -input_len + query_len] = -100

        return _SftBatchSample(
            input_ids=batched_input_ids.input_ids,
            attention_mask=batched_input_ids.attention_mask,
            labels=labels,
        )

    def init_trainer(self) -> None:
        training_args = self.training_cfgs.get("training_args", {})
        project_name = self.training_cfgs.get("project_name", "sft")
        if (
            training_args.get("report_to") is not None
            and "wandb" in training_args["report_to"]
        ):
            os.environ["WANDB_PROJECT"] = project_name
        training_config = TrainingArguments(
            **training_args,
        )
        self.trainer = Trainer(
            model=self.model,
            args=training_config,
            data_collator=self.collate_fn,
            train_dataset=self.dataset,
        )
