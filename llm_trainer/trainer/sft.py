import os

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from trl import SFTConfig, SFTTrainer
from trl.trainer.utils import pad

from ..custom_dataset.sft_dataset import SftDataset
from ..data_formatter import DataFormatterRegistry
from ..utils.type_utils import TrainingDataSample
from .base import BaseTrainer


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
        tokenizer_name_or_path = self.model_cfgs.get(
            "tokenizer_path", model_name_or_path
        )
        tokenizer_args = self.model_cfgs.get("tokenizer_args", {})
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path,
            use_fast=True,
            **tokenizer_args,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def init_datasets(self) -> None:
        data_path = self.data_cfgs["data_path"]
        load_cfgs = self.data_cfgs.get("load_configs", {})
        raw_dataset = load_dataset(data_path, **load_cfgs)
        data_size = self.data_cfgs.get("data_size", None)
        if data_size is not None:
            raw_dataset = raw_dataset.select(range(int(data_size)))
        data_template = self.data_cfgs.get("data_template", "default")
        data_formatter = DataFormatterRegistry.get_by_name(data_template)()
        self.dataset = SftDataset(
            raw_dataset=raw_dataset,
            data_formatter=data_formatter,
            tokenizer=self.tokenizer,
        )

    def collate_fn(self, batch: list[TrainingDataSample]) -> TrainingDataSample:
        input_ids_list = [sample["input_ids"] for sample in batch]
        attention_mask_list = [sample["attention_mask"] for sample in batch]
        labels_list = [sample["labels"] for sample in batch]
        input_ids = pad(
            input_ids_list,
            padding_value=self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.pad_token
            ),
            padding_side="left",
        )
        attention_mask = pad(
            attention_mask_list,
            padding_value=0,
            padding_side="left",
        )
        labels = pad(
            labels_list,
            padding_value=-100,
            padding_side="left",
        )
        return TrainingDataSample(
            input_ids=input_ids,
            attention_mask=attention_mask,
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
        training_config = SFTConfig(
            **training_args,
        )
        self.trainer = SFTTrainer(
            model=self.model,
            args=training_config,
            data_collator=self.collate_fn,
            train_dataset=self.dataset,
        )

    def train(self) -> None:
        self.trainer.train()
