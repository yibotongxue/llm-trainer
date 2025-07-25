import os
from typing import Any

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from trl.trainer import IterativeSFTConfig, IterativeSFTTrainer
from trl.trainer.utils import pad

from ..batch_producer import get_batch_producer
from ..custom_dataset.example_dataset import ExampleDataset
from ..custom_dataset.iterative_sft_dataset import IterativeSftDataset
from ..example_formatter import ExampleFormatterRegistry
from ..utils.type_utils import TrainingDataSample
from .base import BaseTrainer


class IterativeSftTrainer(BaseTrainer):
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
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def init_datasets(self) -> None:
        data_path = self.data_cfgs["data_path"]
        load_cfgs = self.data_cfgs.get("load_configs", {})
        raw_dataset = load_dataset(data_path, **load_cfgs)
        data_size = self.data_cfgs.get("data_size", None)
        shuffle_seed = self.data_cfgs.get("shuffle_seed", 42)
        raw_dataset = raw_dataset.shuffle(seed=shuffle_seed)
        if data_size is not None:
            raw_dataset = raw_dataset.select(range(int(data_size)))
        template = self.data_cfgs.get("data_template", "default")
        example_formatter = ExampleFormatterRegistry.get_by_name(template)()
        self.example_dataset = ExampleDataset(
            raw_dataset=raw_dataset, example_formatter=example_formatter
        )
        batch_cfgs: dict[str, Any] = self.data_cfgs.get("batch_configs", {})
        self.batch_producer = get_batch_producer(batch_cfgs=batch_cfgs)
        self.example_batch_size = batch_cfgs.get("example_batch_size", 1)

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
        self.save_steps = self.training_cfgs.get("save_steps", 10)
        self.training_config = IterativeSFTConfig(**training_args)
        self.trainer = IterativeSFTTrainer(
            model=self.model,
            args=self.training_config,
            data_collator=self.collate_fn,
        )

    def train(self) -> None:
        batches = [
            self.example_dataset[i : i + self.example_batch_size]
            for i in range(0, len(self.example_dataset), self.example_batch_size)
        ]
        for i, batch in enumerate(batches):
            training_data_samples = self.batch_producer.generate_batch(batch)
            training_dataset = IterativeSftDataset(
                sample_list=training_data_samples,
                tokenizer=self.tokenizer,
            )

            self.trainer.step(
                input_ids=[sample["input_ids"] for sample in training_dataset],
                attention_mask=[
                    sample["attention_mask"] for sample in training_dataset
                ],
                labels=[sample["labels"] for sample in training_dataset],
            )
            if (i + 1) % self.save_steps == 0:
                self.save_model(
                    step=f"step-{i}",
                    tag=f"step-{i}",
                )
        self.save_model(
            step="end",
            tag="end",
        )

    def save_model(self, step: str, tag: str) -> None:
        if not self.training_config.output_dir:
            raise ValueError("Output directory is not set in training arguments.")
        if not os.path.exists(self.training_config.output_dir):
            os.makedirs(self.training_config.output_dir)

        save_dir = f"{self.training_config.output_dir}/{step}"

        self.trainer.model.save_checkpoint(
            save_dir=save_dir,
            tag=tag,
            client_state={},
        )
        self.tokenizer.save_pretrained(save_dir)
        self.model.config.save_pretrained(save_dir)
