import os

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from trl import SFTConfig, SFTTrainer

from ..data_formatter import DataFormatterRegistry
from .base import BaseTrainer


class SftTrainer(BaseTrainer):
    def init_model(self) -> None:
        model_name_or_path = self.model_cfgs["model_path"]
        model_args = self.model_cfgs.get("model_args", {})
        self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            **model_args,
        )
        tokenizer_args = self.model_cfgs.get("tokenizer_args", {})
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            model_name_or_path,
            use_fast=True,
            **tokenizer_args,
        )

    def init_datasets(self) -> None:
        data_path = self.data_cfgs["data_path"]
        load_cfgs = self.data_cfgs.get("load_configs", {})
        self.dataset = load_dataset(data_path, **load_cfgs)
        data_size = self.data_cfgs.get("data_size", None)
        if data_size is not None:
            self.dataset = self.dataset.select(range(int(data_size)))
        data_template = self.data_cfgs.get("data_template", "default")
        data_formatter = DataFormatterRegistry.get_by_name(data_template)()
        self.dataset = self.dataset.map(
            lambda x: data_formatter.format_conversation(x).model_dump(),
            remove_columns=self.dataset.column_names,
            desc="Formatting dataset",
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
            train_dataset=self.dataset,
        )


def main() -> None:
    import argparse

    from ..utils.config import (
        deepcopy_config,
        load_config,
        update_config_with_unparsed_args,
    )

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--config-file-path",
        type=str,
        required=True,
        help="The path to the config file",
    )
    args, unparsed_args = parser.parse_known_args()

    cfgs = load_config(args.config_file_path)
    update_config_with_unparsed_args(unparsed_args=unparsed_args, cfgs=cfgs)

    cfgs = deepcopy_config(cfgs)

    trainer = SftTrainer(**cfgs)

    trainer.train()
