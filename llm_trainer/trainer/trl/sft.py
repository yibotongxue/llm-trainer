import os

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from trl import SFTConfig, SFTTrainer

from ...data_formatter import DataFormatterRegistry
from .base import BaseTRLTrainer


class TRLSftTrainer(BaseTRLTrainer):
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
        self.tokenizer.chat_template = """{%- for message in messages %}
    {%- if (message.role == "user") or (message.role == "system" and not loop.first) %}
        {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}
    {%- elif (message.role == "assistant") %}
        {% generation %}    {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}    {% endgeneration %}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\\n' }}
{%- endif %}"""

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
            processing_class=self.tokenizer,
            train_dataset=self.dataset,
        )
