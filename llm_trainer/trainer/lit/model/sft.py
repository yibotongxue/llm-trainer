import torch
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer

from ..data.sft import SftBatchSample
from .base import BaseCustomLightningModule


class LightningSftModule(BaseCustomLightningModule):
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

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.BoolTensor,
        labels: torch.LongTensor = None,
    ) -> CausalLMOutput:
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )

    def training_step(self, batch: SftBatchSample, batch_idx: int) -> torch.Tensor:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        assistant_mask = batch["assistant_mask"]

        labels = input_ids.clone()
        labels[~assistant_mask] = -100

        outputs = self.forward(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )

        loss = outputs.loss
        logits = outputs.logits

        preds = torch.argmax(logits, dim=-1)
        valid_positions = labels != -100
        correct = (preds == labels) & valid_positions
        accuracy = correct.sum().float() / valid_positions.sum()

        self.log("train_loss", loss, on_step=True, prog_bar=True)
        self.log("train_accuracy", accuracy, on_step=True, prog_bar=True)
        return loss

    def configure_optimizers(self) -> Optimizer:
        optimizer_cfgs = self.training_cfgs.get("optimizer", {})
        optimizer = AdamW(
            self.model.parameters(),
            **optimizer_cfgs,
        )
        lr_scheduler_cfgs = self.training_cfgs.get("lr_scheduler", {})
        if lr_scheduler_cfgs:
            lr_scheduler = CosineAnnealingLR(optimizer=optimizer, **lr_scheduler_cfgs)
            return {
                "optimizer": optimizer,
                "lr_scheduler": lr_scheduler,
            }
