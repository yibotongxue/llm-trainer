# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
# Copyright 2025 yibotongxue. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Modified by yibotongxue on 2025-07-27 for suit the custom need of iterative sft

# type: ignore

import warnings
from collections.abc import Callable

import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BaseImageProcessor,
    DataCollator,
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq,
    FeatureExtractionMixin,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import EvalLoopOutput
from trl.trainer.iterative_sft_config import IterativeSFTConfig


class CustomIterativeSFTTrainer(Trainer):

    _tag_names = ["custom", "iterative-sft"]

    def __init__(
        self,
        model: str | PreTrainedModel,
        args: IterativeSFTConfig | TrainingArguments | None = None,
        data_collator: DataCollator | None = None,
        eval_dataset: Dataset | dict[str, Dataset] | None = None,
        processing_class: None | (
            PreTrainedTokenizerBase
            | BaseImageProcessor
            | FeatureExtractionMixin
            | ProcessorMixin
        ) = None,
        optimizers: tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
            None,
            None,
        ),
        preprocess_logits_for_metrics: None | (
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        ) = None,
        compute_metrics: Callable[[EvalLoopOutput], dict] | None = None,
    ):
        # Args
        model_id = model if isinstance(model, str) else model.config._name_or_path
        if args is None:
            model_name = model_id.split("/")[-1]
            args = IterativeSFTConfig(f"{model_name}-IterativeSFT")
        elif isinstance(args, TrainingArguments) and not isinstance(
            args, IterativeSFTConfig
        ):
            dict_args = args.to_dict()
            dict_args["hub_token"] = args.hub_token  # to_dict hides the hub_token
            dict_args.pop("push_to_hub_token")
            args = IterativeSFTConfig(**dict_args)

        # Handle the tokenizer
        if processing_class is None:
            processing_class = AutoTokenizer.from_pretrained(model_id)

        # Model
        if args.model_init_kwargs is not None and not isinstance(model, str):
            warnings.warn(
                "You passed model_init_kwargs to the `IterativeSFTConfig`, but your model is already instantiated. "
                "The `model_init_kwargs` will be ignored."
            )
        if isinstance(model, str):
            model = self._create_model_from_path(model, args)

        self.is_peft_model = False

        self.processing_class = processing_class
        self.is_encoder_decoder = getattr(model.config, "is_encoder_decoder", False)

        if data_collator is None:
            if self.is_encoder_decoder:
                self.data_collator = DataCollatorForSeq2Seq(
                    processing_class, label_pad_token_id=-100, pad_to_multiple_of=8
                )
            else:
                self.data_collator = DataCollatorForLanguageModeling(
                    self.processing_class, mlm=False
                )
        else:
            self.data_collator = data_collator

        self.max_length = args.max_length
        self.truncation_mode = args.truncation_mode
        self.optimize_device_cache = args.optimize_device_cache

        super().__init__(
            model=model,
            args=args,
            data_collator=self.data_collator,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            compute_metrics=compute_metrics,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        if hasattr(self.model, "add_model_tags"):
            self.model.add_model_tags(self._tag_names)

        self.create_optimizer_and_scheduler(self.args.max_steps)

        # prepare model, optimizer and lr_scheduler
        self.model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.lr_scheduler
        )

        self.processing_class.truncation_side = (
            "left" if self.truncation_mode == "keep_end" else "right"
        )

        if not hasattr(self, "accelerator"):
            raise AttributeError(
                "Your `Trainer` does not have an `accelerator` object. Consider upgrading `transformers`."
            )

        PPODecorators.optimize_device_cache = self.optimize_device_cache

    def _create_model_from_path(
        self, model_path: str, args: IterativeSFTConfig
    ) -> PreTrainedModel:
        """Creates a model from a path or model identifier."""
        model_init_kwargs = args.model_init_kwargs or {}
        return AutoModelForCausalLM.from_pretrained(model_path, **model_init_kwargs)

    def prepare_model_inputs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ):
        if attention_mask is None:
            attention_mask = [torch.ones_like(ids) for ids in input_ids]

        input_data = self.data_collator(
            [
                {
                    "input_ids": ids.to(self.model.device),
                    "attention_mask": att.to(self.model.device),
                    "labels": lab.to(self.model.device),
                }
                for ids, att, lab in zip(input_ids, attention_mask, labels)
            ]
        )

        if self.max_length is not None:
            if self.truncation_mode == "keep_start":
                input_data = {k: v[: self.max_length] for k, v in input_data.items()}
            elif self.truncation_mode == "keep_end":
                input_data = {k: v[-self.max_length :] for k, v in input_data.items()}
            else:
                raise ValueError(f"Unknown truncation mode: {self.truncation_mode}")

        return input_data

    @staticmethod
    def _step_safety_checker(
        input_ids: list[torch.LongTensor],
        attention_mask: list[torch.LongTensor],
        labels: list[torch.LongTensor],
    ):
        """
        Check if the input data is valid for training.

        Args:
            input_ids (list[`torch.LongTensor`]):
                List of tensors containing the input_ids
            attention_mask (list[`torch.LongTensor`]):
                List of tensors containing the attention_mask
            labels (list[`torch.FloatTensor`]):
                List of tensors containing the labels

        Returns:
            `tuple`: The input data.
        """
        if attention_mask is None:
            for name, tensor_list in zip(["input_ids", "labels"], [input_ids, labels]):
                if not isinstance(tensor_list, list):
                    raise ValueError(
                        f"{name} must be a list of tensors - got {type(tensor_list)}"
                    )
                if not isinstance(tensor_list[0], torch.Tensor):
                    raise ValueError(
                        f"Elements in {name} must be tensors - got {type(tensor_list[0])}"
                    )
        else:
            for name, tensor_list in zip(
                ["input_ids", "attention_mask", "labels"],
                [input_ids, attention_mask, labels],
            ):
                if not isinstance(tensor_list, list):
                    raise ValueError(
                        f"{name} must be a list of tensors - got {type(tensor_list)}"
                    )
                if not isinstance(tensor_list[0], torch.Tensor):
                    raise ValueError(
                        f"Elements in {name} must be tensors - got {type(tensor_list[0])}"
                    )

        return input_ids, attention_mask, labels

    def step(
        self,
        input_ids: list[torch.LongTensor],
        attention_mask: list[torch.LongTensor],
        labels: list[torch.LongTensor] | None = None,
    ):
        self.model.train()

        if self.state.global_step == 0:
            self.tr_loss = torch.tensor(0.0).to(self.args.device)
            self._globalstep_last_logged = self.state.global_step

        # Convert Column to list if not already
        input_ids = input_ids[:] if input_ids is not None else None
        attention_mask = attention_mask[:] if attention_mask is not None else None
        labels = labels[:] if labels is not None else None

        input_ids, attention_mask, labels = self._step_safety_checker(
            input_ids, attention_mask, labels
        )
        if labels is None:
            labels = input_ids

        model_inputs = self.prepare_model_inputs(input_ids, attention_mask, labels)

        model_inputs_names = list(model_inputs.keys())

        batch_dict = {}
        batch_dict.update(model_inputs)

        def collator(data):
            return_dict = dict()
            for key in data[0]:
                if key in ["input_ids", "attention_mask", "labels"]:
                    return_dict[key] = torch.stack([d[key] for d in data]).to(
                        self.model.device
                    )
            return return_dict

        batch_data = Dataset.from_dict(batch_dict)
        batch_data.set_format("torch")

        step_dataloader = DataLoader(
            batch_data,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=True,
            collate_fn=collator,
        )

        for _, batch in enumerate(step_dataloader):
            with self.accelerator.accumulate(self.model):
                gc.collect()
                torch.cuda.empty_cache()
                gc.collect()
                model_inputs = {k: batch[k] for k in model_inputs_names}
                loss = self.compute_loss(self.model, model_inputs)

                if self.args.n_gpu > 1:
                    loss = loss.mean()

                tr_loss_step = loss.detach()

                self.accelerator.backward(loss)

                if (
                    self.accelerator.sync_gradients
                    and self.args.max_grad_norm is not None
                ):
                    self.accelerator.clip_grad_norm_(
                        self.model.parameters(),
                        self.args.max_grad_norm,
                    )

                self.optimizer.step()
                self.optimizer.zero_grad()
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

                self.state.global_step += 1

                # update stats etc
                self.tr_loss += tr_loss_step

                self._maybe_log_save_evaluate()

    def _maybe_log_save_evaluate(self):
        # check if eval is required
        if self.args.eval_steps is not None:
            if (
                self.state.global_step % self.args.eval_steps == 0
                and self.state.global_step != 0
            ):
                self.evaluate(self.eval_dataset)

        # check if logging is required
        if self.args.logging_steps is not None:
            if (
                self.state.global_step % self.args.logging_steps == 0
                and self.state.global_step != 0
            ):
                logs: dict[str, float] = {}

                tr_loss_scalar = self._nested_gather(self.tr_loss).mean().item()

                # reset tr_loss to zero
                self.tr_loss -= self.tr_loss

                logs["loss"] = round(
                    tr_loss_scalar
                    / (self.state.global_step - self._globalstep_last_logged),
                    4,
                )
                logs["learning_rate"] = self._get_learning_rate()

                self._globalstep_last_logged = self.state.global_step

                self.log(logs)
