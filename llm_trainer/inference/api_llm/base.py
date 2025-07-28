from abc import abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from tqdm import tqdm

from ...utils.type_utils import InferenceInput, InferenceOutput
from ..base import BaseInference


class BaseApiLLMInference(BaseInference):
    """
    API调用型大语言模型推理基类

    为通过API调用大语言模型提供基础功能，包括重试、并行处理等

    参数
    ----
    model_cfgs : dict[str, Any]
        模型配置参数
    inference_cfgs : dict[str, Any]
        推理配置参数，可包含max_retry, max_workers, sleep_seconds等
    """

    def __init__(
        self, model_cfgs: dict[str, Any], inference_cfgs: dict[str, Any]
    ) -> None:
        super().__init__(model_cfgs=model_cfgs, inference_cfgs=inference_cfgs)
        self.max_retry: int = self.inference_cfgs.pop("max_retry", 3)
        self.max_workers: int = self.inference_cfgs.pop("max_workers", 32)
        self.sleep_seconds: int = self.inference_cfgs.pop("sleep_seconds", 30)

    def _generate(
        self,
        inputs: list[InferenceInput],
        enable_tqdm: bool = False,
        tqdm_args: dict[str, Any] | None = None,
    ) -> list[InferenceOutput]:
        if len(inputs) == 0:
            return []
        if len(inputs) == 1:
            return [self._single_generate(inputs[0])]
        return self._parallel_generate(inputs, enable_tqdm, tqdm_args)

    def _parallel_generate(
        self,
        inputs: list[InferenceInput],
        enable_tqdm: bool = False,
        tqdm_args: dict[str, Any] | None = None,
    ) -> list[InferenceOutput]:
        """
        并行执行多个推理请求

        参数
        ----
        inputs : list[InferenceInput]
            输入数据列表
        enable_tqdm : bool, 默认为False
            是否显示进度条
        tqdm_args : dict[str, Any] | None, 默认为None
            进度条的参数配置

        返回
        ----
        list[InferenceOutput]
            推理结果列表
        """
        results: dict[int, InferenceOutput] = {}
        max_workers = min(len(inputs), self.max_workers)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(self._single_generate, input_item): idx
                for idx, input_item in enumerate(inputs)
            }

            futures = as_completed(future_to_index)
            if enable_tqdm:
                tqdm_args = tqdm_args or {"desc": "Generating responses"}
                futures = tqdm(futures, total=len(inputs), **tqdm_args)

            for future in futures:
                idx = future_to_index[future]
                result = future.result()
                results[idx] = result

        return [results[i] for i in range(len(inputs))]

    @abstractmethod
    def _single_generate(self, inference_input: InferenceInput) -> InferenceOutput:
        """
        处理单个推理请求

        参数
        ----
        inference_input : InferenceInput
            单个输入数据

        返回
        ----
        InferenceOutput
            推理结果

        注意
        ----
        此方法必须由子类实现
        """

    def _get_inference_essential_cfgs(self) -> dict[str, Any]:
        """
        获取推理配置中必要的部分

        返回
        ----
        dict[str, Any]
            包含必要推理配置的字典
        """
        essential_inference_cfgs = self.inference_cfgs.copy()
        essential_inference_cfgs.pop("max_retry", None)
        essential_inference_cfgs.pop("max_workers", None)
        essential_inference_cfgs.pop("sleep_seconds", None)
        essential_cfgs = {
            "model_cfgs": self.model_cfgs,
            "inference_cfgs": essential_inference_cfgs,
        }
        return essential_cfgs
