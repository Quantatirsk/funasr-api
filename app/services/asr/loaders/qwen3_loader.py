# -*- coding: utf-8 -*-
"""
Qwen3-ASR 模型加载器
支持 vLLM 后端和离线加载
"""

import logging
import os
from typing import Any, Dict, Optional

from .base_loader import BaseModelLoader
from ....core.exceptions import DefaultServerErrorException
from ....infrastructure.model_utils import resolve_model_path, resolve_forced_aligner_path

logger = logging.getLogger(__name__)

# 延迟导入 qwen_asr，避免启动时加载
try:
    from qwen_asr import Qwen3ASRModel
except ImportError:
    Qwen3ASRModel = None
    logger.warning("qwen-asr 未安装，Qwen3-ASR 功能不可用")


class Qwen3ModelLoader(BaseModelLoader):
    """Qwen3-ASR 模型加载器

    特性：
    - 使用 vLLM 后端进行高效推理
    - 内置 VAD、标点、语言识别
    - 支持字级时间戳（通过 Qwen3-ForcedAligner）
    - 强制离线模式，支持无互联网环境
    """

    @property
    def model_type(self) -> str:
        return "qwen3"

    @property
    def supports_external_vad(self) -> bool:
        return False  # Qwen3 内置 VAD

    @property
    def supports_external_punc(self) -> bool:
        return False  # Qwen3 内置标点

    @property
    def supports_lm(self) -> bool:
        return False  # Qwen3 不使用外部 LM

    def load(self) -> Any:
        """加载 Qwen3-ASR 模型"""
        if Qwen3ASRModel is None:
            raise ImportError(
                "qwen-asr 未安装，请运行: pip install qwen-asr[vllm]"
            )

        try:
            # 解析模型路径（支持环境变量、HF缓存、ModelScope缓存）
            resolved_path = resolve_model_path(
                self.model_path,
                env_var="QWEN_ASR_MODEL_PATH"
            )
            logger.info(f"正在加载 Qwen3-ASR 模型: {resolved_path}")

            # 解析 forced_aligner 路径
            forced_aligner_path = self.extra_kwargs.get("forced_aligner_path")
            if forced_aligner_path:
                resolved_aligner = resolve_forced_aligner_path(forced_aligner_path)
                self.extra_kwargs["forced_aligner_path"] = resolved_aligner
                logger.info(f"使用 ForcedAligner: {resolved_aligner}")

            # 提取 vLLM 参数
            gpu_memory_utilization = self.extra_kwargs.get("gpu_memory_utilization", 0.4)
            max_model_len = self.extra_kwargs.get("max_model_len")
            max_inference_batch_size = self.extra_kwargs.get("max_inference_batch_size", 32)
            max_new_tokens = self.extra_kwargs.get("max_new_tokens", 1024)

            # 构建 vLLM 初始化参数
            llm_kwargs = {
                "model": resolved_path,
                "gpu_memory_utilization": gpu_memory_utilization,
                "forced_aligner": self.extra_kwargs.get("forced_aligner_path"),
                "max_inference_batch_size": max_inference_batch_size,
                "max_new_tokens": max_new_tokens,
            }

            if max_model_len:
                llm_kwargs["max_model_len"] = max_model_len

            # 使用 vLLM 后端初始化
            model = Qwen3ASRModel.LLM(**llm_kwargs)
            logger.info(f"Qwen3-ASR 模型加载成功: {resolved_path}")

            return model

        except Exception as e:
            raise DefaultServerErrorException(f"Qwen3-ASR 模型加载失败: {str(e)}")

    def prepare_generate_kwargs(
        self,
        audio_path: Optional[str],
        hotwords: str = "",
        enable_punctuation: bool = False,
        enable_itn: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """准备 Qwen3-ASR 推理参数"""
        generate_kwargs = {
            "audio": audio_path,
            "context": hotwords if hotwords else "",
        }

        # Qwen3 内置标点和 ITN，无需额外参数

        return generate_kwargs
