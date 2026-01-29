# -*- coding: utf-8 -*-
"""
Paraformer 传统模型加载器
支持外部 VAD、PUNC、LM
"""

import logging
from typing import Any, Dict, Optional

from funasr import AutoModel

from .base_loader import BaseModelLoader
from ....core.exceptions import DefaultServerErrorException

logger = logging.getLogger(__name__)


class ParaformerModelLoader(BaseModelLoader):
    """Paraformer 传统模型加载器

    特性：
    - 使用外部 VAD 模型进行语音分割
    - 使用外部 PUNC 模型添加标点
    - 支持外部 N-gram 语言模型
    - 通过 TempAutoModelWrapper 动态组合模型
    """

    @property
    def model_type(self) -> str:
        return "paraformer"

    @property
    def supports_external_vad(self) -> bool:
        return True

    @property
    def supports_external_punc(self) -> bool:
        return True

    @property
    def supports_lm(self) -> bool:
        return True

    def _resolve_model_path(self, model_id: str) -> str:
        """解析模型路径，优先使用本地缓存"""
        import os
        from pathlib import Path

        # ModelScope 缓存目录
        cache_dir = os.environ.get("MODELSCOPE_CACHE", os.path.expanduser("~/.cache/modelscope"))

        possible_paths = [
            Path(cache_dir) / "hub" / model_id,
            Path(cache_dir) / "models" / model_id,
        ]

        for local_path in possible_paths:
            if local_path.exists() and local_path.is_dir():
                return str(local_path)

        return model_id

    def load(self) -> AutoModel:
        """加载 Paraformer 模型"""
        try:
            resolved_path = self._resolve_model_path(self.model_path)
            logger.info(f"正在加载 Paraformer 模型: {resolved_path}")

            model_kwargs = {
                "model": resolved_path,
                "device": self.device,
                "trust_remote_code": False,
                "disable_update": True,
                "disable_pbar": True,
                "disable_log": True,
                "local_files_only": True,
            }

            # 添加语言模型支持
            if self.enable_lm and self.lm_model:
                resolved_lm_path = self._resolve_model_path(self.lm_model)
                logger.info(f"启用语言模型: {resolved_lm_path}")
                model_kwargs["lm_model"] = resolved_lm_path
                model_kwargs["beam_size"] = self.lm_beam_size

            model = AutoModel(**model_kwargs)
            logger.info("Paraformer 模型加载成功")
            return model

        except Exception as e:
            raise DefaultServerErrorException(f"Paraformer 模型加载失败: {str(e)}")

    def prepare_generate_kwargs(
        self,
        audio_path: str,
        hotwords: str = "",
        enable_punctuation: bool = False,
        enable_itn: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """准备 Paraformer 推理参数"""
        generate_kwargs = {
            "input": audio_path,
            "cache": {},
        }

        if hotwords:
            generate_kwargs["hotword"] = hotwords

        # 如果启用了 LM，添加权重参数
        if self.enable_lm and self.lm_model:
            generate_kwargs["lm_weight"] = self.lm_weight

        return generate_kwargs
