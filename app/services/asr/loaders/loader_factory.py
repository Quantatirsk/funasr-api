# -*- coding: utf-8 -*-
"""
模型加载器工厂
根据配置创建对应的模型加载器
"""

import logging
from typing import Dict, Optional, Type

from .base_loader import BaseModelLoader
from .paraformer_loader import ParaformerModelLoader
from .funasrnano_loader import FunASRNanoModelLoader

logger = logging.getLogger(__name__)


class ModelLoaderFactory:
    """模型加载器工厂类

    根据模型配置自动选择并创建合适的加载器
    """

    # 模型类型到加载器的映射
    _LOADER_REGISTRY: Dict[str, Type[BaseModelLoader]] = {
        "paraformer": ParaformerModelLoader,
        "fun-asr-nano": FunASRNanoModelLoader,
    }

    @classmethod
    def create_loader(
        cls,
        model_path: str,
        device: str,
        extra_kwargs: Optional[Dict] = None,
        enable_lm: bool = False,
        lm_model: Optional[str] = None,
        lm_weight: float = 0.15,
        lm_beam_size: int = 10,
    ) -> BaseModelLoader:
        """创建模型加载器

        根据 extra_kwargs 中的特征自动识别模型类型并创建对应的加载器

        Args:
            model_path: 模型路径或模型ID
            device: 推理设备
            extra_kwargs: 额外参数（用于识别模型类型）
            enable_lm: 是否启用语言模型
            lm_model: 语言模型路径
            lm_weight: 语言模型权重
            lm_beam_size: beam size

        Returns:
            对应的模型加载器实例
        """
        extra_kwargs = extra_kwargs or {}

        # 识别模型类型
        model_type = cls._detect_model_type(model_path, extra_kwargs)
        loader_class = cls._LOADER_REGISTRY.get(model_type, ParaformerModelLoader)

        logger.info(f"检测到模型类型: {model_type}, 使用加载器: {loader_class.__name__}")

        # 创建加载器实例
        return loader_class(
            model_path=model_path,
            device=device,
            enable_lm=enable_lm,
            lm_model=lm_model,
            lm_weight=lm_weight,
            lm_beam_size=lm_beam_size,
            **extra_kwargs
        )

    @classmethod
    def _detect_model_type(cls, model_path: str, extra_kwargs: Dict) -> str:
        """检测模型类型

        根据模型路径和额外参数识别模型类型
        """
        # 检查是否是 Fun-ASR-Nano
        if "fun-asr-nano" in model_path.lower() or "Fun-ASR-Nano" in model_path:
            return "fun-asr-nano"

        # 检查是否使用了 trust_remote_code（通常是端到端模型）
        if extra_kwargs.get("trust_remote_code"):
            # 可以进一步扩展识别其他远程代码模型
            return "fun-asr-nano"

        # 默认使用传统 Paraformer 模型
        return "paraformer"

    @classmethod
    def register_loader(cls, model_type: str, loader_class: Type[BaseModelLoader]) -> None:
        """注册新的模型加载器

        Args:
            model_type: 模型类型标识
            loader_class: 加载器类
        """
        cls._LOADER_REGISTRY[model_type] = loader_class
        logger.info(f"已注册模型加载器: {model_type} -> {loader_class.__name__}")
