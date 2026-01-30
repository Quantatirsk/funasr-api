# -*- coding: utf-8 -*-
"""
模型加载器抽象基类
定义所有模型加载器的通用接口
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BaseModelLoader(ABC):
    """模型加载器抽象基类

    所有具体的模型加载器都应该继承此类，实现特定模型的加载逻辑。
    """

    def __init__(
        self,
        model_path: str,
        device: str,
        enable_lm: bool = False,
        lm_model: Optional[str] = None,
        lm_weight: float = 0.15,
        lm_beam_size: int = 10,
        **kwargs
    ):
        """初始化加载器

        Args:
            model_path: 模型路径或模型ID
            device: 推理设备 (cpu/cuda:0/npu:0)
            enable_lm: 是否启用语言模型
            lm_model: 语言模型路径
            lm_weight: 语言模型权重
            lm_beam_size: 语言模型 beam size
            **kwargs: 额外的模型特定参数
        """
        self.model_path = model_path
        self.device = device
        self.enable_lm = enable_lm
        self.lm_model = lm_model
        self.lm_weight = lm_weight
        self.lm_beam_size = lm_beam_size
        self.extra_kwargs = kwargs

    @abstractmethod
    def load(self) -> Any:
        """加载模型

        Returns:
            加载好的模型实例

        Raises:
            DefaultServerErrorException: 模型加载失败时抛出
        """
        pass

    @abstractmethod
    def prepare_generate_kwargs(
        self,
        audio_path: Optional[str],
        hotwords: str = "",
        enable_punctuation: bool = False,
        enable_itn: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """准备模型推理参数

        Args:
            audio_path: 音频文件路径，批量推理时为 None
            hotwords: 热词
            enable_punctuation: 是否启用标点
            enable_itn: 是否启用ITN
            **kwargs: 额外参数

        Returns:
            用于 model.generate() 的参数字典
        """
        pass

    @property
    @abstractmethod
    def supports_external_vad(self) -> bool:
        """是否支持外部 VAD 模型"""
        pass

    @property
    @abstractmethod
    def supports_external_punc(self) -> bool:
        """是否支持外部标点模型"""
        pass

    @property
    @abstractmethod
    def supports_lm(self) -> bool:
        """是否支持外部语言模型"""
        pass

    @property
    @abstractmethod
    def model_type(self) -> str:
        """模型类型标识"""
        pass
