# -*- coding: utf-8 -*-
"""
统一音频参数验证模块
为ASR API提供一致的参数验证逻辑
"""

import json
from pathlib import Path
from typing import Optional, List
from ...core.exceptions import InvalidParameterException
from ...models.common import SampleRate, AudioFormat
from ...core.config import settings


def _detect_qwen_model_by_vram() -> str:
    """根据显存检测应该使用哪个 Qwen 模型

    < 32GB 用 0.6b, >= 32GB 用 1.7b
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return "qwen3-asr-0.6b"

        # 获取最小显存（多卡情况下）
        gpu_count = torch.cuda.device_count()
        min_vram = float('inf')
        for i in range(gpu_count):
            vram = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            min_vram = min(min_vram, vram)

        if min_vram >= 32:
            return "qwen3-asr-1.7b"
        else:
            return "qwen3-asr-0.6b"
    except Exception:
        return "qwen3-asr-0.6b"


def _get_active_qwen_model() -> str:
    """获取当前激活的 Qwen 模型

    根据 QWEN_ASR_MODEL 环境变量或显存自动检测
    """
    model_config = settings.QWEN_ASR_MODEL

    if model_config == "Qwen3-ASR-1.7B":
        return "qwen3-asr-1.7b"
    elif model_config == "Qwen3-ASR-0.6B":
        return "qwen3-asr-0.6b"
    else:  # auto
        return _detect_qwen_model_by_vram()


def _load_supported_models() -> List[str]:
    """从 models.json 加载支持的模型列表"""
    try:
        models_file = Path(settings.ASR_MODELS_CONFIG)
        if models_file.exists():
            with open(models_file, "r", encoding="utf-8") as f:
                config = json.load(f)
            return list(config.get("models", {}).keys())
    except Exception:
        pass
    # Fallback: 如果加载失败，返回默认列表
    return ["qwen3-asr-1.7b", "paraformer-large"]


def _get_dynamic_model_list() -> List[str]:
    """获取动态的模型列表（根据显存配置）

    返回的列表中，Qwen 模型在前，Paraformer 在后
    且只返回当前显存配置下可用的 Qwen 模型
    无 CUDA 时禁用 Qwen3（vLLM 不支持 CPU），只返回 paraformer-large
    """
    import torch

    if not torch.cuda.is_available():
        # CPU 模式：禁用 Qwen3，只返回 paraformer
        return ["paraformer-large"] if "paraformer-large" in _load_supported_models() else []

    active_qwen = _get_active_qwen_model()

    # Qwen 模型在前，按显存配置只返回可用的那个
    models = [active_qwen]

    # Paraformer 在后
    if "paraformer-large" in _load_supported_models():
        models.append("paraformer-large")

    return models


def _get_default_model() -> str:
    """获取默认模型（根据显存配置）"""
    return _get_active_qwen_model()


class AudioParamsValidator:
    """音频参数验证器 - 统一验证ASR相关参数"""

    # 支持的模型ID（动态加载，根据显存配置）
    # 使用 _get_dynamic_model_list() 获取当前可用的模型列表
    SUPPORTED_MODELS = _load_supported_models()

    # 支持的音频格式
    SUPPORTED_FORMATS = AudioFormat.get_enums()

    # 支持的采样率
    SUPPORTED_SAMPLE_RATES = SampleRate.get_enums()

    @staticmethod
    def validate_model_id(model_id: Optional[str]) -> str:
        """
        验证模型ID

        Args:
            model_id: 模型ID字符串，支持模糊匹配如 "qwen3-asr" 自动路由到已启动的版本

        Returns:
            验证后的模型ID（如果输入为None则返回默认值）

        Raises:
            InvalidParameterException: 当模型ID不支持时
        """
        # 动态获取最新支持的模型列表（避免类变量缓存问题）
        supported_models = _load_supported_models()

        if not model_id:
            return _get_default_model()

        # 处理模糊匹配：qwen3-asr 自动映射到当前激活的模型版本
        if model_id.lower() == "qwen3-asr":
            return _get_active_qwen_model()

        if model_id not in supported_models:
            raise InvalidParameterException(
                f"不支持的模型ID: {model_id}。支持的模型: {', '.join(supported_models)}"
            )

        return model_id

    @staticmethod
    def validate_audio_format(format: Optional[str]) -> str:
        """
        验证音频格式

        Args:
            format: 音频格式字符串

        Returns:
            验证后的音频格式（小写，如果输入为None则返回默认值）

        Raises:
            InvalidParameterException: 当音频格式不支持时
        """
        if not format:
            return "wav"  # 默认格式

        format_lower = format.lower()

        if format_lower not in AudioParamsValidator.SUPPORTED_FORMATS:
            raise InvalidParameterException(
                f"不支持的音频格式: {format}。支持的格式: {', '.join(AudioParamsValidator.SUPPORTED_FORMATS)}"
            )

        return format_lower

    @staticmethod
    def validate_sample_rate(rate: Optional[int]) -> int:
        """
        验证采样率

        Args:
            rate: 采样率数值

        Returns:
            验证后的采样率（如果输入为None则返回默认值）

        Raises:
            InvalidParameterException: 当采样率不支持时
        """
        if not rate:
            return 16000  # 默认采样率

        if rate not in AudioParamsValidator.SUPPORTED_SAMPLE_RATES:
            raise InvalidParameterException(
                f"不支持的采样率: {rate}。支持的采样率: {', '.join(map(str, AudioParamsValidator.SUPPORTED_SAMPLE_RATES))}"
            )

        return rate

    @staticmethod
    def validate_audio_size(size_bytes: int, max_size: int, task_id: str = ""):
        """
        验证音频文件大小

        Args:
            size_bytes: 文件大小（字节）
            max_size: 最大允许大小（字节）
            task_id: 任务ID（用于错误响应）

        Raises:
            InvalidParameterException: 当文件大小超过限制时
        """
        if size_bytes > max_size:
            max_mb = max_size // 1024 // 1024
            raise InvalidParameterException(
                f"音频文件过大，最大支持 {max_mb}MB", task_id=task_id
            )

    @staticmethod
    def validate_language(language: Optional[str]) -> Optional[str]:
        """
        验证语言代码

        Args:
            language: ISO-639-1 语言代码

        Returns:
            验证后的语言代码（小写）或None
        """
        if not language:
            return None

        # 支持的语言代码列表
        supported_languages = [
            "zh",  # 中文
            "en",  # 英文
            "ja",  # 日文
            "ko",  # 韩文
            "fr",  # 法文
            "de",  # 德文
            "es",  # 西班牙文
            "it",  # 意大利文
            "pt",  # 葡萄牙文
            "ru",  # 俄文
            "ar",  # 阿拉伯文
            "hi",  # 印地文
            "th",  # 泰文
            "vi",  # 越南文
            "id",  # 印尼文
            "ms",  # 马来文
            "tr",  # 土耳其文
            "pl",  # 波兰文
            "nl",  # 荷兰文
            "sv",  # 瑞典文
        ]

        language_lower = language.lower()

        if language_lower not in supported_languages:
            # 对于不支持的语言，记录警告但允许通过（让模型自动检测）
            return language_lower

        return language_lower


# 便捷函数 - 用于快速验证
def validate_model_id(model_id: Optional[str]) -> str:
    """验证模型ID的便捷函数"""
    return AudioParamsValidator.validate_model_id(model_id)


def validate_audio_format(format: Optional[str]) -> str:
    """验证音频格式的便捷函数"""
    return AudioParamsValidator.validate_audio_format(format)


def validate_sample_rate(rate: Optional[int]) -> int:
    """验证采样率的便捷函数"""
    return AudioParamsValidator.validate_sample_rate(rate)


def validate_audio_size(size_bytes: int, max_size: int, task_id: str = ""):
    """验证音频文件大小的便捷函数"""
    return AudioParamsValidator.validate_audio_size(size_bytes, max_size, task_id)
