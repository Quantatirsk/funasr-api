# -*- coding: utf-8 -*-
"""
统一音频参数验证模块
为ASR API提供一致的参数验证逻辑
"""

from typing import Optional
from ...core.exceptions import InvalidParameterException
from ...models.common import SampleRate, AudioFormat


class AudioParamsValidator:
    """音频参数验证器 - 统一验证ASR相关参数"""

    # 支持的模型ID
    SUPPORTED_MODELS = ["qwen3-asr-1.7b", "paraformer-large", "fun-asr-nano"]

    # 支持的音频格式
    SUPPORTED_FORMATS = AudioFormat.get_enums()

    # 支持的采样率
    SUPPORTED_SAMPLE_RATES = SampleRate.get_enums()

    @staticmethod
    def validate_model_id(model_id: Optional[str]) -> str:
        """
        验证模型ID

        Args:
            model_id: 模型ID字符串

        Returns:
            验证后的模型ID（如果输入为None则返回默认值）

        Raises:
            InvalidParameterException: 当模型ID不支持时
        """
        if not model_id:
            return "qwen3-asr-1.7b"  # 默认模型

        if model_id not in AudioParamsValidator.SUPPORTED_MODELS:
            raise InvalidParameterException(
                f"不支持的模型ID: {model_id}。支持的模型: {', '.join(AudioParamsValidator.SUPPORTED_MODELS)}"
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
