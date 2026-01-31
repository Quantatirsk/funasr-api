# -*- coding: utf-8 -*-
"""
音频处理服务模块

提供统一的音频处理服务层，封装音频下载、格式转换、归一化等功能。
"""

from .audio_service import AudioProcessingService, get_audio_service

__all__ = ["AudioProcessingService", "get_audio_service"]
