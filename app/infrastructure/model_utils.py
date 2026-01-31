# -*- coding: utf-8 -*-
"""
模型工具模块 - 提供模型路径解析等通用功能
"""

import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def resolve_model_path(model_id: Optional[str]) -> str:
    """将模型 ID 解析为本地缓存路径（如果存在）

    FunASR/ModelScope 的缓存目录结构:
    ~/.cache/modelscope/hub/{model_id}/

    如果本地缓存存在，返回本地路径；否则返回原始 model_id
    """
    if not model_id:
        raise ValueError("model_id 不能为空")

    # 获取 ModelScope 缓存目录
    cache_dir = os.environ.get("MODELSCOPE_CACHE", os.path.expanduser("~/.cache/modelscope"))

    # ModelScope 有两种可能的缓存路径结构
    possible_paths = [
        Path(cache_dir) / "hub" / model_id,
        Path(cache_dir) / "models" / model_id,
    ]

    # 检查哪个路径存在
    for local_path in possible_paths:
        if local_path.exists() and local_path.is_dir():
            resolved = str(local_path)
            logger.info(f"模型 {model_id} 使用本地缓存: {resolved}")
            return resolved

    # 都不存在，返回模型ID（运行时会自动下载）
    logger.warning(f"模型 {model_id} 本地缓存不存在，将在运行时下载")
    return model_id
