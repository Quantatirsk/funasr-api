# -*- coding: utf-8 -*-
"""
模型工具模块 - 提供模型路径解析等通用功能
"""

import logging
from pathlib import Path
from typing import Optional

from app.core.config import settings

logger = logging.getLogger(__name__)


def resolve_model_path(model_id: Optional[str]) -> str:
    """将模型 ID 解析为本地缓存路径（如果存在）

    ModelScope 标准缓存目录结构:
    ~/.cache/modelscope/hub/models/{model_id}/

    如果本地缓存存在，返回本地路径；否则返回原始 model_id
    """
    if not model_id:
        raise ValueError("model_id 不能为空")

    # 标准 ModelScope 路径
    local_path = Path(settings.MODELSCOPE_PATH) / model_id

    if local_path.exists() and local_path.is_dir():
        resolved = str(local_path)
        logger.info(f"模型 {model_id} 使用本地缓存: {resolved}")
        return resolved

    logger.warning(f"模型 {model_id} 本地缓存不存在，将在运行时下载")
    return model_id
