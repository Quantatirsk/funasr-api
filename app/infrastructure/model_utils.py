# -*- coding: utf-8 -*-
"""
模型工具模块 - 提供模型路径解析等通用功能
"""

import logging
import os
from pathlib import Path
from typing import Optional

from app.core.config import settings

logger = logging.getLogger(__name__)


def resolve_model_path(model_id: Optional[str], env_var: Optional[str] = None) -> str:
    """将模型 ID 解析为本地绝对路径

    解析优先级（从高到低）：
    1. 环境变量指定的路径（如果 env_var 提供且存在）
    2. 绝对路径（直接返回）
    3. HuggingFace 缓存路径: ~/.cache/huggingface/hub/models--{org}--{model}/snapshots/{commit}/
    4. ModelScope 缓存路径: ~/.cache/modelscope/hub/models/{model_id}/
    5. 原始 model_id（回退）

    Args:
        model_id: 模型 ID（如 "Qwen/Qwen3-ASR-0.6B"）或路径
        env_var: 可选的环境变量名，如果设置则优先使用

    Returns:
        解析后的本地绝对路径
    """
    if not model_id:
        raise ValueError("model_id 不能为空")

    # 1. 检查环境变量（最高优先级）
    if env_var:
        env_path = os.getenv(env_var)
        if env_path:
            logger.info(f"模型 {model_id} 使用环境变量 {env_var} 路径: {env_path}")
            return env_path

    # 2. 已是绝对路径
    if os.path.isabs(model_id):
        return model_id

    # 解析模型组织名和模型名
    # 处理格式: "org/model" -> org, model
    parts = model_id.split("/")
    if len(parts) == 2:
        org, model = parts
    else:
        # 没有组织名的模型，直接尝试作为目录名
        org, model = None, model_id

    # 3. 检查 HuggingFace 缓存路径
    # HF 缓存格式: models--{org}--{model}/snapshots/{commit}/
    hf_cache = _find_hf_cache_path(org, model) if org else None
    if hf_cache:
        logger.info(f"模型 {model_id} 使用 HuggingFace 缓存: {hf_cache}")
        return str(hf_cache)

    # 4. 检查 ModelScope 缓存路径
    ms_path = Path(settings.MODELSCOPE_PATH) / model_id
    if ms_path.exists() and ms_path.is_dir():
        logger.info(f"模型 {model_id} 使用 ModelScope 缓存: {ms_path}")
        return str(ms_path)

    # 5. 回退到原始 model_id
    logger.warning(f"模型 {model_id} 本地缓存不存在，将尝试运行时解析")
    return model_id


def _find_hf_cache_path(org: str, model: str) -> Optional[Path]:
    """在 HuggingFace 缓存中查找模型路径

    HF 缓存结构:
    ~/.cache/huggingface/hub/
        models--{org}--{model}/
            snapshots/
                {commit_hash}/
                    <实际模型文件>

    Returns:
        最新 snapshot 的绝对路径，如果不存在返回 None
    """
    # 构建缓存目录名
    cache_name = f"models--{org}--{model}"
    hf_hub_path = Path.home() / ".cache" / "huggingface" / "hub" / cache_name

    if not hf_hub_path.exists():
        return None

    # 查找 snapshots 目录
    snapshots_dir = hf_hub_path / "snapshots"
    if not snapshots_dir.exists():
        return None

    # 获取最新的 snapshot（按修改时间）
    snapshots = [d for d in snapshots_dir.iterdir() if d.is_dir()]
    if not snapshots:
        return None

    # 按修改时间排序，取最新的
    latest_snapshot = max(snapshots, key=lambda p: p.stat().st_mtime)
    return latest_snapshot


def resolve_forced_aligner_path(model_id: Optional[str]) -> Optional[str]:
    """解析 ForcedAligner 模型路径

    支持环境变量: QWEN_FORCED_ALIGNER_PATH
    """
    if not model_id:
        return None
    return resolve_model_path(model_id, env_var="QWEN_FORCED_ALIGNER_PATH")
