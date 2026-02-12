# -*- coding: utf-8 -*-
"""
安全相关功能
包含鉴权、token验证等安全功能
"""

from typing import Optional
from fastapi import Request
from .config import settings


def mask_sensitive_data(
    data: str, mask_char: str = "*", keep_prefix: int = 4, keep_suffix: int = 4
) -> str:
    """遮盖敏感数据

    Args:
        data: 需要遮盖的数据
        mask_char: 遮盖字符
        keep_prefix: 保留前缀字符数
        keep_suffix: 保留后缀字符数

    Returns:
        遮盖后的数据
    """
    if not data or len(data) <= keep_prefix + keep_suffix:
        return data

    prefix = data[:keep_prefix]
    suffix = data[-keep_suffix:] if keep_suffix > 0 else ""
    mask_length = len(data) - keep_prefix - keep_suffix
    mask = mask_char * mask_length

    return f"{prefix}{mask}{suffix}"


def validate_token_value(token: str, expected_token: Optional[str] = None) -> bool:
    """验证访问令牌

    Args:
        token: 客户端提供的token
        expected_token: 期望的token值（从环境变量读取），如果为None则鉴权可选

    Returns:
        bool: 验证结果
    """
    # 如果没有配置期望的token（None/空字符串），则鉴权是可选的
    normalized_expected_token = (
        expected_token.strip()
        if isinstance(expected_token, str)
        else expected_token
    )
    if not normalized_expected_token:
        return True

    # 如果配置了期望的token，则必须提供token
    if not token:
        return False

    # 简单的token格式验证（长度检查）
    if len(token) < 10:
        return False

    # 验证token是否匹配
    if token != normalized_expected_token:
        return False

    return True


def validate_token(request: Request, task_id: str = "") -> tuple[bool, str]:
    """验证X-NLS-Token头部"""
    # 获取认证token
    token = request.headers.get("X-NLS-Token")

    # 如果没有配置API_KEY环境变量（None/空字符串），则鉴权是可选的
    if not settings.API_KEY:
        return True, token or "optional"

    # 如果配置了API_KEY，则必须提供token
    if not token:
        return False, "缺少X-NLS-Token头部"

    if not validate_token_value(token, settings.API_KEY):
        masked_token = mask_sensitive_data(token)
        return False, f"Gateway:ACCESS_DENIED:The token '{masked_token}' is invalid!"

    return True, token


def validate_token_websocket(token: str, task_id: str = "") -> tuple[bool, str]:
    """验证WebSocket连接中的token"""
    # 如果没有配置API_KEY环境变量（None/空字符串），则鉴权是可选的
    if not settings.API_KEY:
        return True, token or "optional"

    # 如果配置了API_KEY，则必须提供token
    if not token:
        return False, "缺少token参数"

    if not validate_token_value(token, settings.API_KEY):
        masked_token = mask_sensitive_data(token)
        return False, f"Gateway:ACCESS_DENIED:The token '{masked_token}' is invalid!"

    return True, token
