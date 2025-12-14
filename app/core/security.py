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
    # 如果没有配置期望的token，则鉴权是可选的
    if expected_token is None:
        return True

    # 如果配置了期望的token，则必须提供token
    if not token:
        return False

    # 简单的token格式验证（长度检查）
    if len(token) < 10:
        return False

    # 验证token是否匹配
    if token != expected_token:
        return False

    return True


def validate_appkey(appkey: str, expected_appkey: Optional[str] = None) -> bool:
    """验证应用Appkey

    Args:
        appkey: 客户端提供的appkey
        expected_appkey: 期望的appkey值（从环境变量读取），如果为None则appkey可选

    Returns:
        bool: 验证结果
    """
    # 如果没有配置期望的appkey，则appkey是可选的
    if expected_appkey is None:
        return True

    # 如果配置了期望的appkey，则必须提供appkey
    if not appkey:
        return False

    # 简单的appkey格式验证（长度检查）
    if len(appkey) < 3:
        return False

    # 验证appkey是否匹配
    if appkey != expected_appkey:
        return False

    return True


def validate_token(request: Request, task_id: str = "") -> tuple[bool, str]:
    """验证X-NLS-Token头部"""
    # 获取认证token
    token = request.headers.get("X-NLS-Token")

    # 如果没有配置APPTOKEN环境变量，则鉴权是可选的
    if settings.APPTOKEN is None:
        return True, token or "optional"

    # 如果配置了APPTOKEN，则必须提供token
    if not token:
        return False, "缺少X-NLS-Token头部"

    if not validate_token_value(token, settings.APPTOKEN):
        masked_token = mask_sensitive_data(token)
        return False, f"Gateway:ACCESS_DENIED:The token '{masked_token}' is invalid!"

    return True, token




def validate_request_appkey(appkey: str, task_id: str = "") -> tuple[bool, str]:
    """验证请求中的appkey参数"""
    # 如果没有配置APPKEY环境变量，则appkey是可选的
    if settings.APPKEY is None:
        return True, appkey or "optional"

    # 如果配置了APPKEY，则必须提供appkey
    if not appkey:
        return False, "缺少appkey参数"

    if not validate_appkey(appkey, settings.APPKEY):
        masked_appkey = mask_sensitive_data(appkey)
        return False, f"Gateway:ACCESS_DENIED:The appkey '{masked_appkey}' is invalid!"

    return True, appkey


def validate_token_websocket(token: str, task_id: str = "") -> tuple[bool, str]:
    """验证WebSocket连接中的token"""
    # 如果没有配置APPTOKEN环境变量，则鉴权是可选的
    if settings.APPTOKEN is None:
        return True, token or "optional"
    
    # 如果配置了APPTOKEN，则必须提供token
    if not token:
        return False, "缺少token参数"
    
    if not validate_token_value(token, settings.APPTOKEN):
        masked_token = mask_sensitive_data(token)
        return False, f"Gateway:ACCESS_DENIED:The token '{masked_token}' is invalid!"
    
    return True, token
