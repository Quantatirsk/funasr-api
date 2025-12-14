# -*- coding: utf-8 -*-
"""
通用工具函数
包含任务ID生成、参数验证等通用功能
"""

import uuid
import hashlib
import time
import re
from typing import Optional


def generate_task_id(prefix: str = "") -> str:
    """生成唯一的任务ID

    Args:
        prefix: 任务ID前缀

    Returns:
        生成的任务ID
    """
    timestamp = str(int(time.time() * 1000))
    random_id = str(uuid.uuid4()).replace("-", "")
    combined = timestamp + random_id

    # 使用MD5哈希生成32位字符串
    task_id = hashlib.md5(combined.encode()).hexdigest()

    if prefix:
        return f"{prefix}_{task_id}"
    return task_id


def validate_text_input(text: str, max_length: int = 10000) -> tuple[bool, str]:
    """验证输入文本

    Args:
        text: 待验证的文本
        max_length: 最大长度限制

    Returns:
        (is_valid, message): 验证结果和消息
    """
    if not text or not text.strip():
        return False, "文本内容不能为空"

    text = text.strip()

    if len(text) > max_length:
        return False, f"文本长度超过限制，最大支持{max_length}个字符"

    # 检查是否包含有效字符
    if not re.search(r"[\u4e00-\u9fff\w\s]", text):
        return False, "文本内容无效，请输入有效的中文、英文或数字"

    return True, "验证通过"


def parse_language_code(lang_code: Optional[str]) -> str:
    """解析语言代码

    Args:
        lang_code: 语言代码（如 zh, zh-cn, en, ja等）

    Returns:
        标准化的语言代码
    """
    if not lang_code:
        return "zh"  # 默认中文

    lang_code = lang_code.lower().strip()

    # 语言代码映射
    lang_mapping = {
        "zh": "zh",
        "zh-cn": "zh",
        "zh-tw": "zh",
        "zh-hk": "zh",
        "en": "en",
        "en-us": "en",
        "en-gb": "en",
        "ja": "jp",
        "jp": "jp",
        "ko": "kr",
        "kr": "kr",
        "yue": "yue",  # 粤语
    }

    return lang_mapping.get(lang_code, "zh")
