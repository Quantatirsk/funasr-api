# -*- coding: utf-8 -*-
"""
ASR 模型注册表
跟踪已加载的模型，供 API 层动态展示可用模型列表
"""

from typing import Set, List
from threading import Lock


class ModelRegistry:
    """已加载 ASR 模型注册表（线程安全）

    用于：
    1. 跟踪实际已加载的模型
    2. 供 API docs 动态展示可用模型
    3. 供验证器校验模型 ID
    """

    def __init__(self):
        self._loaded_models: Set[str] = set()
        self._lock = Lock()

    def register(self, model_id: str) -> None:
        """注册已加载的模型"""
        with self._lock:
            self._loaded_models.add(model_id)

    def unregister(self, model_id: str) -> None:
        """注销模型（用于热卸载等场景）"""
        with self._lock:
            self._loaded_models.discard(model_id)

    def is_loaded(self, model_id: str) -> bool:
        """检查模型是否已加载"""
        with self._lock:
            return model_id in self._loaded_models

    def list_models(self) -> List[str]:
        """获取已加载模型列表（按字母顺序）"""
        with self._lock:
            return sorted(self._loaded_models)

    def clear(self) -> None:
        """清空注册表"""
        with self._lock:
            self._loaded_models.clear()


# 全局注册表实例
model_registry = ModelRegistry()


def register_loaded_model(model_id: str) -> None:
    """便捷函数：注册已加载模型"""
    model_registry.register(model_id)


def get_available_models() -> List[str]:
    """便捷函数：获取可用模型列表"""
    return model_registry.list_models()


def is_model_available(model_id: str) -> bool:
    """便捷函数：检查模型是否可用"""
    return model_registry.is_loaded(model_id)
