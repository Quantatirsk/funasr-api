# -*- coding: utf-8 -*-
"""
ASR模型管理器
支持多模型缓存和动态加载
重构以支持离线和实时模型的分离管理
"""

import json
import torch
from typing import Dict, Any, Optional, List
from pathlib import Path

from ...core.config import settings
from ...core.exceptions import DefaultServerErrorException, InvalidParameterException
from .engine import BaseASREngine, FunASREngine


class ModelConfig:
    """模型配置类"""

    def __init__(self, model_id: str, config: Dict[str, Any]):
        self.model_id = model_id
        self.name = config["name"]
        self.engine = config["engine"]
        self.description = config.get("description", "")
        self.languages = config.get("languages", [])
        self.is_default = config.get("default", False)
        self.supports_realtime = config.get("supports_realtime", False)

        # 模型路径结构
        self.models = config.get("models", {})
        self.offline_model_path = self.models.get("offline")
        self.realtime_model_path = self.models.get("realtime")

    @property
    def has_offline_model(self) -> bool:
        """是否有离线模型"""
        return bool(self.offline_model_path)

    @property
    def has_realtime_model(self) -> bool:
        """是否有实时模型"""
        return bool(self.realtime_model_path)

    def get_model_path(self, model_type: str = "offline") -> Optional[str]:
        """根据模型类型获取模型路径"""
        if model_type == "offline":
            return self.offline_model_path
        elif model_type == "realtime":
            return self.realtime_model_path
        return None


class ModelManager:
    """模型管理器，支持多模型缓存"""

    def __init__(self):
        self._models_config: Dict[str, ModelConfig] = {}
        self._loaded_engines: Dict[str, BaseASREngine] = {}
        self._default_model_id: Optional[str] = None
        self._load_models_config()

    def _load_models_config(self) -> None:
        """加载模型配置文件"""
        models_file = Path(settings.models_config_path)
        if not models_file.exists():
            raise DefaultServerErrorException("models.json 配置文件不存在")

        try:
            with open(models_file, "r", encoding="utf-8") as f:
                config = json.load(f)

            for model_id, model_config in config["models"].items():
                self._models_config[model_id] = ModelConfig(model_id, model_config)
                if model_config.get("default", False):
                    self._default_model_id = model_id

            if not self._default_model_id and self._models_config:
                # 如果没有指定默认模型，选择第一个
                self._default_model_id = list(self._models_config.keys())[0]

        except (json.JSONDecodeError, KeyError) as e:
            raise DefaultServerErrorException(f"模型配置文件格式错误: {str(e)}")

    def get_model_config(self, model_id: Optional[str] = None) -> ModelConfig:
        """获取模型配置"""
        if model_id is None:
            model_id = self._default_model_id

        if not model_id:
            raise InvalidParameterException("未指定模型且没有默认模型")

        if model_id not in self._models_config:
            available_models = ", ".join(self._models_config.keys())
            raise InvalidParameterException(
                f"未知的模型: {model_id}，可用模型: {available_models}"
            )

        return self._models_config[model_id]

    def list_models(self) -> List[Dict[str, Any]]:
        """列出所有可用模型"""
        models = []
        for model_id, config in self._models_config.items():
            # 检查模型文件是否存在
            offline_path_exists = False
            realtime_path_exists = False

            if config.offline_model_path:
                offline_model_path = (
                    Path(settings.MODELSCOPE_PATH) / config.offline_model_path
                )
                offline_path_exists = offline_model_path.exists()

            if config.realtime_model_path:
                realtime_model_path = (
                    Path(settings.MODELSCOPE_PATH) / config.realtime_model_path
                )
                realtime_path_exists = realtime_model_path.exists()

            # 检查模型是否已加载
            loaded = model_id in self._loaded_engines

            models.append(
                {
                    "id": model_id,
                    "name": config.name,
                    "engine": config.engine,
                    "description": config.description,
                    "languages": config.languages,
                    "default": config.is_default,
                    "loaded": loaded,
                    "supports_realtime": config.supports_realtime,
                    "offline_model": (
                        {
                            "path": config.offline_model_path,
                            "exists": offline_path_exists,
                        }
                        if config.offline_model_path
                        else None
                    ),
                    "realtime_model": (
                        {
                            "path": config.realtime_model_path,
                            "exists": realtime_path_exists,
                        }
                        if config.realtime_model_path
                        else None
                    ),
                    "asr_model_mode": settings.ASR_MODEL_MODE,
                }
            )

        return models

    def get_asr_engine(self, model_id: Optional[str] = None) -> BaseASREngine:
        """获取ASR引擎，支持缓存"""
        if model_id is None:
            model_id = self._default_model_id

        if not model_id:
            raise InvalidParameterException("未指定模型且没有默认模型")

        # 如果已经加载，直接返回
        if model_id in self._loaded_engines:
            return self._loaded_engines[model_id]

        # 加载新模型
        config = self.get_model_config(model_id)
        engine = self._create_engine(config)

        # 缓存引擎
        self._loaded_engines[model_id] = engine

        return engine

    def _create_engine(self, config: ModelConfig) -> BaseASREngine:
        """根据配置创建ASR引擎"""
        if config.engine.lower() == "funasr":
            return FunASREngine(
                offline_model_path=config.offline_model_path,
                realtime_model_path=config.realtime_model_path,
                device=settings.DEVICE,
                vad_model=settings.VAD_MODEL,
                vad_model_revision=settings.VAD_MODEL_REVISION,
                punc_model=settings.PUNC_MODEL,
                punc_model_revision=settings.PUNC_MODEL_REVISION,
                punc_realtime_model=settings.PUNC_REALTIME_MODEL,
            )
        else:
            raise InvalidParameterException(f"不支持的引擎类型: {config.engine}")

    def unload_model(self, model_id: str) -> bool:
        """卸载指定模型"""
        if model_id in self._loaded_engines:
            del self._loaded_engines[model_id]
            # 强制垃圾回收
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return True
        return False

    def get_memory_usage(self) -> Dict[str, Any]:
        """获取内存使用情况"""
        memory_info = {
            "model_list": list(self._loaded_engines.keys()),
            "loaded_count": len(self._loaded_engines),
            "asr_model_mode": settings.ASR_MODEL_MODE,
        }

        if torch.cuda.is_available():
            memory_info["gpu_memory"] = {
                "allocated": f"{torch.cuda.memory_allocated() / 1024**3:.2f}GB",
                "cached": f"{torch.cuda.memory_reserved() / 1024**3:.2f}GB",
                "max_allocated": f"{torch.cuda.max_memory_allocated() / 1024**3:.2f}GB",
            }

        return memory_info

    def clear_cache(self) -> None:
        """清空模型缓存"""
        self._loaded_engines.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def validate_model_mode_compatibility(self, model_id: str) -> Dict[str, Any]:
        """验证模型与当前ASR_MODEL_MODE的兼容性"""
        config = self.get_model_config(model_id)
        mode = settings.ASR_MODEL_MODE.lower()

        errors = []

        if mode == "offline" and not config.has_offline_model:
            errors.append(
                f"模型 {model_id} 没有离线版本，但 ASR_MODEL_MODE 设置为 offline"
            )
        elif mode == "realtime" and not config.has_realtime_model:
            errors.append(
                f"模型 {model_id} 没有实时版本，但 ASR_MODEL_MODE 设置为 realtime"
            )
        elif mode == "all":
            if not config.has_offline_model and not config.has_realtime_model:
                errors.append(f"模型 {model_id} 既没有离线版本也没有实时版本")

        return {
            "model_id": model_id,
            "mode": mode,
            "errors": errors,
            "compatible": len(errors) == 0,
        }


# 全局模型管理器实例
_model_manager: Optional[ModelManager] = None


def get_model_manager() -> ModelManager:
    """获取全局模型管理器实例"""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager
