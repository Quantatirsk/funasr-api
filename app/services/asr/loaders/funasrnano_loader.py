# -*- coding: utf-8 -*-
"""
Fun-ASR-Nano 模型加载器
端到端 Audio-LLM，内置所有功能
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from funasr import AutoModel

from .base_loader import BaseModelLoader
from ....core.exceptions import DefaultServerErrorException

logger = logging.getLogger(__name__)


class FunASRNanoModelLoader(BaseModelLoader):
    """Fun-ASR-Nano 模型加载器

    特性：
    - 端到端 Audio-LLM 架构
    - 内置 VAD、PUNC、多语言支持
    - 不支持外部 LM（使用内部 LLM）
    - 需要本地 model.py 代码文件
    """

    # 本地代码目录：implementations/funasrnano/
    LOCAL_CODE_DIR = Path(__file__).parent.parent / "implementations" / "funasrnano"

    @property
    def model_type(self) -> str:
        return "fun-asr-nano"

    @property
    def supports_external_vad(self) -> bool:
        return False

    @property
    def supports_external_punc(self) -> bool:
        return False

    @property
    def supports_lm(self) -> bool:
        return False

    def _resolve_model_path(self, model_id: str) -> str:
        """解析模型路径，优先使用本地缓存"""
        import os

        # ModelScope 缓存目录
        cache_dir = os.environ.get("MODELSCOPE_CACHE", os.path.expanduser("~/.cache/modelscope"))

        possible_paths = [
            Path(cache_dir) / "hub" / model_id,
            Path(cache_dir) / "models" / model_id,
        ]

        for local_path in possible_paths:
            if local_path.exists() and local_path.is_dir():
                return str(local_path)

        return model_id

    def _setup_local_code_path(self) -> None:
        """设置本地代码路径，使 Python 能导入 model.py

        将整个 funasrnano 包添加到 sys.path，使相对导入能正常工作
        """
        import sys

        parent_dir = str(self.LOCAL_CODE_DIR.parent)
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
            logger.debug(f"添加到 sys.path: {parent_dir}")

        logger.info(f"Fun-ASR-Nano 使用本地代码: {self.LOCAL_CODE_DIR}")

    def load(self) -> AutoModel:
        """加载 Fun-ASR-Nano 模型"""
        try:
            # 先设置本地代码路径
            self._setup_local_code_path()

            resolved_path = self._resolve_model_path(self.model_path)
            logger.info(f"正在加载 Fun-ASR-Nano 模型: {resolved_path}")

            # 先导入模型类，确保注册到 FunASR tables
            try:
                from funasrnano.model import FunASRNano

                logger.debug(f"FunASRNano 类已导入并注册: {FunASRNano}")
            except ImportError as e:
                logger.warning(f"预导入 FunASRNano 失败（可能不影响加载）: {e}")

            model_kwargs = {
                "model": resolved_path,
                "device": self.device,
                "trust_remote_code": True,
                # 不指定 remote_code，让 FunASR 通过 sys.path 找到已注册的 FunASRNano
                "disable_update": True,
                "disable_pbar": True,
                "disable_log": True,
                "local_files_only": True,
            }

            model = AutoModel(**model_kwargs)
            logger.info("Fun-ASR-Nano 模型加载成功（端到端 Audio-LLM）")
            return model

        except Exception as e:
            raise DefaultServerErrorException(f"Fun-ASR-Nano 模型加载失败: {str(e)}")

    def prepare_generate_kwargs(
        self,
        audio_path: Optional[str],
        hotwords: str = "",
        enable_punctuation: bool = False,  # noqa: ARG002
        enable_itn: bool = True,
        language: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """准备 Fun-ASR-Nano 推理参数

        Args:
            audio_path: 音频文件路径，批量推理时为 None
            hotwords: 热词（逗号分隔）
            enable_punctuation: 忽略（内置支持）
            enable_itn: 是否启用 ITN
            language: 语言代码（如 "中文"、"英文"）
            **kwargs: 额外参数
        """
        # Fun-ASR-Nano 需要列表格式的输入
        generate_kwargs: Dict[str, Any] = {
            "input": [audio_path],
            "cache": {},
            "batch_size": 1,  # 只支持 batch_size=1
        }

        # 处理热词
        if hotwords:
            hotword_list = [w.strip() for w in hotwords.split(",") if w.strip()]
            if hotword_list:
                generate_kwargs["hotwords"] = hotword_list

        # 处理语言
        if language:
            generate_kwargs["language"] = language

        # 处理 ITN
        generate_kwargs["itn"] = enable_itn

        return generate_kwargs
