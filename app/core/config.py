# -*- coding: utf-8 -*-
"""
统一配置管理
ASR语音识别配置选项
"""

import os
from typing import Optional
from pathlib import Path


class Settings:
    """统一应用配置类"""

    # 应用信息
    APP_NAME: str = "FunASR-API Server"
    APP_VERSION: str = "1.0.0"
    APP_DESCRIPTION: str = "基于FunASR的语音识别API服务"

    # 服务器配置
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False

    # 鉴权配置
    APPTOKEN: Optional[str] = None  # 从环境变量APPTOKEN读取，如果为None则鉴权可选
    APPKEY: Optional[str] = None  # 从环境变量APPKEY读取，如果为None则appkey可选

    # 设备配置
    DEVICE: str = "auto"  # auto, cpu, cuda:0, npu:0

    # 路径配置
    BASE_DIR: Path = Path(__file__).parent.parent.parent
    TEMP_DIR: str = "temp"
    DATA_DIR: str = "data"  # 数据持久化目录
    MODELSCOPE_PATH: str = os.path.expanduser("~/.cache/modelscope/hub")

    # 日志配置
    LOG_LEVEL: str = "INFO"
    LOG_FILE: Optional[str] = str(BASE_DIR / "logs" / "funasr-api.log")
    LOG_MAX_BYTES: int = 20 * 1024 * 1024  # 20MB
    LOG_BACKUP_COUNT: int = 50  # 保留50个备份文件

    # ASR模型配置
    FUNASR_AUTOMODEL_KWARGS = {
        "trust_remote_code": False,
        "disable_update": True,
        "disable_pbar": True,
        "disable_log": True,  # 禁用FunASR的tables输出
        "local_files_only": True,  # 强制使用本地模型，禁止联网下载
    }
    ASR_MODELS_CONFIG: str = str(BASE_DIR / "app/services/asr/models.json")
    ASR_MODEL_MODE: str = "all"  # ASR模型加载模式: realtime, offline, all
    ASR_ENABLE_REALTIME_PUNC: bool = True  # 是否启用实时标点模型（用于中间结果展示）
    AUTO_LOAD_CUSTOM_ASR_MODELS: str = (
        ""  # 启动时自动加载的自定义ASR模型列表（逗号分隔，如: fun-asr-nano）
    )
    VAD_MODEL: str = "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"
    PUNC_MODEL: str = "iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch"
    PUNC_REALTIME_MODEL: str = (
        "iic/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727"
    )

    # 语言模型配置
    LM_MODEL: str = "iic/speech_ngram_lm_zh-cn-ai-wesp-fst"
    LM_WEIGHT: float = 0.15  # 语言模型权重，建议范围 0.1-0.3
    LM_BEAM_SIZE: int = 10  # 语言模型解码 beam size
    ASR_ENABLE_LM: bool = True  # 是否启用语言模型（默认启用）

    # 流式ASR远场过滤配置
    ASR_ENABLE_NEARFIELD_FILTER: bool = True  # 是否启用远场声音过滤
    ASR_NEARFIELD_RMS_THRESHOLD: float = 0.01  # RMS能量阈值（宽松模式，适合大多数场景）
    ASR_NEARFIELD_FILTER_LOG_ENABLED: bool = True  # 是否记录过滤日志（默认启用）

    # 音频处理配置
    MAX_AUDIO_SIZE: int = 2048 * 1024 * 1024  # 2GB

    # 批处理推理配置（GPU 真并行）
    ASR_BATCH_SIZE: int = 8  # ASR 批处理大小（同时推理的片段数），建议 2-8

    def __init__(self):
        """从环境变量读取配置"""
        self._load_from_env()
        self._ensure_directories()

    def _load_from_env(self):
        """从环境变量加载配置"""
        # 服务器配置
        self.HOST = os.getenv("HOST", self.HOST)
        self.PORT = int(os.getenv("PORT", str(self.PORT)))
        self.DEBUG = os.getenv("DEBUG", "false").lower() == "true"

        # 日志配置
        self.LOG_LEVEL = os.getenv("LOG_LEVEL", self.LOG_LEVEL)
        self.LOG_FILE = os.getenv("LOG_FILE", self.LOG_FILE)
        self.LOG_MAX_BYTES = int(os.getenv("LOG_MAX_BYTES", str(self.LOG_MAX_BYTES)))
        self.LOG_BACKUP_COUNT = int(
            os.getenv("LOG_BACKUP_COUNT", str(self.LOG_BACKUP_COUNT))
        )

        # 鉴权配置
        self.APPTOKEN = os.getenv("APPTOKEN", self.APPTOKEN)
        self.APPKEY = os.getenv("APPKEY", self.APPKEY)

        # 设备配置
        self.DEVICE = os.getenv("DEVICE", self.DEVICE)

        # ASR模型配置
        self.ASR_MODEL_MODE = os.getenv("ASR_MODEL_MODE", self.ASR_MODEL_MODE)
        self.ASR_ENABLE_REALTIME_PUNC = (
            os.getenv("ASR_ENABLE_REALTIME_PUNC", "true").lower() == "true"
        )
        self.AUTO_LOAD_CUSTOM_ASR_MODELS = os.getenv(
            "AUTO_LOAD_CUSTOM_ASR_MODELS", self.AUTO_LOAD_CUSTOM_ASR_MODELS
        )

        # 语言模型配置
        self.ASR_ENABLE_LM = (
            os.getenv("ASR_ENABLE_LM", "true").lower() == "true"
        )
        self.LM_WEIGHT = float(os.getenv("LM_WEIGHT", str(self.LM_WEIGHT)))
        self.LM_BEAM_SIZE = int(os.getenv("LM_BEAM_SIZE", str(self.LM_BEAM_SIZE)))

        # 远场过滤配置
        self.ASR_ENABLE_NEARFIELD_FILTER = (
            os.getenv("ASR_ENABLE_NEARFIELD_FILTER", "true").lower() == "true"
        )
        self.ASR_NEARFIELD_RMS_THRESHOLD = float(
            os.getenv(
                "ASR_NEARFIELD_RMS_THRESHOLD", str(self.ASR_NEARFIELD_RMS_THRESHOLD)
            )
        )
        self.ASR_NEARFIELD_FILTER_LOG_ENABLED = (
            os.getenv("ASR_NEARFIELD_FILTER_LOG_ENABLED", "true").lower() == "true"
        )

        # 音频处理配置
        # 支持简化格式：纯数字表示MB，或带单位（如 2048MB, 2GB）
        max_audio_size_str = os.getenv("MAX_AUDIO_SIZE")
        if max_audio_size_str:
            self.MAX_AUDIO_SIZE = self._parse_size(max_audio_size_str)

        self.ASR_BATCH_SIZE = int(
            os.getenv("ASR_BATCH_SIZE", str(self.ASR_BATCH_SIZE))
        )

    def _parse_size(self, size_str: str) -> int:
        """解析带单位的大小字符串

        支持格式：
        - 纯数字：视为 MB（如 2048 = 2048MB = 2147483648 bytes）
        - 带单位：如 2GB, 2048MB, 1.5GB
        """
        size_str = size_str.strip().upper()

        # 如果纯数字，视为 MB
        if size_str.isdigit():
            return int(size_str) * 1024 * 1024

        # 带单位的处理
        if size_str.endswith('GB'):
            return int(float(size_str[:-2]) * 1024 * 1024 * 1024)
        elif size_str.endswith('MB'):
            return int(float(size_str[:-2]) * 1024 * 1024)
        elif size_str.endswith('KB'):
            return int(float(size_str[:-2]) * 1024)
        else:
            # 默认视为字节
            return int(size_str)

    def _ensure_directories(self):
        """确保必需的目录存在"""
        os.makedirs(self.TEMP_DIR, exist_ok=True)
        os.makedirs(self.DATA_DIR, exist_ok=True)

    @property
    def models_config_path(self) -> str:
        """获取模型配置文件的完整路径"""
        return str(self.BASE_DIR / self.ASR_MODELS_CONFIG)

    @property
    def docs_url(self) -> Optional[str]:
        """获取文档URL"""
        return "/docs"

    @property
    def redoc_url(self) -> Optional[str]:
        """获取ReDoc URL"""
        return "/redoc"


# 全局配置实例
settings = Settings()
