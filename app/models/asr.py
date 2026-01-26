# -*- coding: utf-8 -*-
"""
ASR数据模型
定义语音识别相关的请求和响应模型
"""

from typing import Optional, List, Union
from pydantic import BaseModel, Field

from .common import (
    SampleRate,
    BaseResponse,
    HealthCheckResponse,
    ErrorResponse,
)


# ============= 请求模型 =============


class ASRQueryParams(BaseModel):
    """ASR接口查询参数模型"""

    # 1. 核心参数
    model_id: Optional[str] = Field(
        default=None,
        description="ASR模型ID，不指定则使用默认模型(paraformer-large)",
        max_length=64,
    )

    # 2. 输入源
    audio_address: Optional[str] = Field(
        default=None,
        description="音频文件下载链接（HTTP/HTTPS），格式自动识别",
        max_length=512,
    )

    # 3. 音频属性
    sample_rate: Optional[SampleRate] = Field(
        default=SampleRate.RATE_16000,
        description=f"音频采样率（Hz）。支持: {', '.join(map(str, SampleRate.get_enums()))}",
    )

    # 4. 功能开关
    enable_speaker_diarization: Optional[bool] = Field(
        default=True,
        description="是否启用说话人分离。启用后响应会包含 speaker_id",
    )

    # 5. 增强选项
    vocabulary_id: Optional[str] = Field(
        default=None,
        description="热词字符串，格式：热词1 权重1 热词2 权重2（如：阿里巴巴 20 腾讯 15）",
        max_length=512,
    )

    # 6. 认证参数
    appkey: Optional[str] = Field(
        default=None,
        description="应用Appkey，用于API调用认证",
        min_length=1,
        max_length=64,
    )


# ============= 响应模型 =============


class ASRSegment(BaseModel):
    """ASR 识别分段结果"""

    text: str = Field(
        ...,
        description="该段识别文本",
    )
    start_time: float = Field(
        ...,
        description="段落开始时间（秒）",
    )
    end_time: float = Field(
        ...,
        description="段落结束时间（秒）",
    )
    speaker_id: Optional[str] = Field(
        default=None,
        description="说话人ID（如 说话人1），仅启用说话人分离时返回",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "text": "今天天气不错。",
                "start_time": 0.0,
                "end_time": 2.5,
                "speaker_id": "说话人1",
            }
        }
    }


class ASRSuccessResponse(BaseResponse):
    """ASR成功响应模型"""

    result: str = Field(
        ...,
        description="识别结果文本（完整）",
        max_length=100000,
    )

    segments: Optional[List[ASRSegment]] = Field(
        default=None,
        description="分段识别结果（含时间戳），仅长音频分段识别时返回",
    )

    duration: Optional[float] = Field(
        default=None,
        description="音频总时长（秒）",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "task_id": "cf7b0c5339244ee29cd4e43fb97f1234",
                "result": "今天天气不错。明天可能会下雨。",
                "segments": [
                    {"text": "今天天气不错。", "start_time": 0.0, "end_time": 2.5, "speaker_id": "说话人1"},
                    {"text": "明天可能会下雨。", "start_time": 3.2, "end_time": 5.8, "speaker_id": "说话人2"},
                ],
                "duration": 5.8,
                "status": 200,
                "message": "SUCCESS",
            }
        }
    }


class ASRErrorResponse(ErrorResponse):
    """ASR错误响应模型"""

    result: str = Field(default="", description="识别结果（错误时为空）")

    model_config = {
        "json_schema_extra": {
            "example": {
                "task_id": "8bae3613dfc54ebfa811a17d8a7a1234",
                "result": "",
                "status": 40000001,
                "message": "Gateway:ACCESS_DENIED:The token 'invalid_token' is invalid!",
            }
        }
    }


class ASRHealthCheckResponse(HealthCheckResponse):
    """ASR健康检查响应模型"""

    model_config = {
        "protected_namespaces": (),
        "json_schema_extra": {
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "device": "cuda:0",
                "version": "1.0.0",
                "message": "ASR service is running normally",
                "loaded_models": ["paraformer-large", "fun-asr-nano"],
                "memory_usage": {
                    "gpu_memory_used": "2.1GB",
                    "gpu_memory_total": "8.0GB",
                },
                "asr_model_mode": "realtime",
            },
        },
    }

    model_loaded: bool = Field(..., description="模型是否已加载")
    device: str = Field(..., description="推理设备")
    loaded_models: Optional[List[str]] = Field(default=[], description="已加载的模型列表")
    memory_usage: Optional[dict] = Field(default=None, description="内存使用情况")
    asr_model_mode: Optional[str] = Field(default=None, description="当前ASR模型加载模式")


# ============= 模型相关 =============


class ASRModelInfo(BaseModel):
    """新的ASR模型信息模型，支持离线和实时模型分离"""

    id: str = Field(..., description="模型id")
    name: str = Field(..., description="模型名称")
    engine: str = Field(..., description="引擎类型")
    description: str = Field(..., description="模型描述")
    languages: List[str] = Field(..., description="支持的语言列表")
    default: bool = Field(default=False, description="是否为默认模型")
    loaded: bool = Field(default=False, description="是否已加载")
    supports_realtime: bool = Field(default=False, description="是否支持实时识别")
    offline_model: Optional[dict] = Field(default=None, description="离线模型信息")
    realtime_model: Optional[dict] = Field(default=None, description="实时模型信息")
    asr_model_mode: str = Field(..., description="当前ASR模型加载模式")

    model_config = {
        "json_schema_extra": {
            "example": {
                "id": "paraformer-large",
                "name": "Paraformer Large",
                "engine": "funasr",
                "description": "高精度中文语音识别模型",
                "languages": ["zh"],
                "default": True,
                "loaded": True,
                "supports_realtime": True,
                "offline_model": {
                    "path": "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
                    "exists": True,
                },
                "realtime_model": {
                    "path": "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online",
                    "exists": True,
                },
                "asr_model_mode": "realtime",
            }
        }
    }


class ASRModelsResponse(BaseModel):
    """ASR模型列表响应模型"""

    models: List[ASRModelInfo] = Field(..., description="模型列表")
    total: int = Field(..., description="模型总数")
    loaded_count: int = Field(..., description="已加载模型数量")
    asr_model_mode: str = Field(..., description="当前ASR模型加载模式")

    model_config = {
        "json_schema_extra": {
            "example": {
                "models": [
                    {
                        "id": "paraformer-large",
                        "name": "Paraformer Large",
                        "engine": "funasr",
                        "description": "高精度中文语音识别模型",
                        "languages": ["zh"],
                        "default": True,
                        "loaded": True,
                        "supports_realtime": True,
                        "offline_model": {
                            "path": "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
                            "exists": True,
                        },
                        "realtime_model": {
                            "path": "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online",
                            "exists": True,
                        },
                        "asr_model_mode": "realtime",
                    }
                ],
                "total": 3,
                "loaded_count": 1,
                "asr_model_mode": "realtime",
            }
        }
    }


# ============= 联合响应类型 =============

ASRResponse = Union[ASRSuccessResponse, ASRErrorResponse]
