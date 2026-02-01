# -*- coding: utf-8 -*-
"""
OpenAI 兼容 API
实现 OpenAI Audio API 规范，兼容 OpenAI SDK 和第三方客户端
"""

import time
import logging
from typing import Optional, List
from enum import Enum

from fastapi import APIRouter, File, Form, UploadFile, Request, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field

from ...core.config import settings
from ...core.executor import run_sync
from ...core.security import validate_token
from ...core.exceptions import (
    AuthenticationException,
    InvalidParameterException,
    create_error_response,
)
from ...services.asr.manager import get_model_manager
from ...services.asr.validators import AudioParamsValidator, _get_default_model, _get_dynamic_model_list
from ...services.audio import get_audio_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["OpenAI Compatible"])


# ============= 枚举类型 =============

class ResponseFormat(str, Enum):
    JSON = "json"
    TEXT = "text"
    SRT = "srt"
    VERBOSE_JSON = "verbose_json"
    VTT = "vtt"


# ============= 响应模型 =============

class TranscriptionSegment(BaseModel):
    """转写分段"""
    id: int
    seek: int = 0
    start: float
    end: float
    text: str
    tokens: List[int] = Field(default_factory=list)
    temperature: float = 0.0
    avg_logprob: float = 0.0
    compression_ratio: float = 0.0
    no_speech_prob: float = 0.0
    speaker: Optional[str] = Field(default=None, description="说话人ID")


class TranscriptionWord(BaseModel):
    """转写词级别信息"""
    word: str
    start: float
    end: float


class TranscriptionResponse(BaseModel):
    """简单转写响应 (json 格式)"""
    text: str


class VerboseTranscriptionResponse(BaseModel):
    """详细转写响应 (verbose_json 格式)"""
    task: str = "transcribe"
    language: str
    duration: float
    text: str
    segments: List[TranscriptionSegment] = Field(default_factory=list)
    words: Optional[List[TranscriptionWord]] = None


class ModelObject(BaseModel):
    """模型对象"""
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "funasr-api"


class ModelsResponse(BaseModel):
    """模型列表响应"""
    object: str = "list"
    data: List[ModelObject]


# ============= 辅助函数 =============

def format_timestamp_srt(seconds: float) -> str:
    """格式化时间戳为 SRT 格式 (HH:MM:SS,mmm)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def format_timestamp_vtt(seconds: float) -> str:
    """格式化时间戳为 VTT 格式 (HH:MM:SS.mmm)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def generate_srt(segments: List[TranscriptionSegment]) -> str:
    """生成 SRT 字幕格式"""
    lines = []
    for i, seg in enumerate(segments, 1):
        start = format_timestamp_srt(seg.start)
        end = format_timestamp_srt(seg.end)
        lines.append(f"{i}")
        lines.append(f"{start} --> {end}")
        text = seg.text.strip()
        if seg.speaker:
            text = f"[{seg.speaker}] {text}"
        lines.append(text)
        lines.append("")
    return "\n".join(lines)


def generate_vtt(segments: List[TranscriptionSegment]) -> str:
    """生成 WebVTT 字幕格式"""
    lines = ["WEBVTT", ""]
    for seg in segments:
        start = format_timestamp_vtt(seg.start)
        end = format_timestamp_vtt(seg.end)
        lines.append(f"{start} --> {end}")
        text = seg.text.strip()
        if seg.speaker:
            text = f"[{seg.speaker}] {text}"
        lines.append(text)
        lines.append("")
    return "\n".join(lines)


def map_model_id(model: str) -> Optional[str]:
    """将 OpenAI 模型 ID 映射到 FunASR-API 模型 ID"""
    # whisper-* 映射到默认模型（兼容 OpenAI SDK）
    if model.lower().startswith("whisper"):
        return None  # 使用默认模型

    # 验证模型ID是否受支持
    if model not in AudioParamsValidator.SUPPORTED_MODELS:
        raise InvalidParameterException(
            f"不支持的模型ID: {model}。支持的模型: {', '.join(AudioParamsValidator.SUPPORTED_MODELS)}"
        )

    # 其他情况直接使用原模型 ID
    return model


# ============= API 端点 =============

def _get_openai_model_description() -> str:
    """获取动态的模型描述"""
    available_models = _get_dynamic_model_list()
    default_model = _get_default_model()

    model_descriptions = {
        "qwen3-asr-1.7b": "Qwen3-ASR 1.7B，52 种语言，vLLM 高性能",
        "qwen3-asr-0.6b": "Qwen3-ASR 0.6B，轻量版，适合小显存环境",
        "paraformer-large": "高精度中文 ASR，内置 VAD+标点",
    }

    # 构建表格行
    table_rows = []
    for m in available_models:
        desc = model_descriptions.get(m, "")
        if m == default_model:
            desc += "（默认）"
        table_rows.append(f"| `{m}` | {desc} |")

    return f"""返回当前可用的 ASR 模型列表（OpenAI `/v1/models` 兼容）。

**可用模型：**

| 模型 ID | 说明 |
|---------|------|
{chr(10).join(table_rows)}

**兼容性说明：**
- `whisper-1` 等 OpenAI 模型 ID 会自动映射到默认模型
- 支持 OpenAI SDK 和第三方客户端调用
- 当前默认模型根据显存自动选择：<48GB 用 0.6b，>=48GB 用 1.7b
"""


@router.get(
    "/models",
    response_model=ModelsResponse,
    summary="列出可用模型",
    description=_get_openai_model_description(),
)
async def list_models(request: Request):
    """列出可用模型 (OpenAI 兼容)"""
    # 可选鉴权
    result, _ = validate_token(request)
    if not result and settings.APPTOKEN:
        response_data = create_error_response(
            error_code="AUTHENTICATION_FAILED",
            message="Invalid authentication",
        )
        return JSONResponse(content=response_data, status_code=401)

    try:
        # 使用动态模型列表
        model_ids = _get_dynamic_model_list()

        model_objects = []
        for model_id in model_ids:
            model_objects.append(ModelObject(
                id=model_id,
                owned_by="funasr-api",
            ))

        return ModelsResponse(data=model_objects)
    except Exception as e:
        logger.error(f"获取模型列表失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def update_openapi_schema():
    """在应用启动时更新 OpenAPI schema，使 model 参数显示为下拉框"""
    from fastapi.routing import APIRoute

    available_models = _get_dynamic_model_list()
    default_model = _get_default_model()

    for route in router.routes:
        if isinstance(route, APIRoute) and route.endpoint.__name__ == "create_transcription":
            if not route.openapi_extra:
                route.openapi_extra = {}

            # 定义 model 参数的 schema，添加 enum 使其显示为下拉框
            model_param = {
                "name": "model",
                "in": "formData",
                "required": False,
                "schema": {
                    "type": "string",
                    "default": default_model,
                    "enum": available_models,
                },
                "description": f"ASR 模型 ID。可选值：{', '.join(available_models)}（默认：{default_model}）",
            }

            # 获取或初始化 parameters
            params = route.openapi_extra.get("parameters", [])

            # 移除已有的 model 参数定义（如果存在）
            params = [p for p in params if p.get("name") != "model"]

            # 添加新的 model 参数定义
            params.append(model_param)
            route.openapi_extra["parameters"] = params
            break


def _get_transcription_description() -> str:
    """获取动态的转写端点描述"""
    available_models = _get_dynamic_model_list()
    default_model = _get_default_model()

    model_descriptions = {
        "qwen3-asr-1.7b": "Qwen3-ASR 1.7B，52种语言，vLLM高性能",
        "qwen3-asr-0.6b": "Qwen3-ASR 0.6B，轻量版，适合小显存",
        "paraformer-large": "高精度中文 ASR",
    }

    # 构建模型映射部分
    model_mapping_lines = [f"- `whisper-1` → 使用默认模型 ({default_model})"]
    for m in available_models:
        desc = model_descriptions.get(m, "")
        model_mapping_lines.append(f"- `{m}` → {desc}")

    return f"""将音频文件转写为文本（完全兼容 OpenAI Audio API）。

**支持的音频格式：**
`mp3`, `mp4`, `mpeg`, `mpga`, `m4a`, `wav`, `webm`, `flac`, `ogg`, `amr`, `pcm`

**文件大小限制：**
- 最大支持 {settings.MAX_AUDIO_SIZE // (1024 * 1024)}MB（可通过 `MAX_AUDIO_SIZE` 环境变量配置）
- OpenAI 原生限制为 25MB

**说话人分离：**
- 默认开启 (`enable_speaker_diarization=true`)
- 启用后 `verbose_json` 格式的 segments 会包含 `speaker` 字段（如 "说话人1"）
- 可设置 `enable_speaker_diarization=false` 关闭

**输出格式：**
| 格式 | Content-Type | 说明 |
|------|-------------|------|
| `json` | application/json | 简单 JSON，仅含 text 字段（默认） |
| `text` | text/plain | 纯文本 |
| `verbose_json` | application/json | 详细 JSON，含时间戳、分段和说话人 |
| `srt` | text/plain | SRT 字幕格式 |
| `vtt` | text/vtt | WebVTT 字幕格式 |

**模型映射：**
{chr(10).join(model_mapping_lines)}

**暂不支持的参数：**
`prompt`、`temperature`、`timestamp_granularities` 参数已保留但暂不生效
"""


@router.post(
    "/audio/transcriptions",
    summary="音频转写",
    description=_get_transcription_description(),
    responses={
        200: {
            "description": "转写成功",
            "content": {
                "application/json": {
                    "example": {"text": "今天天气不错，明天可能会下雨。"}
                },
                "text/plain": {
                    "example": "今天天气不错，明天可能会下雨。"
                },
            },
        },
        400: {
            "description": "请求错误",
            "content": {
                "application/json": {
                    "example": {
                        "error_code": "INVALID_PARAMETER",
                        "message": f"File too large. Maximum size is {settings.MAX_AUDIO_SIZE // (1024 * 1024)}MB",
                        "task_id": "",
                        "timestamp": "2025-01-31T12:00:00Z",
                        "details": {}
                    }
                }
            },
        },
        401: {
            "description": "认证失败",
            "content": {
                "application/json": {
                    "example": {
                        "error_code": "AUTHENTICATION_FAILED",
                        "message": "Invalid API key",
                        "task_id": "",
                        "timestamp": "2025-01-31T12:00:00Z",
                        "details": {}
                    }
                }
            },
        },
    },
)
async def create_transcription(
    request: Request,
    # 1. 必需参数 - 输入
    file: UploadFile = File(
        ...,
        description="要转写的音频文件，支持 mp3/wav/flac/ogg/m4a/amr/pcm 等格式"
    ),
    # 2. 核心参数
    model: str = Form(
        default_factory=_get_default_model,
        description="ASR 模型选择。可用模型通过 /v1/models 端点获取",
    ),
    # 3. 音频属性
    language: Optional[str] = Form(
        None,
        description="音频语言代码（ISO-639-1），如 zh/en/ja，不填则自动检测",
        examples=["zh", "en", "ja"],
    ),
    # 4. 功能开关
    enable_speaker_diarization: bool = Form(
        True,
        description="是否启用说话人分离（默认开启）。启用后响应 segments 会包含 speaker 字段"
    ),
    word_timestamps: bool = Form(
        False,
        description="是否返回字词级时间戳（仅 Qwen3-ASR 模型支持）"
    ),
    # 5. 输出选项
    response_format: ResponseFormat = Form(
        ResponseFormat.JSON,
        description="输出格式",
        examples=["json", "text", "verbose_json", "srt", "vtt"],
    ),
    # 6. 兼容性参数（暂不支持）
    prompt: Optional[str] = Form(None, description="提示文本（暂不支持，保留兼容）"),  # noqa: ARG001
    temperature: Optional[float] = Form(0, description="采样温度（暂不支持，保留兼容）"),  # noqa: ARG001
    timestamp_granularities: Optional[List[str]] = Form(  # noqa: ARG001
        None,
        alias="timestamp_granularities[]",
        description="时间戳粒度（暂不支持，保留兼容）"
    ),
):
    """音频转写 API (OpenAI Audio API 兼容)"""
    # 标记暂不支持的参数（保留以兼容 OpenAI API）
    _ = (prompt, temperature, timestamp_granularities)

    audio_path = None
    normalized_audio_path = None

    # 性能计时
    request_start_time = time.time()

    logger.info(f"[OpenAI API] 收到转写请求: model={model}, format={response_format}, "
                f"speaker_diarization={enable_speaker_diarization}, word_level={word_timestamps}")

    # 获取音频处理服务
    audio_service = get_audio_service()

    try:
        # 可选鉴权 (支持 Bearer Token)
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header[7:]
            if settings.APPTOKEN and token != settings.APPTOKEN:
                response_data = create_error_response(
                    error_code="AUTHENTICATION_FAILED",
                    message="Invalid API key",
                )
                return JSONResponse(content=response_data, status_code=401)
        elif settings.APPTOKEN:
            # 如果配置了 APPTOKEN 但请求没有提供
            result, _ = validate_token(request)
            if not result:
                response_data = create_error_response(
                    error_code="AUTHENTICATION_FAILED",
                    message="Invalid authentication",
                )
                return JSONResponse(content=response_data, status_code=401)

        # 读取上传的音频文件
        audio_data = await file.read()

        # 使用音频服务处理上传的音频文件
        normalized_audio_path, audio_duration, audio_path = await audio_service.process_upload_file(
            audio_data=audio_data,
            filename=file.filename,
            sample_rate=16000,
        )

        # 映射模型 ID
        mapped_model_id = map_model_id(model)

        # 获取 ASR 引擎
        model_manager = get_model_manager()
        asr_engine = model_manager.get_asr_engine(mapped_model_id)

        # 执行语音识别
        # 注：prompt 参数接收但不使用，FunASR 热词格式与 OpenAI prompt 不兼容
        asr_result = await run_sync(
            asr_engine.transcribe_long_audio,
            audio_path=normalized_audio_path,
            hotwords="",
            enable_punctuation=True,
            enable_itn=True,
            sample_rate=16000,
            enable_speaker_diarization=enable_speaker_diarization,
            word_timestamps=word_timestamps,
        )

        logger.info(f"[OpenAI API] 识别完成: {len(asr_result.text)} 字符")

        # 构建分段信息
        segments = []
        words = []
        for i, seg in enumerate(asr_result.segments):
            segments.append(TranscriptionSegment(
                id=i,
                seek=int(seg.start_time * 100),
                start=seg.start_time,
                end=seg.end_time,
                text=seg.text,
                speaker=seg.speaker_id,
            ))
            # 收集字词级时间戳（加上 segment 起始时间，转换为全局时间戳）
            if seg.word_tokens:
                for wt in seg.word_tokens:
                    words.append(TranscriptionWord(
                        word=wt.text,
                        start=round(seg.start_time + wt.start_time, 3),
                        end=round(seg.start_time + wt.end_time, 3),
                    ))

        # 检测语言 (简单实现)
        detected_language = language or "zh"
        if not language:
            # 简单的语言检测：检查是否包含中文字符
            import re
            if re.search(r'[\u4e00-\u9fff]', asr_result.text):
                detected_language = "zh"
            else:
                detected_language = "en"

        # 根据 response_format 返回不同格式
        if response_format == ResponseFormat.TEXT:
            return PlainTextResponse(content=asr_result.text)

        elif response_format == ResponseFormat.SRT:
            if not segments:
                # 如果没有分段，创建一个完整的分段
                segments = [TranscriptionSegment(
                    id=0,
                    start=0,
                    end=audio_duration,
                    text=asr_result.text,
                )]
            srt_content = generate_srt(segments)
            return PlainTextResponse(content=srt_content, media_type="text/plain")

        elif response_format == ResponseFormat.VTT:
            if not segments:
                segments = [TranscriptionSegment(
                    id=0,
                    start=0,
                    end=audio_duration,
                    text=asr_result.text,
                )]
            vtt_content = generate_vtt(segments)
            return PlainTextResponse(content=vtt_content, media_type="text/vtt")

        elif response_format == ResponseFormat.VERBOSE_JSON:
            return JSONResponse(content=VerboseTranscriptionResponse(
                task="transcribe",
                language=detected_language,
                duration=audio_duration,
                text=asr_result.text,
                segments=[seg.model_dump() for seg in segments],
                words=[w.model_dump() for w in words] if words else None,
            ).model_dump())

        else:  # JSON (默认)
            return JSONResponse(content={"text": asr_result.text})

    except HTTPException as http_exc:
        # 将 HTTPException 转换为标准错误格式
        logger.error(f"[OpenAI API] HTTP异常: {http_exc.detail}")

        response_data = create_error_response(
            error_code="DEFAULT_CLIENT_ERROR" if http_exc.status_code < 500 else "DEFAULT_SERVER_ERROR",
            message=http_exc.detail,
        )
        return JSONResponse(content=response_data, status_code=http_exc.status_code)
    except Exception as e:
        logger.error(f"[OpenAI API] 转写失败: {e}")

        # 使用标准错误格式
        response_data = create_error_response(
            error_code="DEFAULT_SERVER_ERROR",
            message=str(e),
        )
        return JSONResponse(content=response_data, status_code=500)

    finally:
        # 使用音频服务清理临时文件
        audio_service.cleanup(audio_path, normalized_audio_path)
