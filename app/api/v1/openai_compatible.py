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
from ...utils.audio import (
    save_audio_to_temp_file,
    cleanup_temp_file,
    get_audio_file_suffix,
    normalize_audio_for_asr,
    get_audio_duration,
)
from ...services.asr.manager import get_model_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["OpenAI Compatible"])


# ============= 响应格式枚举 =============

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
        lines.append(seg.text.strip())
        lines.append("")
    return "\n".join(lines)


def generate_vtt(segments: List[TranscriptionSegment]) -> str:
    """生成 WebVTT 字幕格式"""
    lines = ["WEBVTT", ""]
    for seg in segments:
        start = format_timestamp_vtt(seg.start)
        end = format_timestamp_vtt(seg.end)
        lines.append(f"{start} --> {end}")
        lines.append(seg.text.strip())
        lines.append("")
    return "\n".join(lines)


def map_model_id(model: str) -> Optional[str]:
    """将 OpenAI 模型 ID 映射到 FunASR-API 模型 ID"""
    model_lower = model.lower()

    # 所有 whisper 开头的模型映射到默认模型
    if model_lower.startswith("whisper"):
        return None  # 使用默认模型

    # 其他情况直接使用原模型 ID
    return model


# ============= API 端点 =============

@router.get(
    "/models",
    response_model=ModelsResponse,
    summary="列出可用模型",
    description="返回当前可用的 ASR 模型列表",
)
async def list_models(request: Request):
    """列出可用模型 (OpenAI 兼容)"""
    # 可选鉴权
    result, _ = validate_token(request)
    if not result and settings.APPTOKEN:
        raise HTTPException(status_code=401, detail="Invalid authentication")

    try:
        model_manager = get_model_manager()
        models = model_manager.list_models()

        model_objects = []
        for m in models:
            model_objects.append(ModelObject(
                id=m["id"],
                owned_by="funasr-api",
            ))

        return ModelsResponse(data=model_objects)
    except Exception as e:
        logger.error(f"获取模型列表失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/audio/transcriptions",
    summary="音频转写",
    description="将音频转写为文本 (OpenAI Audio API 兼容)",
    responses={
        200: {
            "description": "转写成功",
            "content": {
                "application/json": {
                    "example": {"text": "转写的文本内容"}
                },
                "text/plain": {
                    "example": "转写的文本内容"
                },
            },
        },
    },
)
async def create_transcription(
    request: Request,
    file: UploadFile = File(..., description="要转写的音频文件"),
    model: str = Form("whisper-1", description="模型 ID"),
    language: Optional[str] = Form(None, description="音频的语言 (ISO-639-1 格式)"),
    prompt: Optional[str] = Form(None, description="可选的提示文本（暂不支持）"),  # noqa: ARG001
    response_format: ResponseFormat = Form(
        ResponseFormat.JSON,
        description="输出格式: json, text, srt, verbose_json, vtt"
    ),
    temperature: Optional[float] = Form(0, description="采样温度（暂不支持）"),  # noqa: ARG001
    timestamp_granularities: Optional[List[str]] = Form(  # noqa: ARG001
        None,
        alias="timestamp_granularities[]",
        description="时间戳粒度（暂不支持）"
    ),
):
    """
    音频转写 API (OpenAI 兼容)

    支持的音频格式: mp3, mp4, mpeg, mpga, m4a, wav, webm, flac, ogg
    最大文件大小: 25MB (OpenAI 限制) / 100MB (FunASR-API 扩展)
    """
    # 标记暂不支持的参数（保留以兼容 OpenAI API）
    _ = (prompt, temperature, timestamp_granularities)

    audio_path = None
    normalized_audio_path = None

    logger.info(f"[OpenAI API] 收到转写请求: model={model}, format={response_format}")

    try:
        # 可选鉴权 (支持 Bearer Token)
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header[7:]
            if settings.APPTOKEN and token != settings.APPTOKEN:
                raise HTTPException(status_code=401, detail="Invalid API key")
        elif settings.APPTOKEN:
            # 如果配置了 APPTOKEN 但请求没有提供
            result, _ = validate_token(request)
            if not result:
                raise HTTPException(status_code=401, detail="Invalid authentication")

        # 读取上传的音频文件
        audio_data = await file.read()
        if not audio_data:
            raise HTTPException(status_code=400, detail="Empty audio file")

        file_size = len(audio_data)
        logger.info(f"[OpenAI API] 音频文件大小: {file_size / 1024 / 1024:.2f}MB")

        # 检查文件大小 (OpenAI 限制 25MB，这里扩展到 100MB)
        if file_size > settings.MAX_AUDIO_SIZE:
            max_mb = settings.MAX_AUDIO_SIZE // 1024 // 1024
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size is {max_mb}MB"
            )

        # 检测音频格式并保存临时文件
        file_suffix = get_audio_file_suffix(
            audio_address=file.filename,
            audio_data=audio_data
        )
        audio_path = save_audio_to_temp_file(audio_data, file_suffix)
        logger.info(f"[OpenAI API] 临时文件: {audio_path}")

        # 标准化音频格式
        normalized_audio_path = normalize_audio_for_asr(audio_path, target_sr=16000)

        # 获取音频时长
        audio_duration = get_audio_duration(normalized_audio_path)
        logger.info(f"[OpenAI API] 音频时长: {audio_duration:.1f}s")

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
        )

        logger.info(f"[OpenAI API] 识别完成: {len(asr_result.text)} 字符")

        # 构建分段信息
        segments = []
        for i, seg in enumerate(asr_result.segments):
            segments.append(TranscriptionSegment(
                id=i,
                seek=int(seg.start_time * 100),
                start=seg.start_time,
                end=seg.end_time,
                text=seg.text,
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
            ).model_dump())

        else:  # JSON (默认)
            return JSONResponse(content={"text": asr_result.text})

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[OpenAI API] 转写失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # 清理临时文件
        if audio_path:
            cleanup_temp_file(audio_path)
        if normalized_audio_path and normalized_audio_path != audio_path:
            cleanup_temp_file(normalized_audio_path)
