# -*- coding: utf-8 -*-
"""WebSocket ASR API - 合并 FunASR 和 Qwen3 协议"""

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from ...core.executor import run_sync
from ...core.exceptions import create_error_response
from ...services.asr.manager import get_model_manager
from ...services.asr.qwen3_engine import Qwen3ASREngine, Qwen3StreamingState
from ...services.websocket_asr import get_aliyun_websocket_asr_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ws/v1/asr", tags=["WebSocket ASR"])


# =============================================================================
# 工具函数
# =============================================================================

async def _close_ws(websocket: WebSocket):
    try:
        await websocket.close()
    except Exception:
        pass


def _load_template(filename: str) -> str:
    path = os.path.join(os.path.dirname(__file__), "..", "..", "templates", filename)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _convert_audio(audio_bytes: bytes, fmt: str, sample_rate: int) -> Optional[np.ndarray]:
    """转换音频字节为 numpy 数组 (16kHz float32)"""
    try:
        if fmt == "wav" and len(audio_bytes) > 44:
            audio_bytes = audio_bytes[44:]  # 跳过 WAV 头

        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        # 重采样到 16kHz
        if sample_rate != 16000:
            import scipy.signal
            num = int(len(audio) * 16000 / sample_rate)
            audio = scipy.signal.resample(audio, num)
            if isinstance(audio, tuple):
                audio = audio[0]

        return audio
    except Exception as e:
        logger.error(f"音频转换失败: {e}")
        return None


# =============================================================================
# FunASR 端点（阿里云协议兼容）
# =============================================================================

async def _funasr_handler(websocket: WebSocket):
    await websocket.accept()
    service = get_aliyun_websocket_asr_service()
    task_id = f"funasr_ws_{int(time.time())}_{id(websocket)}"

    try:
        await service._process_websocket_connection(websocket, task_id)
    except WebSocketDisconnect:
        logger.info(f"[{task_id}] 客户端断开")
    except Exception as e:
        logger.error(f"[{task_id}] 连接异常: {e}")
    finally:
        await _close_ws(websocket)


@router.websocket("")
@router.websocket("/funasr")
async def funasr_websocket(websocket: WebSocket):
    """FunASR WebSocket 端点（向后兼容）"""
    await _funasr_handler(websocket)


# =============================================================================
# Qwen3 流式识别
# =============================================================================

class ConnectionState(IntEnum):
    READY = 1
    STARTED = 2
    STREAMING = 3
    COMPLETED = 4


@dataclass
class ConnectionContext:
    state: ConnectionState = ConnectionState.READY
    params: Dict[str, Any] = field(default_factory=dict)
    audio_buffer: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    streaming_state: Optional[Qwen3StreamingState] = None
    silence_samples: int = 0
    total_samples: int = 0
    confirmed_segments: List = field(default_factory=list)
    segment_index: int = 0

    # 常量
    SILENCE_THRESHOLD = 32000  # 2秒 @ 16kHz
    MAX_BUFFER = 960000        # 60秒 @ 16kHz
    VAD_THRESHOLD = 0.015      # RMS 能量阈值


class Qwen3ASRService:
    def __init__(self):
        self.engine: Optional[Qwen3ASREngine] = None

    def _ensure_engine(self) -> Qwen3ASREngine:
        if self.engine is None:
            manager = get_model_manager()
            default = manager._default_model_id
            model = default if default in ["qwen3-asr-0.6b", "qwen3-asr-1.7b"] else "qwen3-asr-1.7b"
            logger.info(f"使用 Qwen3-ASR 模型: {model}")

            engine = manager.get_asr_engine(model, streaming=True)
            if not isinstance(engine, Qwen3ASREngine):
                raise Exception("当前模型不是 Qwen3-ASR")
            self.engine = engine
        return self.engine

    def _has_voice(self, audio: np.ndarray) -> bool:
        return np.sqrt(np.mean(audio ** 2)) >= 0.015

    def _need_truncate(self, ctx: ConnectionContext) -> bool:
        return ctx.total_samples >= ctx.MAX_BUFFER or ctx.silence_samples >= ctx.SILENCE_THRESHOLD

    async def _truncate(self, websocket: WebSocket, ctx: ConnectionContext, task_id: str, reason: str):
        """执行截断并重置状态"""
        try:
            engine = self._ensure_engine()

            # 处理剩余音频
            if len(ctx.audio_buffer) > 0:
                ctx.streaming_state = await run_sync(
                    engine.streaming_transcribe, ctx.audio_buffer, ctx.streaming_state
                )

            ctx.streaming_state = await run_sync(
                engine.finish_streaming_transcribe, ctx.streaming_state
            )

            segment_text = ctx.streaming_state.last_text or ""
            is_valid = len(segment_text.strip()) >= 3  # 过滤短语气词

            if is_valid:
                ctx.confirmed_segments.append({
                    "index": ctx.segment_index,
                    "text": segment_text,
                    "language": ctx.streaming_state.last_language,
                    "reason": reason,
                })

                confirmed = "\n".join([s["text"] for s in ctx.confirmed_segments])
                full_text = confirmed + "\n" + segment_text if confirmed else segment_text

                await websocket.send_json({
                    "type": "segment_end",
                    "task_id": task_id,
                    "segment_index": ctx.segment_index,
                    "reason": reason,
                    "result": {
                        "text": full_text,
                        "segment_text": segment_text,
                        "language": ctx.streaming_state.last_language,
                    },
                    "confirmed_texts": [s["text"] for s in ctx.confirmed_segments],
                })

                ctx.segment_index += 1
                logger.info(f"[{task_id}] 截断完成，新段落 {ctx.segment_index}")
            else:
                logger.debug(f"[{task_id}] 段落太短已过滤: '{segment_text}'")

            # 重置状态
            ctx.audio_buffer = np.array([], dtype=np.float32)
            ctx.silence_samples = 0
            ctx.total_samples = 0
            ctx.streaming_state = engine.init_streaming_state(
                context=ctx.params.get("context", ""),
                language=ctx.params.get("language"),
                chunk_size_sec=ctx.params.get("chunk_size_sec", 2.0),
                unfixed_chunk_num=ctx.params.get("unfixed_chunk_num", 2),
                unfixed_token_num=ctx.params.get("unfixed_token_num", 5),
            )

            if is_valid:
                await websocket.send_json({
                    "type": "segment_start",
                    "task_id": task_id,
                    "segment_index": ctx.segment_index,
                })

        except Exception as e:
            logger.error(f"[{task_id}] 截断失败: {e}")
            raise

    async def _send_error(self, websocket: WebSocket, message: str, task_id: str, code: str = "DEFAULT_SERVER_ERROR"):
        try:
            error = create_error_response(error_code=code, message=message, task_id=task_id)
            error["type"] = "error"
            await websocket.send_json(error)
        except Exception:
            pass

    async def handle_connection(self, websocket: WebSocket, task_id: str):
        await websocket.accept()
        logger.info(f"[{task_id}] Qwen3 WebSocket 连接已建立")

        ctx = ConnectionContext()

        try:
            while True:
                message = await websocket.receive()

                if "text" in message:
                    data = json.loads(message["text"])
                    msg_type = data.get("type", "")

                    if msg_type == "start":
                        if ctx.state != ConnectionState.READY:
                            await self._send_error(websocket, "识别已在进行中", task_id, "INVALID_STATE")
                            continue

                        payload = data.get("payload", {})
                        ctx.params = {
                            "format": payload.get("format", "pcm"),
                            "sample_rate": payload.get("sample_rate", 16000),
                            "language": payload.get("language"),
                            "context": payload.get("context", ""),
                            "chunk_size_sec": payload.get("chunk_size_sec", 2.0),
                            "unfixed_chunk_num": payload.get("unfixed_chunk_num", 2),
                            "unfixed_token_num": payload.get("unfixed_token_num", 5),
                        }

                        engine = self._ensure_engine()
                        ctx.streaming_state = engine.init_streaming_state(
                            context=ctx.params["context"],
                            language=ctx.params["language"],
                            chunk_size_sec=ctx.params["chunk_size_sec"],
                            unfixed_chunk_num=ctx.params["unfixed_chunk_num"],
                            unfixed_token_num=ctx.params["unfixed_token_num"],
                        )

                        await websocket.send_json({"type": "started", "task_id": task_id, "params": ctx.params})
                        ctx.state = ConnectionState.STARTED
                        logger.info(f"[{task_id}] 识别已启动: {ctx.params}")

                    elif msg_type == "stop":
                        if ctx.state in (ConnectionState.STARTED, ConnectionState.STREAMING):
                            await self._stop(websocket, ctx, task_id)
                        break

                    else:
                        await self._send_error(websocket, f"未知消息类型: {msg_type}", task_id, "INVALID_MESSAGE")

                elif "bytes" in message:
                    if ctx.state not in (ConnectionState.STARTED, ConnectionState.STREAMING):
                        await self._send_error(websocket, "请先发送 start", task_id, "INVALID_STATE")
                        continue

                    audio = _convert_audio(message["bytes"], ctx.params["format"], ctx.params["sample_rate"])
                    if audio is None:
                        continue

                    ctx.total_samples += len(audio)

                    # VAD 检测
                    if self._has_voice(audio):
                        ctx.silence_samples = 0
                    else:
                        ctx.silence_samples += len(audio)

                    # 检查截断
                    if self._need_truncate(ctx):
                        reason = "silence" if ctx.silence_samples >= ctx.SILENCE_THRESHOLD else "max_duration"
                        await self._truncate(websocket, ctx, task_id, reason)
                        ctx.audio_buffer = np.concatenate([ctx.audio_buffer, audio])
                    else:
                        ctx.audio_buffer = np.concatenate([ctx.audio_buffer, audio])

                    # 处理完整 chunks
                    chunk_size = int(ctx.params["chunk_size_sec"] * 16000)
                    results = []

                    while len(ctx.audio_buffer) >= chunk_size:
                        chunk = ctx.audio_buffer[:chunk_size]
                        ctx.audio_buffer = ctx.audio_buffer[chunk_size:]

                        engine = self._ensure_engine()
                        ctx.streaming_state = await run_sync(
                            engine.streaming_transcribe, chunk, ctx.streaming_state
                        )

                        current = ctx.streaming_state.last_text or ""
                        confirmed = "\n".join([s["text"] for s in ctx.confirmed_segments])
                        full = confirmed + "\n" + current if confirmed else current

                        results.append({
                            "text": full,
                            "current_segment_text": current,
                            "language": ctx.streaming_state.last_language,
                            "chunk_id": ctx.streaming_state.chunk_count,
                            "is_partial": True,
                            "segment_index": ctx.segment_index,
                        })

                    if results:
                        await websocket.send_json({
                            "type": "result",
                            "task_id": task_id,
                            "results": results,
                            "segment_index": ctx.segment_index,
                            "confirmed_segments_count": len(ctx.confirmed_segments),
                        })
                        ctx.state = ConnectionState.STREAMING

        except WebSocketDisconnect:
            logger.info(f"[{task_id}] WebSocket 断开")
        except Exception as e:
            logger.error(f"[{task_id}] 连接错误: {e}")
            await self._send_error(websocket, str(e), task_id)
        finally:
            logger.info(f"[{task_id}] 连接已关闭")

    async def _stop(self, websocket: WebSocket, ctx: ConnectionContext, task_id: str):
        try:
            engine = self._ensure_engine()

            if len(ctx.audio_buffer) > 0:
                ctx.streaming_state = await run_sync(
                    engine.streaming_transcribe, ctx.audio_buffer, ctx.streaming_state
                )

            ctx.streaming_state = await run_sync(
                engine.finish_streaming_transcribe, ctx.streaming_state
            )

            final = ctx.streaming_state.last_text or ""
            if final.strip():
                ctx.confirmed_segments.append({
                    "index": ctx.segment_index,
                    "text": final,
                    "language": ctx.streaming_state.last_language,
                    "reason": "final",
                })

            all_texts = [s["text"] for s in ctx.confirmed_segments if s["text"].strip()]
            full_text = "\n".join(all_texts)

            await websocket.send_json({
                "type": "final",
                "task_id": task_id,
                "result": {
                    "text": final,
                    "full_text": full_text,
                    "language": ctx.streaming_state.last_language,
                    "total_chunks": ctx.streaming_state.chunk_count,
                    "total_segments": len(ctx.confirmed_segments),
                    "segments": ctx.confirmed_segments,
                },
            })

            logger.info(f"[{task_id}] 识别完成，共 {len(ctx.confirmed_segments)} 段落")

        except Exception as e:
            logger.error(f"[{task_id}] 结束识别失败: {e}")
            await self._send_error(websocket, f"结束识别失败: {e}", task_id)


# 全局服务实例
_qwen3_service = Qwen3ASRService()


@router.websocket("/qwen")
async def qwen_asr_websocket(websocket: WebSocket, task_id: Optional[str] = None):
    """Qwen3-ASR WebSocket 流式识别端点"""
    if task_id is None:
        task_id = str(uuid.uuid4())[:8]
    await _qwen3_service.handle_connection(websocket, task_id)


# =============================================================================
# 测试页面
# =============================================================================


@router.get("/test", response_class=HTMLResponse)
async def websocket_asr_test_page():
    """WebSocket ASR 测试页面"""
    return HTMLResponse(content=_load_template("asr_test.html"))
