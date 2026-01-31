# -*- coding: utf-8 -*-
"""
WebSocket ASR API路由

合并了 FunASR(阿里云协议) 和 Qwen3 协议的统一 WebSocket ASR 端点。

路由结构：
- GET /ws/v1/asr/test → 统一测试页面
- WS /ws/v1/asr/funasr → FunASR 协议（阿里云兼容）
- WS /ws/v1/asr/qwen → Qwen3-ASR 协议
"""

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, Optional

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from ...core.executor import run_sync
from ...core.exceptions import create_error_response
from ...services.asr.manager import get_model_manager
from ...services.asr.qwen3_engine import Qwen3ASREngine, Qwen3StreamingState
from ...services.websocket_asr import get_aliyun_websocket_asr_service

logger = logging.getLogger(__name__)

# 创建路由器
router = APIRouter(prefix="/ws/v1/asr", tags=["WebSocket ASR"])


# =============================================================================
# 公共工具函数
# =============================================================================


async def _close_ws(websocket: WebSocket):
    """安全关闭 WebSocket 连接"""
    try:
        await websocket.close()
    except Exception:
        pass


def _load_template(filename: str) -> str:
    """加载模板文件内容"""
    path = os.path.join(os.path.dirname(__file__), "..", "..", "templates", filename)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# =============================================================================
# 阿里云端点（极简，调用外部服务）
# =============================================================================


async def _funasr_handler(websocket: WebSocket):
    """FunASR WebSocket 处理逻辑"""
    await websocket.accept()
    service = get_aliyun_websocket_asr_service()
    task_id = f"funasr_ws_{int(time.time())}_{id(websocket)}"

    try:
        await service._process_websocket_connection(websocket, task_id)
    except WebSocketDisconnect:
        logger.info(f"[{task_id}] 客户端断开连接")
    except Exception as e:
        logger.error(f"[{task_id}] 连接处理异常: {e}")
    finally:
        await _close_ws(websocket)


@router.websocket("")
async def funasr_websocket_legacy(websocket: WebSocket):
    """FunASR WebSocket 端点（向后兼容，已弃用，请使用 /funasr）"""
    await _funasr_handler(websocket)


@router.websocket("/funasr")
async def funasr_websocket_endpoint(websocket: WebSocket):
    """FunASR WebSocket 实时 ASR 端点（阿里云协议兼容）"""
    await _funasr_handler(websocket)


# =============================================================================
# Qwen3 相关类
# =============================================================================


class ConnectionState(IntEnum):
    """连接状态"""

    READY = 1
    STARTED = 2
    STREAMING = 3
    COMPLETED = 4


@dataclass
class ConnectionContext:
    """连接上下文，存储每个连接的状态"""

    state: ConnectionState = ConnectionState.READY
    params: Dict[str, Any] = field(default_factory=dict)
    audio_buffer: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.float32)
    )
    streaming_state: Optional[Qwen3StreamingState] = None


class Qwen3WebSocketASRService:
    """Qwen3-ASR WebSocket 流式服务"""

    def __init__(self):
        self.engine: Optional[Qwen3ASREngine] = None

    def _ensure_engine(self) -> Qwen3ASREngine:
        """确保 Qwen3-ASR 流式引擎已加载（使用独立实例）"""
        if self.engine is None:
            model_manager = get_model_manager()

            # 使用默认 Qwen3-ASR 模型，并获取流式专用实例
            from app.core.config import settings
            default_model = model_manager._default_model_id
            # 如果默认模型是 qwen3-asr-0.6b 或 qwen3-asr-1.7b，使用它；否则尝试 1.7b
            if default_model in ["qwen3-asr-0.6b", "qwen3-asr-1.7b"]:
                qwen_model = default_model
            else:
                qwen_model = "qwen3-asr-1.7b"
            logger.info(f"使用 Qwen3-ASR 流式模型: {qwen_model}")
            asr_engine = model_manager.get_asr_engine(qwen_model, streaming=True)

            if not isinstance(asr_engine, Qwen3ASREngine):
                raise Exception("当前模型不是 Qwen3-ASR，无法使用流式识别")

            self.engine = asr_engine

        return self.engine

    async def handle_connection(self, websocket: WebSocket, task_id: str):
        """处理 WebSocket 连接"""
        await websocket.accept()
        logger.info(f"[{task_id}] Qwen3-ASR WebSocket 连接已建立")

        ctx = ConnectionContext()

        try:
            while True:
                message = await websocket.receive()

                if "text" in message:
                    # 处理控制消息
                    data = json.loads(message["text"])
                    msg_type = data.get("type", "")

                    if msg_type == "start":
                        # 开始识别
                        if ctx.state != ConnectionState.READY:
                            await self._send_error(
                                websocket, "识别已在进行中", task_id, error_code="INVALID_STATE"
                            )
                            continue

                        ctx.params = self._parse_start_params(data)
                        ctx.streaming_state = await self._start_recognition(
                            websocket, ctx.params, task_id
                        )
                        ctx.state = ConnectionState.STARTED

                    elif msg_type == "stop":
                        # 停止识别
                        if ctx.state in (ConnectionState.STARTED, ConnectionState.STREAMING):
                            await self._stop_recognition(websocket, ctx, task_id)
                            ctx.state = ConnectionState.COMPLETED
                        break

                    else:
                        await self._send_error(
                            websocket, f"未知消息类型: {msg_type}", task_id, error_code="INVALID_MESSAGE"
                        )

                elif "bytes" in message:
                    # 处理音频数据
                    if ctx.state not in (ConnectionState.STARTED, ConnectionState.STREAMING):
                        await self._send_error(
                            websocket, "请先发送 start 消息", task_id, error_code="INVALID_STATE"
                        )
                        continue

                    audio_bytes = message["bytes"]
                    result = await self._process_audio_chunk(
                        websocket, audio_bytes, ctx, task_id
                    )

                    if result:
                        ctx.state = ConnectionState.STREAMING

        except WebSocketDisconnect:
            logger.info(f"[{task_id}] WebSocket 连接已断开")
        except Exception as e:
            logger.error(f"[{task_id}] 处理连接时出错: {e}")
            await self._send_error(websocket, str(e), task_id, error_code="DEFAULT_SERVER_ERROR")
        finally:
            logger.info(f"[{task_id}] Qwen3-ASR WebSocket 连接已关闭")

    def _parse_start_params(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """解析开始识别参数"""
        payload = data.get("payload", {})

        return {
            "format": payload.get("format", "pcm"),  # pcm, wav
            "sample_rate": payload.get("sample_rate", 16000),
            "language": payload.get("language"),  # None 表示自动检测
            "context": payload.get("context", ""),  # 热词/上下文
            "chunk_size_sec": payload.get("chunk_size_sec", 2.0),
            "unfixed_chunk_num": payload.get("unfixed_chunk_num", 2),
            "unfixed_token_num": payload.get("unfixed_token_num", 5),
        }

    async def _start_recognition(
        self, websocket: WebSocket, params: Dict[str, Any], task_id: str
    ) -> Qwen3StreamingState:
        """初始化流式识别"""
        try:
            engine = self._ensure_engine()

            # 初始化流式状态
            streaming_state = engine.init_streaming_state(
                context=params.get("context", ""),
                language=params.get("language"),
                chunk_size_sec=params.get("chunk_size_sec", 2.0),
                unfixed_chunk_num=params.get("unfixed_chunk_num", 2),
                unfixed_token_num=params.get("unfixed_token_num", 5),
            )

            await websocket.send_json({
                "type": "started",
                "task_id": task_id,
                "params": params,
            })

            logger.info(f"[{task_id}] 流式识别已启动，参数: {params}")
            return streaming_state

        except Exception as e:
            logger.error(f"[{task_id}] 启动识别失败: {e}")
            await self._send_error(websocket, f"启动识别失败: {e}", task_id, error_code="DEFAULT_SERVER_ERROR")
            raise

    async def _process_audio_chunk(
        self,
        websocket: WebSocket,
        audio_bytes: bytes,
        ctx: ConnectionContext,
        task_id: str,
    ) -> bool:
        """处理音频块并返回识别结果"""
        try:
            engine = self._ensure_engine()
            params = ctx.params

            # 转换音频格式
            audio_format = params.get("format", "pcm")
            sample_rate = params.get("sample_rate", 16000)

            if audio_format == "pcm":
                # PCM 16-bit signed int
                audio_array = (
                    np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
                    / 32768.0
                )
            elif audio_format == "wav":
                # 简单处理：跳过 WAV 头（44字节），后续需要更完整的解析
                if len(audio_bytes) > 44:
                    audio_bytes = audio_bytes[44:]
                audio_array = (
                    np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
                    / 32768.0
                )
            else:
                raise ValueError(f"不支持的音频格式: {audio_format}")

            # 重采样到 16kHz（如果不是的话）
            if sample_rate != 16000:
                # 简单线性重采样
                import scipy.signal

                num_samples = int(len(audio_array) * 16000 / sample_rate)
                audio_array = scipy.signal.resample(audio_array, num_samples)

            # 累积音频到 buffer
            ctx.audio_buffer = np.concatenate([ctx.audio_buffer, audio_array])

            # 当 buffer 达到 chunk_size 时，触发识别
            chunk_size_samples = int(params.get("chunk_size_sec", 2.0) * 16000)

            results = []

            # 处理完整的 chunks
            while len(ctx.audio_buffer) >= chunk_size_samples:
                chunk = ctx.audio_buffer[:chunk_size_samples]
                ctx.audio_buffer = ctx.audio_buffer[chunk_size_samples:]

                # 执行流式识别
                ctx.streaming_state = await run_sync(
                    engine.streaming_transcribe,
                    chunk,
                    ctx.streaming_state,
                )

                results.append({
                    "text": ctx.streaming_state.last_text,
                    "language": ctx.streaming_state.last_language,
                    "chunk_id": ctx.streaming_state.chunk_count,
                    "is_partial": True,
                })

            # 发送识别结果
            if results:
                await websocket.send_json({
                    "type": "result",
                    "task_id": task_id,
                    "results": results,
                })

            return len(results) > 0

        except Exception as e:
            logger.error(f"[{task_id}] 处理音频块失败: {e}")
            await self._send_error(websocket, f"处理音频失败: {e}", task_id, error_code="DEFAULT_SERVER_ERROR")
            return False

    async def _stop_recognition(self, websocket: WebSocket, ctx: ConnectionContext, task_id: str):
        """结束流式识别，处理剩余音频"""
        try:
            engine = self._ensure_engine()

            # 处理 buffer 中剩余的音频（如果有）
            if len(ctx.audio_buffer) > 0:
                # 填充到 chunk_size 或保持原样（Qwen3 会处理）
                ctx.streaming_state = await run_sync(
                    engine.streaming_transcribe,
                    ctx.audio_buffer,
                    ctx.streaming_state,
                )
                ctx.audio_buffer = np.array([], dtype=np.float32)

            # 结束识别
            ctx.streaming_state = await run_sync(
                engine.finish_streaming_transcribe,
                ctx.streaming_state,
            )

            # 发送最终结果
            await websocket.send_json({
                "type": "final",
                "task_id": task_id,
                "result": {
                    "text": ctx.streaming_state.last_text,
                    "language": ctx.streaming_state.last_language,
                    "total_chunks": ctx.streaming_state.chunk_count,
                },
            })

            logger.info(f"[{task_id}] 流式识别已完成")

        except Exception as e:
            logger.error(f"[{task_id}] 结束识别失败: {e}")
            await self._send_error(websocket, f"结束识别失败: {e}", task_id, error_code="DEFAULT_SERVER_ERROR")

    async def _send_error(
        self,
        websocket: WebSocket,
        message: str,
        task_id: str,
        error_code: str = "DEFAULT_SERVER_ERROR",
        details: Optional[Dict[str, Any]] = None,
    ):
        """发送错误消息（使用统一错误格式）"""
        try:
            error_response = create_error_response(
                error_code=error_code,
                message=message,
                task_id=task_id,
                details=details,
            )
            # WebSocket 错误消息添加 type 字段以兼容现有协议
            error_response["type"] = "error"
            await websocket.send_json(error_response)
        except Exception:
            pass


# 全局 Qwen3 服务实例
qwen3_service = Qwen3WebSocketASRService()


# =============================================================================
# Qwen3 端点
# =============================================================================


@router.websocket("/qwen")
async def qwen_asr_websocket(websocket: WebSocket, task_id: Optional[str] = None):
    """
    Qwen3-ASR WebSocket 流式识别端点

    连接后流程：
    1. 客户端发送: {"type": "start", "payload": {...}}
    2. 服务端返回: {"type": "started", ...}
    3. 客户端持续发送二进制音频数据（PCM 16kHz 16bit）
    4. 服务端返回: {"type": "result", "results": [...]}
    5. 客户端发送: {"type": "stop"}
    6. 服务端返回: {"type": "final", "result": {...}}

    参数：
    - format: "pcm" 或 "wav"
    - sample_rate: 采样率（默认 16000）
    - language: 强制语言（如 "Chinese"），null 表示自动检测
    - context: 热词/上下文提示
    - chunk_size_sec: 每块音频长度（默认 2.0 秒）
    """
    if task_id is None:
        task_id = str(uuid.uuid4())[:8]

    await qwen3_service.handle_connection(websocket, task_id)


# =============================================================================
# 统一测试页面
# =============================================================================


@router.get("/test", response_class=HTMLResponse)
async def websocket_asr_test_page():
    """WebSocket ASR 统一测试页面"""
    html_content = _load_template("asr_test.html")
    return HTMLResponse(content=html_content)
