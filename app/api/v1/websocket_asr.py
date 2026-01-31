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

    # VAD 和音频管理
    silence_samples: int = 0  # 连续静音样本数
    total_samples: int = 0  # 总音频样本数
    confirmed_segments: list = field(default_factory=list)  # 已确认的识别段落
    segment_index: int = 0  # 当前段落索引

    # 常数配置
    SILENCE_THRESHOLD_SAMPLES: int = 32000  # 2秒 @ 16kHz
    MAX_BUFFER_SAMPLES: int = 960000  # 60秒 @ 16kHz
    VAD_ENERGY_THRESHOLD: float = 0.015  # 能量阈值（RMS）


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

    def _detect_voice_activity(self, audio_array: np.ndarray) -> bool:
        """
        检测音频块是否包含语音（快速能量检测）

        Returns:
            True 表示有语音，False 表示静音
        """
        # 计算 RMS 能量
        rms = np.sqrt(np.mean(audio_array ** 2))
        return rms >= 0.015  # 能量阈值

    def _check_and_reset_if_needed(
        self,
        websocket: WebSocket,
        ctx: ConnectionContext,
        task_id: str,
    ) -> bool:
        """
        检查是否需要截断并重置状态

        Returns:
            True 表示已触发截断，False 表示继续正常处理
        """
        # 检查强制截断条件（60秒上限）
        if ctx.total_samples >= ctx.MAX_BUFFER_SAMPLES:
            logger.info(f"[{task_id}] 音频累积达到 60 秒上限，触发强制截断")
            return True  # 需要截断

        # 检查静音截断条件（连续 2 秒）
        if ctx.silence_samples >= ctx.SILENCE_THRESHOLD_SAMPLES:
            logger.info(f"[{task_id}] 检测到连续 2 秒静音，触发截断")
            return True  # 需要截断

        return False  # 不需要截断

    async def _truncate_and_restart(
        self,
        websocket: WebSocket,
        ctx: ConnectionContext,
        task_id: str,
        reason: str,
    ):
        """
        执行截断并重置识别状态

        流程：
        1. 完成当前段落的识别
        2. 保存已确认文本
        3. 发送 segment_end 事件
        4. 重置音频缓冲区和状态
        5. 初始化新的流式状态
        6. 发送 segment_start 事件
        """
        try:
            engine = self._ensure_engine()

            # 1. 处理剩余音频（如果有）
            if len(ctx.audio_buffer) > 0:
                ctx.streaming_state = await run_sync(
                    engine.streaming_transcribe,
                    ctx.audio_buffer,
                    ctx.streaming_state,
                )

            # 2. 结束当前识别段落
            ctx.streaming_state = await run_sync(
                engine.finish_streaming_transcribe,
                ctx.streaming_state,
            )

            # 3. 保存已确认文本
            segment_text = ctx.streaming_state.last_text or ""
            if segment_text.strip():
                ctx.confirmed_segments.append({
                    "index": ctx.segment_index,
                    "text": segment_text,
                    "language": ctx.streaming_state.last_language,
                    "reason": reason,
                })

            # 4. 发送段落结束事件
            # 构建截断前的完整文本
            confirmed_text = "".join([s["text"] for s in ctx.confirmed_segments])
            full_text_before_cut = confirmed_text + segment_text
            await websocket.send_json({
                "type": "segment_end",
                "task_id": task_id,
                "segment_index": ctx.segment_index,
                "reason": reason,
                "result": {
                    "text": full_text_before_cut,  # 截断前的完整文本
                    "segment_text": segment_text,  # 当前段落文本
                    "language": ctx.streaming_state.last_language,
                },
                "confirmed_texts": [s["text"] for s in ctx.confirmed_segments],
            })

            # 5. 重置状态
            ctx.segment_index += 1
            ctx.audio_buffer = np.array([], dtype=np.float32)
            ctx.silence_samples = 0
            ctx.total_samples = 0

            # 6. 初始化新的流式状态（保留原始配置）
            ctx.streaming_state = engine.init_streaming_state(
                context=ctx.params.get("context", ""),
                language=ctx.params.get("language"),
                chunk_size_sec=ctx.params.get("chunk_size_sec", 2.0),
                unfixed_chunk_num=ctx.params.get("unfixed_chunk_num", 2),
                unfixed_token_num=ctx.params.get("unfixed_token_num", 5),
            )

            # 7. 发送新段落开始事件
            await websocket.send_json({
                "type": "segment_start",
                "task_id": task_id,
                "segment_index": ctx.segment_index,
            })

            logger.info(f"[{task_id}] 截断完成，新段落 {ctx.segment_index} 开始 (原因: {reason})")

        except Exception as e:
            logger.error(f"[{task_id}] 截断操作失败: {e}")
            raise

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
        """处理音频块并返回识别结果（带 VAD 静音检测和自动截断）"""
        try:
            engine = self._ensure_engine()
            params = ctx.params

            # 转换音频格式
            audio_array = self._convert_audio_bytes(audio_bytes, params)
            if audio_array is None:
                return False

            # 更新总样本数
            ctx.total_samples += len(audio_array)

            # VAD 检测 - 更新静音计数器
            has_voice = self._detect_voice_activity(audio_array)
            if has_voice:
                ctx.silence_samples = 0  # 检测到语音，重置静音计数
            else:
                ctx.silence_samples += len(audio_array)  # 累积静音样本

            # 检查是否需要截断（2秒静音 或 60秒上限）
            if self._check_and_reset_if_needed(websocket, ctx, task_id):
                # 执行截断并重启
                await self._truncate_and_restart(websocket, ctx, task_id, reason="silence" if ctx.silence_samples >= ctx.SILENCE_THRESHOLD_SAMPLES else "max_duration")
                # 截断后，将当前音频继续处理（如果还有）
                # 注意：截断后 audio_buffer 已清空，需要将当前 audio_array 加入
                ctx.audio_buffer = np.concatenate([ctx.audio_buffer, audio_array])
            else:
                # 正常累积
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

                # 构建完整文本：已确认段落 + 当前段落
                current_text = ctx.streaming_state.last_text or ""
                confirmed_text = "".join([s["text"] for s in ctx.confirmed_segments])
                full_text = confirmed_text + current_text

                results.append({
                    "text": full_text,  # 返回完整累积文本
                    "current_segment_text": current_text,  # 当前段落原始文本
                    "language": ctx.streaming_state.last_language,
                    "chunk_id": ctx.streaming_state.chunk_count,
                    "is_partial": True,
                    "segment_index": ctx.segment_index,
                })

            # 发送识别结果
            if results:
                await websocket.send_json({
                    "type": "result",
                    "task_id": task_id,
                    "results": results,
                    "segment_index": ctx.segment_index,
                    "confirmed_segments_count": len(ctx.confirmed_segments),
                })

            return len(results) > 0

        except Exception as e:
            logger.error(f"[{task_id}] 处理音频块失败: {e}")
            await self._send_error(websocket, f"处理音频失败: {e}", task_id, error_code="DEFAULT_SERVER_ERROR")
            return False

    def _convert_audio_bytes(self, audio_bytes: bytes, params: Dict[str, Any]) -> Optional[np.ndarray]:
        """转换音频字节为 numpy 数组"""
        try:
            audio_format = params.get("format", "pcm")
            sample_rate = params.get("sample_rate", 16000)

            if audio_format == "pcm":
                # PCM 16-bit signed int
                audio_array = (
                    np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
                    / 32768.0
                )
            elif audio_format == "wav":
                # 简单处理：跳过 WAV 头（44字节）
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
                import scipy.signal
                num_samples = int(len(audio_array) * 16000 / sample_rate)
                audio_array = scipy.signal.resample(audio_array, num_samples)

            return audio_array

        except Exception as e:
            logger.error(f"音频转换失败: {e}")
            return None

    async def _stop_recognition(self, websocket: WebSocket, ctx: ConnectionContext, task_id: str):
        """结束流式识别，处理剩余音频，返回所有段落结果"""
        try:
            engine = self._ensure_engine()

            # 处理 buffer 中剩余的音频（如果有）
            if len(ctx.audio_buffer) > 0:
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

            # 添加最后一段到 confirmed_segments
            final_text = ctx.streaming_state.last_text or ""
            if final_text.strip():
                ctx.confirmed_segments.append({
                    "index": ctx.segment_index,
                    "text": final_text,
                    "language": ctx.streaming_state.last_language,
                    "reason": "final",
                })

            # 构建完整结果文本（直接拼接，不加空格，适配中文）
            all_texts = [s["text"] for s in ctx.confirmed_segments if s["text"].strip()]
            full_text = "".join(all_texts)

            # 发送最终结果（包含所有段落）
            await websocket.send_json({
                "type": "final",
                "task_id": task_id,
                "result": {
                    "text": final_text,  # 最后一段文本
                    "full_text": full_text,  # 所有段落拼接（无空格）
                    "language": ctx.streaming_state.last_language,
                    "total_chunks": ctx.streaming_state.chunk_count,
                    "total_segments": len(ctx.confirmed_segments),
                    "segments": ctx.confirmed_segments,  # 所有段落详情
                },
            })

            logger.info(f"[{task_id}] 流式识别已完成，共 {len(ctx.confirmed_segments)} 个段落")

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
