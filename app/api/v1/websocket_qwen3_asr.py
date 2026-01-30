# -*- coding: utf-8 -*-
"""
Qwen3-ASR WebSocket 流式识别端点 (POC)

基于 Qwen3-ASR vLLM 后端的流式识别实现。
使用累积重推理机制，支持实时语音识别。

与标准 FunASR WebSocket 协议的区别：
1. 专用于 Qwen3-ASR 模型（vLLM 后端）
2. 简化的协议（非阿里云兼容）
3. 支持语言自动检测

协议格式：
- 连接: ws://host/ws/v1/qwen3/asr
- 发送: JSON 控制消息 或 二进制音频数据
- 接收: JSON 识别结果
"""

import json
import logging
import numpy as np
from typing import Optional, Dict, Any
from enum import IntEnum
from dataclasses import dataclass, field

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from ...core.config import settings
from ...core.executor import run_sync
from ...services.asr.manager import get_model_manager
from ...services.asr.qwen3_engine import Qwen3ASREngine, Qwen3StreamingState

logger = logging.getLogger(__name__)

router = APIRouter()


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
        """确保 Qwen3-ASR 引擎已加载"""
        if self.engine is None:
            model_manager = get_model_manager()
            asr_engine = model_manager.get_asr_engine()

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
                                websocket, "识别已在进行中", task_id
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
                        await self._send_error(websocket, f"未知消息类型: {msg_type}", task_id)

                elif "bytes" in message:
                    # 处理音频数据
                    if ctx.state not in (ConnectionState.STARTED, ConnectionState.STREAMING):
                        await self._send_error(
                            websocket, "请先发送 start 消息", task_id
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
            await self._send_error(websocket, str(e), task_id)
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
            await self._send_error(websocket, f"启动识别失败: {e}", task_id)
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
            await self._send_error(websocket, f"处理音频失败: {e}", task_id)
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
            await self._send_error(websocket, f"结束识别失败: {e}", task_id)

    async def _send_error(self, websocket: WebSocket, message: str, task_id: str):
        """发送错误消息"""
        try:
            await websocket.send_json({
                "type": "error",
                "task_id": task_id,
                "message": message,
            })
        except Exception:
            pass


# 全局服务实例
qwen3_service = Qwen3WebSocketASRService()


@router.websocket("/ws/v1/qwen3/asr")
async def qwen3_asr_websocket(websocket: WebSocket, task_id: Optional[str] = None):
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
    import uuid

    if task_id is None:
        task_id = str(uuid.uuid4())[:8]

    await qwen3_service.handle_connection(websocket, task_id)


@router.get("/ws/v1/qwen3/asr/test", response_class=HTMLResponse)
async def qwen3_asr_test_page():
    """Qwen3-ASR WebSocket 流式识别测试页面"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Qwen3-ASR 流式语音识别测试</title>
        <meta charset="utf-8">
        <style>
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                margin: 0;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
            }
            .container {
                max-width: 1000px;
                margin: 0 auto;
                background: white;
                padding: 30px;
                border-radius: 16px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            }
            .header {
                text-align: center;
                margin-bottom: 30px;
                padding-bottom: 20px;
                border-bottom: 2px solid #eee;
            }
            .header h1 {
                color: #333;
                margin: 0;
                font-size: 28px;
            }
            .header p {
                color: #666;
                margin: 10px 0 0;
            }
            .badge {
                display: inline-block;
                background: #667eea;
                color: white;
                padding: 4px 12px;
                border-radius: 12px;
                font-size: 12px;
                margin-left: 10px;
            }
            .config-panel {
                background: #f8f9fa;
                padding: 20px;
                border-radius: 12px;
                margin-bottom: 20px;
            }
            .form-row {
                display: flex;
                gap: 15px;
                margin-bottom: 15px;
                flex-wrap: wrap;
            }
            .form-group {
                flex: 1;
                min-width: 200px;
            }
            .form-group label {
                display: block;
                margin-bottom: 5px;
                font-weight: 600;
                color: #333;
                font-size: 13px;
            }
            .form-group input,
            .form-group select {
                width: 100%;
                padding: 10px 12px;
                border: 1px solid #ddd;
                border-radius: 8px;
                font-size: 14px;
                box-sizing: border-box;
            }
            .form-group input:focus,
            .form-group select:focus {
                outline: none;
                border-color: #667eea;
                box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
            }
            .controls {
                display: flex;
                justify-content: center;
                gap: 15px;
                margin: 25px 0;
                flex-wrap: wrap;
            }
            button {
                padding: 12px 30px;
                border: none;
                border-radius: 8px;
                font-size: 16px;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s;
                display: flex;
                align-items: center;
                gap: 8px;
            }
            button:disabled {
                opacity: 0.5;
                cursor: not-allowed;
            }
            .btn-primary {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
            }
            .btn-primary:hover:not(:disabled) {
                transform: translateY(-2px);
                box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4);
            }
            .btn-danger {
                background: #dc3545;
                color: white;
            }
            .btn-danger:hover:not(:disabled) {
                background: #c82333;
            }
            .btn-secondary {
                background: #6c757d;
                color: white;
            }
            .btn-secondary:hover:not(:disabled) {
                background: #5a6268;
            }
            .status-indicator {
                display: inline-flex;
                align-items: center;
                gap: 8px;
                padding: 8px 16px;
                border-radius: 20px;
                font-size: 14px;
                font-weight: 500;
            }
            .status-indicator.connected {
                background: #d4edda;
                color: #155724;
            }
            .status-indicator.disconnected {
                background: #f8d7da;
                color: #721c24;
            }
            .status-indicator.recording {
                background: #fff3cd;
                color: #856404;
                animation: pulse 1.5s infinite;
            }
            @keyframes pulse {
                0%, 100% { opacity: 1; }
                50% { opacity: 0.6; }
            }
            .result-panel {
                background: #f8f9fa;
                padding: 20px;
                border-radius: 12px;
                margin-bottom: 20px;
            }
            .result-panel h3 {
                margin: 0 0 15px;
                color: #333;
                font-size: 16px;
            }
            .result-text {
                background: white;
                padding: 20px;
                border-radius: 8px;
                min-height: 120px;
                max-height: 200px;
                overflow-y: auto;
                font-size: 18px;
                line-height: 1.6;
                color: #333;
                border: 1px solid #e0e0e0;
                white-space: pre-wrap;
                word-wrap: break-word;
            }
            .result-text:empty::before {
                content: "识别结果将显示在这里...";
                color: #999;
                font-style: italic;
            }
            .language-tag {
                display: inline-block;
                background: #667eea;
                color: white;
                padding: 4px 12px;
                border-radius: 4px;
                font-size: 12px;
                margin-bottom: 10px;
            }
            .log-panel {
                background: #1e1e1e;
                color: #d4d4d4;
                padding: 15px;
                border-radius: 12px;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 12px;
                max-height: 250px;
                overflow-y: auto;
            }
            .log-panel h3 {
                margin: 0 0 10px;
                color: #fff;
                font-size: 14px;
            }
            .log-entry {
                padding: 3px 0;
                border-bottom: 1px solid #333;
            }
            .log-entry:last-child {
                border-bottom: none;
            }
            .log-entry.info { color: #9cdcfe; }
            .log-entry.success { color: #4ec9b0; }
            .log-entry.error { color: #f48771; }
            .log-entry.warning { color: #dcdcaa; }
            .log-entry.sent { color: #c586c0; }
            .log-entry.received { color: #4fc1ff; }
            .stats {
                display: flex;
                gap: 20px;
                margin: 15px 0;
                flex-wrap: wrap;
            }
            .stat-card {
                background: white;
                padding: 15px 20px;
                border-radius: 8px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                text-align: center;
                flex: 1;
                min-width: 120px;
            }
            .stat-value {
                font-size: 24px;
                font-weight: bold;
                color: #667eea;
            }
            .stat-label {
                font-size: 12px;
                color: #666;
                margin-top: 5px;
            }
            .info-box {
                background: #e3f2fd;
                border-left: 4px solid #2196f3;
                padding: 15px;
                margin-bottom: 20px;
                border-radius: 4px;
            }
            .info-box h4 {
                margin: 0 0 10px;
                color: #1976d2;
            }
            .info-box ul {
                margin: 0;
                padding-left: 20px;
            }
            .info-box li {
                margin: 5px 0;
                color: #424242;
            }
            .mic-icon {
                font-size: 24px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🎙️ Qwen3-ASR 流式语音识别测试</h1>
                <span class="badge">vLLM 后端</span>
                <p>基于累积重推理机制的实时语音识别 | 支持 52+ 种语言自动检测</p>
            </div>

            <div class="info-box">
                <h4>💡 使用说明</h4>
                <ul>
                    <li>本页面用于测试 Qwen3-ASR 的 WebSocket 流式识别功能</li>
                    <li>点击"开始识别"后，对着麦克风说话即可实时看到识别结果</li>
                    <li>支持语言自动检测，也可在下方强制指定语言</li>
                    <li>Chunk Size 越小响应越快，但可能增加边界抖动</li>
                </ul>
            </div>

            <div class="config-panel">
                <div class="form-row">
                    <div class="form-group">
                        <label>WebSocket 服务地址</label>
                        <input type="text" id="wsUrl" value="ws://localhost:8000/ws/v1/qwen3/asr" />
                    </div>
                    <div class="form-group">
                        <label>音频格式</label>
                        <select id="format">
                            <option value="pcm" selected>PCM 16-bit</option>
                        </select>
                    </div>
                </div>
                <div class="form-row">
                    <div class="form-group">
                        <label>强制语言（可选）</label>
                        <select id="language">
                            <option value="" selected>自动检测</option>
                            <option value="Chinese">中文</option>
                            <option value="English">English</option>
                            <option value="Japanese">日本語</option>
                            <option value="Korean">한국어</option>
                            <option value="French">Français</option>
                            <option value="German">Deutsch</option>
                            <option value="Spanish">Español</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>Chunk 大小（秒）</label>
                        <select id="chunkSize">
                            <option value="0.5">0.5s</option>
                            <option value="1.0">1.0s</option>
                            <option value="2.0" selected>2.0s</option>
                            <option value="3.0">3.0s</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>热词/上下文</label>
                        <input type="text" id="context" placeholder="可选：输入热词或上下文提示" />
                    </div>
                </div>
            </div>

            <div class="controls">
                <div id="statusIndicator" class="status-indicator disconnected">
                    <span>●</span> 未连接
                </div>
                <button id="startBtn" class="btn-primary" onclick="startRecognition()">
                    <span class="mic-icon">🎤</span> 开始识别
                </button>
                <button id="stopBtn" class="btn-danger" onclick="stopRecognition()" disabled>
                    🛑 停止识别
                </button>
                <button class="btn-secondary" onclick="clearAll()">🗑️ 清空</button>
            </div>

            <div class="stats">
                <div class="stat-card">
                    <div class="stat-value" id="chunkCount">0</div>
                    <div class="stat-label">处理块数</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="audioDuration">0s</div>
                    <div class="stat-label">音频时长</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="detectedLang">-</div>
                    <div class="stat-label">检测语言</div>
                </div>
            </div>

            <div class="result-panel">
                <h3>📝 识别结果</h3>
                <div id="languageTag" class="language-tag" style="display: none;">自动检测</div>
                <div id="resultText" class="result-text"></div>
            </div>

            <div class="log-panel">
                <h3>📋 运行日志</h3>
                <div id="logContainer"></div>
            </div>
        </div>

        <script>
            let websocket = null;
            let audioContext = null;
            let mediaStream = null;
            let processor = null;
            let audioBuffer = [];
            let isRecording = false;
            let chunkCount = 0;
            let audioDuration = 0;
            let currentText = "";

            function log(message, type = 'info') {
                const container = document.getElementById('logContainer');
                const entry = document.createElement('div');
                entry.className = `log-entry ${type}`;
                const time = new Date().toLocaleTimeString();
                entry.textContent = `[${time}] ${message}`;
                container.appendChild(entry);
                container.scrollTop = container.scrollHeight;
            }

            function updateStatus(status) {
                const indicator = document.getElementById('statusIndicator');
                indicator.className = `status-indicator ${status}`;
                const texts = {
                    connected: '<span>●</span> 已连接',
                    disconnected: '<span>●</span> 未连接',
                    recording: '<span>●</span> 识别中...'
                };
                indicator.innerHTML = texts[status] || status;
            }

            function updateStats() {
                document.getElementById('chunkCount').textContent = chunkCount;
                document.getElementById('audioDuration').textContent = audioDuration.toFixed(1) + 's';
            }

            function updateResult(text, language) {
                document.getElementById('resultText').textContent = text;
                if (language) {
                    document.getElementById('detectedLang').textContent = language;
                    const langTag = document.getElementById('languageTag');
                    langTag.textContent = language;
                    langTag.style.display = 'inline-block';
                }
            }

            async function startRecognition() {
                if (isRecording) return;

                try {
                    // 获取用户选择的配置
                    const wsUrl = document.getElementById('wsUrl').value;
                    const language = document.getElementById('language').value || null;
                    const chunkSize = parseFloat(document.getElementById('chunkSize').value);
                    const context = document.getElementById('context').value;

                    log('正在连接 WebSocket...', 'info');

                    // 连接 WebSocket
                    websocket = new WebSocket(wsUrl);

                    websocket.onopen = async () => {
                        log('WebSocket 连接成功', 'success');
                        updateStatus('connected');

                        // 发送开始识别消息
                        const startMsg = {
                            type: 'start',
                            payload: {
                                format: 'pcm',
                                sample_rate: 16000,
                                language: language,
                                context: context,
                                chunk_size_sec: chunkSize,
                                unfixed_chunk_num: 2,
                                unfixed_token_num: 5
                            }
                        };

                        websocket.send(JSON.stringify(startMsg));
                        log(`发送: ${JSON.stringify(startMsg)}`, 'sent');
                    };

                    websocket.onmessage = (event) => {
                        const data = JSON.parse(event.data);
                        log(`收到: ${JSON.stringify(data).substring(0, 200)}...`, 'received');

                        if (data.type === 'started') {
                            log('识别已启动，开始采集音频', 'success');
                            startAudioCapture();
                        } else if (data.type === 'result') {
                            data.results.forEach(result => {
                                chunkCount = result.chunk_id;
                                updateResult(result.text, result.language);
                            });
                            updateStats();
                        } else if (data.type === 'final') {
                            const result = data.result;
                            updateResult(result.text, result.language);
                            log(`识别完成！总块数: ${result.total_chunks}`, 'success');
                            updateStats();
                        } else if (data.type === 'error') {
                            log(`错误: ${data.message}`, 'error');
                        }
                    };

                    websocket.onerror = (error) => {
                        log('WebSocket 错误', 'error');
                    };

                    websocket.onclose = () => {
                        log('WebSocket 连接已关闭', 'warning');
                        updateStatus('disconnected');
                        stopAudioCapture();
                    };

                } catch (error) {
                    log(`启动失败: ${error.message}`, 'error');
                }
            }

            async function startAudioCapture() {
                try {
                    audioContext = new (window.AudioContext || window.webkitAudioContext)({
                        sampleRate: 16000
                    });

                    mediaStream = await navigator.mediaDevices.getUserMedia({
                        audio: {
                            sampleRate: 16000,
                            channelCount: 1,
                            echoCancellation: true,
                            noiseSuppression: true
                        }
                    });

                    const source = audioContext.createMediaStreamSource(mediaStream);
                    processor = audioContext.createScriptProcessor(4096, 1, 1);

                    source.connect(processor);
                    processor.connect(audioContext.destination);

                    let buffer = [];
                    const chunkSize = parseFloat(document.getElementById('chunkSize').value);
                    const samplesPerChunk = 16000 * chunkSize;

                    processor.onaudioprocess = (e) => {
                        if (!isRecording) return;

                        const inputData = e.inputBuffer.getChannelData(0);
                        buffer.push(...inputData);

                        // 累积足够数据后发送
                        while (buffer.length >= samplesPerChunk) {
                            const chunk = buffer.slice(0, samplesPerChunk);
                            buffer = buffer.slice(samplesPerChunk);

                            // 转换为 16-bit PCM
                            const pcmData = new Int16Array(chunk.length);
                            for (let i = 0; i < chunk.length; i++) {
                                pcmData[i] = Math.max(-1, Math.min(1, chunk[i])) * 0x7FFF;
                            }

                            if (websocket && websocket.readyState === WebSocket.OPEN) {
                                websocket.send(pcmData.buffer);
                                audioDuration += chunkSize;
                                updateStats();
                            }
                        }
                    };

                    isRecording = true;
                    updateStatus('recording');
                    document.getElementById('startBtn').disabled = true;
                    document.getElementById('stopBtn').disabled = false;

                    log('音频采集已启动', 'success');

                } catch (error) {
                    log(`音频采集失败: ${error.message}`, 'error');
                }
            }

            function stopRecognition() {
                if (!isRecording) return;

                log('正在停止识别...', 'info');

                // 发送停止消息
                if (websocket && websocket.readyState === WebSocket.OPEN) {
                    const stopMsg = { type: 'stop' };
                    websocket.send(JSON.stringify(stopMsg));
                    log(`发送: ${JSON.stringify(stopMsg)}`, 'sent');
                }

                stopAudioCapture();

                // 延迟关闭 WebSocket，等待最终响应
                setTimeout(() => {
                    if (websocket) {
                        websocket.close();
                        websocket = null;
                    }
                }, 1000);
            }

            function stopAudioCapture() {
                isRecording = false;

                if (processor) {
                    processor.disconnect();
                    processor = null;
                }

                if (mediaStream) {
                    mediaStream.getTracks().forEach(track => track.stop());
                    mediaStream = null;
                }

                if (audioContext) {
                    audioContext.close();
                    audioContext = null;
                }

                document.getElementById('startBtn').disabled = false;
                document.getElementById('stopBtn').disabled = true;

                log('音频采集已停止', 'info');
            }

            function clearAll() {
                document.getElementById('resultText').textContent = '';
                document.getElementById('logContainer').innerHTML = '';
                document.getElementById('chunkCount').textContent = '0';
                document.getElementById('audioDuration').textContent = '0s';
                document.getElementById('detectedLang').textContent = '-';
                document.getElementById('languageTag').style.display = 'none';
                chunkCount = 0;
                audioDuration = 0;
                currentText = '';
                log('已清空', 'info');
            }

            // 页面加载时记录日志
            window.onload = () => {
                log('Qwen3-ASR 测试页面已加载', 'info');
                log('请确保：1) 服务已启动 2) 已配置 Qwen3-ASR 模型', 'info');
            };

            // 页面关闭时清理
            window.onbeforeunload = () => {
                stopAudioCapture();
                if (websocket) {
                    websocket.close();
                }
            };
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)
