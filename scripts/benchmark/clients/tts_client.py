# -*- coding: utf-8 -*-
"""
TTS WebSocket 测试客户端

参考 realtime-llm/backend/cti_websocket_handler.py 中的调用方式。
模拟 LLM 流式输出场景，按句子发送文本进行合成。
"""

import asyncio
import json
import time
import logging
import wave
from pathlib import Path
from typing import Optional

from .base_client import BaseWebSocketClient
from ..metrics.models import TTSMetrics

logger = logging.getLogger(__name__)

# 协议常量
TTS_NAMESPACE = "FlowingSpeechSynthesizer"
MSG_START_SYNTHESIS = "StartSynthesis"
MSG_RUN_SYNTHESIS = "RunSynthesis"
MSG_STOP_SYNTHESIS = "StopSynthesis"
MSG_SYNTHESIS_STARTED = "SynthesisStarted"
MSG_SENTENCE_BEGIN = "SentenceBegin"
MSG_SENTENCE_END = "SentenceEnd"
MSG_SYNTHESIS_COMPLETED = "SynthesisCompleted"
MSG_TASK_FAILED = "TaskFailed"


class TTSWebSocketClient(BaseWebSocketClient):
    """TTS WebSocket 测试客户端 (模拟流式文本输入)"""

    def __init__(
        self,
        ws_url: str,
        text: str,
        voice: str = "中文女",
        audio_format: str = "PCM",
        sample_rate: int = 22050,
        timeout: float = 120.0,
        chunk_interval: float = 0.05,  # 发送间隔 (秒)，模拟 LLM 生成速度
        debug: bool = False,  # 调试模式
        save_audio_dir: Optional[Path] = None,  # 保存音频的目录
    ):
        super().__init__(ws_url, timeout)
        self.text = text
        self.voice = voice
        self.audio_format = audio_format
        self.sample_rate = sample_rate
        self.chunk_interval = chunk_interval
        self.debug = debug
        self.save_audio_dir = save_audio_dir
        self._audio_chunks = []  # 存储接收到的音频数据

    def _log(self, msg: str):
        """调试日志"""
        if self.debug:
            logger.info(f"[{self.task_id[:8]}] {msg}")

    async def run_test(self) -> TTSMetrics:
        """执行 TTS 测试"""
        metrics = TTSMetrics(
            request_id=self.task_id,
            concurrency_level=0,
            start_time=time.perf_counter(),
            text_length=len(self.text),
            sample_rate=self.sample_rate,
        )

        try:
            await asyncio.wait_for(
                self._run_tts_session(metrics),
                timeout=self.timeout,
            )
            metrics.success = True
        except asyncio.TimeoutError:
            metrics.error_message = "Timeout"
            logger.warning(f"TTS 请求超时: {self.task_id}")
        except Exception as e:
            metrics.error_message = str(e)
            logger.warning(f"TTS 请求失败: {self.task_id}, 错误: {e}")
        finally:
            await self.close()

        return metrics

    async def _run_tts_session(self, metrics: TTSMetrics) -> None:
        """运行完整的 TTS 会话"""
        self._log(f"连接 {self.ws_url}")
        await self.connect()

        # 用于同步的事件
        started_event = asyncio.Event()
        completed_event = asyncio.Event()
        error_message = None

        # 1. 发送 StartSynthesis
        await self._send_start_synthesis()

        # 2. 启动接收任务
        async def receive_loop():
            nonlocal error_message
            while True:
                try:
                    response = await self.receive()
                except Exception as e:
                    self._log(f"接收异常: {e}")
                    break

                if isinstance(response, bytes):
                    if metrics.first_chunk_time is None:
                        metrics.first_chunk_time = time.perf_counter()
                    metrics.audio_bytes_received += len(response)
                    # 收集音频数据用于保存
                    if self.save_audio_dir:
                        self._audio_chunks.append(response)
                    self._log(f"← 收到音频: {len(response)} bytes")

                elif isinstance(response, str):
                    try:
                        data = json.loads(response)
                        header = data.get("header", {})
                        name = header.get("name", "")
                        status = header.get("status", 0)

                        self._log(f"← 收到事件: {name} (status={status})")

                        if name == MSG_SYNTHESIS_STARTED:
                            started_event.set()

                        elif name == MSG_SENTENCE_END:
                            metrics.sentence_end_time = time.perf_counter()

                        elif name == MSG_SYNTHESIS_COMPLETED:
                            completed_event.set()
                            break

                        elif name == MSG_TASK_FAILED:
                            status_text = header.get("status_text", "Unknown error")
                            error_message = f"TaskFailed: {status_text}"
                            self._log(f"← 错误: {status_text}")
                            completed_event.set()
                            break

                    except json.JSONDecodeError:
                        pass

        receive_task = asyncio.create_task(receive_loop())

        try:
            # 3. 等待 SynthesisStarted
            self._log("等待 SynthesisStarted...")
            await asyncio.wait_for(started_event.wait(), timeout=10.0)
            self._log("收到 SynthesisStarted")

            # 4. 发送文本 - 直接发送完整文本，不分割
            # 参考 CTI 客户端：发送完整句子而不是切分片段
            await self._send_run_synthesis(self.text)

            # 5. 发送 StopSynthesis
            await self._send_stop_synthesis()

            # 6. 等待 SynthesisCompleted
            self._log("等待 SynthesisCompleted...")
            await completed_event.wait()

            if error_message:
                raise Exception(error_message)

            metrics.complete_time = time.perf_counter()
            self._log(f"完成! 收到 {metrics.audio_bytes_received} bytes 音频")

            # 保存音频文件
            if self.save_audio_dir and self._audio_chunks:
                self._save_audio()

        finally:
            if not receive_task.done():
                receive_task.cancel()
                try:
                    await receive_task
                except asyncio.CancelledError:
                    pass

    async def _send_start_synthesis(self) -> None:
        """发送 StartSynthesis 消息"""
        message = {
            "header": self._create_header(MSG_START_SYNTHESIS, TTS_NAMESPACE),
            "payload": {
                "voice": self.voice,
                "format": self.audio_format,
                "sample_rate": self.sample_rate,
                "volume": 50,
                "speech_rate": 0,
                "pitch_rate": 0,
                "platform": "python",
            },
        }
        self._log(f"→ 发送 StartSynthesis (voice={self.voice}, format={self.audio_format})")
        await self.send_json(message)

    async def _send_run_synthesis(self, text: str) -> None:
        """发送 RunSynthesis 消息"""
        message = {
            "header": self._create_header(MSG_RUN_SYNTHESIS, TTS_NAMESPACE),
            "payload": {
                "text": text,
            },
        }
        # 截断显示
        display_text = text[:50] + "..." if len(text) > 50 else text
        self._log(f"→ 发送 RunSynthesis: \"{display_text}\" ({len(text)} chars)")
        await self.send_json(message)

    async def _send_stop_synthesis(self) -> None:
        """发送 StopSynthesis 消息"""
        message = {
            "header": self._create_header(MSG_STOP_SYNTHESIS, TTS_NAMESPACE),
        }
        self._log("→ 发送 StopSynthesis")
        await self.send_json(message)

    def _save_audio(self) -> None:
        """保存收到的音频数据为 WAV 文件"""
        if self.save_audio_dir is None:
            return

        try:
            # 合并所有音频块
            audio_data = b"".join(self._audio_chunks)
            if not audio_data:
                return

            # 生成文件名
            filename = f"{self.task_id[:8]}_{len(self.text)}chars.wav"
            filepath = self.save_audio_dir / filename

            # PCM 数据保存为 WAV
            with wave.open(str(filepath), 'wb') as wav_file:
                wav_file.setnchannels(1)  # 单声道
                wav_file.setsampwidth(2)  # 16位 = 2字节
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(audio_data)

            self._log(f"音频已保存: {filepath}")
        except Exception as e:
            logger.warning(f"保存音频失败: {e}")
