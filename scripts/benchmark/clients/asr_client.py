# -*- coding: utf-8 -*-
"""
ASR WebSocket 测试客户端
"""

import asyncio
import json
import time
import logging
from pathlib import Path
from typing import Optional

from .base_client import BaseWebSocketClient
from ..metrics.models import ASRMetrics

logger = logging.getLogger(__name__)

# 协议常量
ASR_NAMESPACE = "SpeechTranscriber"
MSG_START_TRANSCRIPTION = "StartTranscription"
MSG_STOP_TRANSCRIPTION = "StopTranscription"
MSG_TRANSCRIPTION_STARTED = "TranscriptionStarted"
MSG_TRANSCRIPTION_RESULT_CHANGED = "TranscriptionResultChanged"
MSG_SENTENCE_END = "SentenceEnd"
MSG_TRANSCRIPTION_COMPLETED = "TranscriptionCompleted"
MSG_TASK_FAILED = "TaskFailed"


class ASRWebSocketClient(BaseWebSocketClient):
    """ASR WebSocket 测试客户端"""

    def __init__(
        self,
        ws_url: str,
        audio_data: bytes,
        audio_duration_ms: float,
        sample_rate: int = 16000,
        chunk_size: int = 9600,
        timeout: float = 120.0,
        save_result_dir: Optional[Path] = None,  # 保存识别结果的目录
    ):
        """
        初始化 ASR 客户端

        Args:
            ws_url: WebSocket URL
            audio_data: PCM 音频数据
            audio_duration_ms: 音频时长 (毫秒)
            sample_rate: 采样率
            chunk_size: 每块采样数
            timeout: 超时时间 (秒)
            save_result_dir: 保存识别结果的目录
        """
        super().__init__(ws_url, timeout)
        self.audio_data = audio_data
        self.audio_duration_ms = audio_duration_ms
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.chunk_bytes = chunk_size * 2  # 16-bit PCM
        self.save_result_dir = save_result_dir

    async def run_test(self) -> ASRMetrics:
        """
        执行 ASR 测试

        Returns:
            ASR 测试指标
        """
        metrics = ASRMetrics(
            request_id=self.task_id,
            concurrency_level=0,  # 由调用者设置
            start_time=time.perf_counter(),
            audio_duration_ms=self.audio_duration_ms,
        )

        try:
            await asyncio.wait_for(
                self._run_asr_session(metrics),
                timeout=self.timeout,
            )
            metrics.success = True
        except asyncio.TimeoutError:
            metrics.error_message = "Timeout"
            logger.warning(f"ASR 请求超时: {self.task_id}")
        except Exception as e:
            metrics.error_message = str(e)
            logger.warning(f"ASR 请求失败: {self.task_id}, 错误: {e}")
        finally:
            await self.close()

        return metrics

    async def _run_asr_session(self, metrics: ASRMetrics) -> None:
        """运行完整的 ASR 会话"""
        await self.connect()

        # 1. 发送 StartTranscription
        await self._send_start_transcription()

        # 2. 等待 TranscriptionStarted
        await self.wait_for_message(MSG_TRANSCRIPTION_STARTED)

        # 3. 启动音频流式发送任务
        stream_task = asyncio.create_task(self._stream_audio())

        # 4. 接收识别结果
        try:
            await self._receive_results(metrics)
        finally:
            # 确保流式任务完成
            if not stream_task.done():
                stream_task.cancel()
                try:
                    await stream_task
                except asyncio.CancelledError:
                    pass

    async def _send_start_transcription(self) -> None:
        """发送 StartTranscription 消息"""
        message = {
            "header": self._create_header(MSG_START_TRANSCRIPTION, ASR_NAMESPACE),
            "payload": {
                "format": "pcm",
                "sample_rate": self.sample_rate,
                "enable_intermediate_result": True,
                "enable_punctuation_prediction": True,
                "enable_inverse_text_normalization": True,
                "max_sentence_silence": 800,
            },
        }
        await self.send_json(message)

    async def _stream_audio(self) -> None:
        """流式发送音频数据"""
        offset = 0
        chunk_duration = self.chunk_size / self.sample_rate  # 秒

        while offset < len(self.audio_data):
            chunk = self.audio_data[offset : offset + self.chunk_bytes]
            await self.send_bytes(chunk)
            offset += self.chunk_bytes

            # 以接近实时的速度发送 (稍快一些)
            await asyncio.sleep(chunk_duration * 0.5)

        # 发送 StopTranscription
        await self._send_stop_transcription()

    async def _send_stop_transcription(self) -> None:
        """发送 StopTranscription 消息"""
        message = {
            "header": self._create_header(MSG_STOP_TRANSCRIPTION, ASR_NAMESPACE),
        }
        await self.send_json(message)

    async def _receive_results(self, metrics: ASRMetrics) -> None:
        """接收识别结果"""
        while True:
            response = await self.receive()

            if isinstance(response, str):
                data = json.loads(response)
                header = data.get("header", {})
                name = header.get("name", "")
                payload = data.get("payload", {})

                if name == MSG_TRANSCRIPTION_RESULT_CHANGED:
                    # 记录首次响应时间
                    if metrics.first_result_time is None:
                        metrics.first_result_time = time.perf_counter()

                elif name == MSG_SENTENCE_END:
                    metrics.sentence_end_time = time.perf_counter()
                    # 更新识别结果
                    result = payload.get("result", "")
                    if result:
                        metrics.result_text = result

                elif name == MSG_TRANSCRIPTION_COMPLETED:
                    metrics.complete_time = time.perf_counter()
                    # 保存识别结果
                    if self.save_result_dir and metrics.result_text:
                        self._save_result(metrics.result_text)
                    break

                elif name == MSG_TASK_FAILED:
                    status_text = header.get("status_text", "Unknown error")
                    raise Exception(f"TaskFailed: {status_text}")

    def _save_result(self, result_text: str) -> None:
        """保存识别结果到文本文件"""
        if self.save_result_dir is None:
            return

        try:
            filename = f"{self.task_id[:8]}_{int(self.audio_duration_ms)}ms.txt"
            filepath = self.save_result_dir / filename

            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(result_text)

            logger.debug(f"识别结果已保存: {filepath}")
        except Exception as e:
            logger.warning(f"保存识别结果失败: {e}")
