# -*- coding: utf-8 -*-
"""Qwen3-ASR 引擎 - 内嵌 vLLM 后端"""

import logging
from typing import Optional, List, Any
from dataclasses import dataclass

import torch
import numpy as np

from .engines import BaseASREngine, ASRRawResult, ASRSegmentResult, WordToken
from ...core.config import settings
from ...core.exceptions import DefaultServerErrorException

logger = logging.getLogger(__name__)

# 延迟导入
_qwen_asr_module = None

def _get_qwen_model():
    global _qwen_asr_module
    if _qwen_asr_module is None:
        try:
            from qwen_asr import Qwen3ASRModel
            _qwen_asr_module = Qwen3ASRModel
        except ImportError:
            logger.error("qwen-asr 未安装，请运行: pip install qwen-asr[vllm]")
            raise
    return _qwen_asr_module


def _handle_asr_error(operation: str):
    """统一错误处理装饰器"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"{operation} 失败: {e}")
                raise DefaultServerErrorException(f"{operation} 失败: {e}")
        return wrapper
    return decorator


def _get_word_tokens(result, word_level: bool) -> Optional[List[WordToken]]:
    """提取字词级时间戳"""
    if not word_level:
        return None
    ts = getattr(result, "time_stamps", None)
    items = getattr(ts, "items", None)
    if not items:
        return None
    return [
        WordToken(text=item.text, start_time=round(item.start_time, 3), end_time=round(item.end_time, 3))
        for item in items
    ]


@dataclass
class Qwen3StreamingState:
    internal_state: Any
    chunk_count: int = 0
    last_text: str = ""
    last_language: str = ""


class Qwen3ASREngine(BaseASREngine):
    model: Any

    @property
    def supports_realtime(self) -> bool:
        return True

    def __init__(
        self,
        model_path: str = "Qwen/Qwen3-ASR-1.7B",
        device: str = "auto",
        gpu_memory_utilization: Optional[float] = None,
        forced_aligner_path: Optional[str] = None,
        max_inference_batch_size: int = 32,
        max_new_tokens: int = 1024,
        max_model_len: Optional[int] = None,
        **kwargs,
    ):
        Qwen3ASRModel = _get_qwen_model()
        self._device = self._detect_device(device)
        self.model_path = model_path

        # 自动设置显存使用率
        if gpu_memory_utilization is None:
            gpu_memory_utilization = 0.3 if "0.6B" in model_path else 0.4

        fa_kwargs = {"dtype": torch.bfloat16, "device_map": self._device.split(":")[0] if ":" in self._device else "cuda:0"} if forced_aligner_path else None

        logger.info(f"加载 Qwen3-ASR: {model_path}, device={self._device}, gpu={gpu_memory_utilization}")

        llm_kwargs = {
            "model": model_path,
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_inference_batch_size": max_inference_batch_size,
            "max_new_tokens": max_new_tokens,
            **({"forced_aligner": forced_aligner_path, "forced_aligner_kwargs": fa_kwargs} if forced_aligner_path else {}),
            **({"max_model_len": max_model_len} if max_model_len else {}),
        }

        try:
            self.model = Qwen3ASRModel.LLM(**llm_kwargs)
            logger.info("模型加载成功")
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise DefaultServerErrorException(f"模型加载失败: {e}")

    @_handle_asr_error("转写")
    def transcribe_file(
        self,
        audio_path: str,
        hotwords: str = "",
        enable_punctuation: bool = True,
        enable_itn: bool = True,
        enable_vad: bool = False,
        sample_rate: int = 16000,
    ) -> str:
        results = self.model.transcribe(audio=audio_path, context=hotwords or "", return_time_stamps=False)
        return results[0].text if results else ""

    @_handle_asr_error("VAD 转写")
    def transcribe_file_with_vad(
        self,
        audio_path: str,
        hotwords: str = "",
        enable_punctuation: bool = True,
        enable_itn: bool = True,
        sample_rate: int = 16000,
        **kwargs,
    ) -> ASRRawResult:
        word_timestamps = kwargs.get("word_timestamps", True)
        results = self.model.transcribe(audio=audio_path, context=hotwords or "", return_time_stamps=True)
        if not results:
            return ASRRawResult(text="", segments=[])

        result = results[0]
        return ASRRawResult(
            text=result.text or "",
            segments=self._to_segments(result.text, result.time_stamps, word_timestamps)
        )

    def _to_segments(self, text: str, time_stamps: Any, word_level: bool) -> List[ASRSegmentResult]:
        """转换时间戳为分段"""
        items = getattr(getattr(time_stamps, "items", None), "__iter__", lambda: [])()
        items = list(items)

        if not items:
            return [ASRSegmentResult(text=text, start_time=0.0, end_time=0.0)] if text else []

        segments = []
        current, start, words = "", items[0].start_time, []
        breaks = set("。！？；\n")

        for i, item in enumerate(items):
            current += item.text
            words.append(WordToken(item.text, round(item.start_time, 3), round(item.end_time, 3))) if word_level else None

            if item.text in breaks or i == len(items) - 1:
                if current.strip():
                    segments.append(ASRSegmentResult(
                        text=current.strip(),
                        start_time=round(start, 2),
                        end_time=round(item.end_time, 2),
                        word_tokens=words if word_level else None,
                    ))
                current, words = "", []
                if i < len(items) - 1:
                    start = items[i + 1].start_time

        return segments

    @_handle_asr_error("批量推理")
    def _transcribe_batch(
        self,
        segments: List[Any],
        hotwords: str = "",
        enable_punctuation: bool = False,
        enable_itn: bool = False,
        sample_rate: int = 16000,
        word_timestamps: bool = False,
    ) -> List[ASRSegmentResult]:
        valid = [(i, s) for i, s in enumerate(segments) if getattr(s, "temp_file", None)]
        if not valid:
            return [ASRSegmentResult(text="", start_time=0.0, end_time=0.0) for _ in segments]

        indices, segs = zip(*valid)
        results = self.model.transcribe(audio=[s.temp_file for s in segs], context=hotwords or "", return_time_stamps=word_timestamps)

        output = [ASRSegmentResult(text="", start_time=0.0, end_time=0.0) for _ in segments]
        for idx, seg, result in zip(indices, segs, results):
            output[idx] = ASRSegmentResult(
                text=result.text or "",
                start_time=round(seg.start_sec, 2),
                end_time=round(seg.end_sec, 2),
                speaker_id=getattr(seg, "speaker_id", None),
                word_tokens=_get_word_tokens(result, word_timestamps),
            )
        return output

    @_handle_asr_error("初始化流式状态")
    def init_streaming_state(self, context: str = "", language: Optional[str] = None, **kwargs) -> Qwen3StreamingState:
        return Qwen3StreamingState(
            internal_state=self.model.init_streaming_state(context=context, language=language, **kwargs),
            chunk_count=0, last_text="", last_language=""
        )

    @_handle_asr_error("流式识别")
    def streaming_transcribe(self, pcm16k: np.ndarray, state: Qwen3StreamingState) -> Qwen3StreamingState:
        pcm = pcm16k.astype(np.float32) / (32768.0 if pcm16k.dtype == np.int16 else 1.0)
        self.model.streaming_transcribe(pcm, state.internal_state)
        state.chunk_count += 1
        state.last_text = state.internal_state.text
        state.last_language = state.internal_state.language
        return state

    @_handle_asr_error("结束流式识别")
    def finish_streaming_transcribe(self, state: Qwen3StreamingState) -> Qwen3StreamingState:
        self.model.finish_streaming_transcribe(state.internal_state)
        state.last_text = state.internal_state.text
        state.last_language = state.internal_state.language
        return state

    def is_model_loaded(self) -> bool:
        return self.model is not None

    @property
    def device(self) -> str:
        return self._device


def _register_qwen3_engine(register_func, model_config_cls):
    from app.core.config import settings

    def _create(config):
        extra = {k: v for k, v in config.extra_kwargs.items() if v is not None}
        model_id = config.models.get("offline")
        return Qwen3ASREngine(model_path=model_id, device=settings.DEVICE, **extra)

    register_func("qwen3", _create)
