# -*- coding: utf-8 -*-
"""Qwen3-ASR 引擎 - 内嵌 vLLM 后端"""

import os
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


def calculate_gpu_memory_utilization(model_path: str) -> float:
    """Calculate optimal gpu_memory_utilization based on model size and available VRAM

    Model memory requirements (observed):
    - 0.6B: ~6GB (model + initial KV cache)
    - 1.7B: ~12GB (model + initial KV cache)

    Target allocation (with 33% buffer for KV cache growth):
    - 0.6B: 8GB
    - 1.7B: 16GB

    Args:
        model_path: Path to model (used to detect model size)

    Returns:
        gpu_memory_utilization ratio (0.0 to 1.0)
    """
    # Check environment variable override first
    env_override = os.getenv("QWEN_GPU_MEMORY_UTILIZATION")
    if env_override:
        try:
            value = float(env_override)
            if 0.0 < value <= 1.0:
                logger.info(f"Using environment override: gpu_memory_utilization={value}")
                return value
            else:
                logger.warning(f"Invalid QWEN_GPU_MEMORY_UTILIZATION={env_override}, must be 0.0-1.0")
        except ValueError:
            logger.warning(f"Invalid QWEN_GPU_MEMORY_UTILIZATION={env_override}, not a float")

    # Model base memory requirements (GB) - observed values
    MODEL_BASE_MEMORY = {
        "0.6B": 6.0,
        "1.7B": 12.0,
    }

    # Detect model size from path
    if "0.6B" in model_path:
        base_memory_gb = MODEL_BASE_MEMORY["0.6B"]
        model_size = "0.6B"
    else:
        base_memory_gb = MODEL_BASE_MEMORY["1.7B"]
        model_size = "1.7B"

    # Add 33% buffer for KV cache growth
    required_memory_gb = base_memory_gb * 1.33

    # Get total VRAM
    try:
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, using fallback gpu_memory_utilization=0.5")
            return 0.5

        # Use first GPU for memory detection
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)

        # Calculate utilization ratio
        utilization = required_memory_gb / total_vram_gb

        # Clamp to safe maximum (0.95)
        utilization = min(utilization, 0.95)

        logger.info(
            f"GPU memory calculation: model={model_size}, "
            f"requires={required_memory_gb:.1f}GB, total_vram={total_vram_gb:.1f}GB, "
            f"utilization={utilization:.2f}"
        )

        # Warn if VRAM is insufficient
        if utilization >= 0.90:
            logger.warning(
                f"VRAM may be insufficient: {total_vram_gb:.1f}GB available, "
                f"{required_memory_gb:.1f}GB required. Consider using smaller model."
            )

        return round(utilization, 2)

    except Exception as e:
        logger.error(f"Failed to detect VRAM: {e}, using fallback gpu_memory_utilization=0.5")
        return 0.5


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
        forced_aligner_path: Optional[str] = None,
        max_inference_batch_size: int = 32,
        max_new_tokens: int = 1024,
        max_model_len: Optional[int] = None,
        **kwargs,
    ):
        """Initialize Qwen3-ASR engine with dynamic GPU memory allocation

        Args:
            model_path: Path to Qwen3-ASR model
            device: Device to use (auto/cuda/cpu)
            forced_aligner_path: Path to forced aligner model (optional)
            max_inference_batch_size: Maximum batch size for inference
            max_new_tokens: Maximum new tokens for generation (qwen-asr param)
            max_model_len: Maximum model context length (vLLM param)
            **kwargs: Additional arguments (ignored for compatibility)

        Environment Variables:
            QWEN_GPU_MEMORY_UTILIZATION: Override automatic calculation (0.0-1.0)
        """
        Qwen3ASRModel = _get_qwen_model()
        self._device = self._detect_device(device)
        self.model_path = model_path

        # Dynamic GPU memory allocation
        gpu_memory_utilization = calculate_gpu_memory_utilization(model_path)

        # Prepare forced aligner kwargs
        fa_kwargs = None
        if forced_aligner_path:
            fa_kwargs = {
                "dtype": torch.bfloat16,
                "device_map": self._device.split(":")[0] if ":" in self._device else "cuda:0"
            }

        logger.info(
            f"Loading Qwen3-ASR: {model_path}, "
            f"device={self._device}, gpu_memory_utilization={gpu_memory_utilization}"
        )

        # Separate vLLM kwargs from qwen-asr kwargs
        # vLLM kwargs (passed to vllm.LLM)
        vllm_kwargs: dict[str, Any] = {
            "model": model_path,
            "gpu_memory_utilization": gpu_memory_utilization,
        }
        if max_model_len is not None:
            vllm_kwargs["max_model_len"] = max_model_len

        # qwen-asr kwargs (NOT passed to vLLM, handled by qwen-asr library)
        qwen_asr_kwargs: dict[str, Any] = {
            "max_inference_batch_size": max_inference_batch_size,
            "max_new_tokens": max_new_tokens,
        }
        if forced_aligner_path:
            qwen_asr_kwargs["forced_aligner"] = forced_aligner_path
            qwen_asr_kwargs["forced_aligner_kwargs"] = fa_kwargs

        # Merge and pass to qwen-asr (it will internally pass vLLM kwargs to vllm.LLM)
        llm_kwargs = {**vllm_kwargs, **qwen_asr_kwargs}

        try:
            self.model = Qwen3ASRModel.LLM(**llm_kwargs)
            logger.info("Qwen3-ASR model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Qwen3-ASR model: {e}")
            raise DefaultServerErrorException(f"Failed to load Qwen3-ASR model: {e}")

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
