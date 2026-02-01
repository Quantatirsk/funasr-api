# -*- coding: utf-8 -*-
"""
Qwen3-ASR 引擎 - 内嵌 vLLM 后端

支持 Qwen3-ASR-1.7B 模型
通过 Qwen3ASRModel.LLM() 直接初始化 vLLM
"""

import logging
import os
from typing import Optional, List, Any, Tuple, Dict, TYPE_CHECKING
from dataclasses import dataclass

import torch
import numpy as np

from .engines import BaseASREngine, ASRRawResult, ASRSegmentResult, ASRFullResult, WordToken
from ...core.config import settings
from ...core.exceptions import DefaultServerErrorException
from ...utils.audio import get_audio_duration

if TYPE_CHECKING:
    from qwen_asr import Qwen3ASRModel


@dataclass
class Qwen3StreamingState:
    """Qwen3-ASR 流式状态包装器"""

    internal_state: Any  # Qwen3ASRModel.ASRStreamingState
    chunk_count: int = 0
    last_text: str = ""
    last_language: str = ""

logger = logging.getLogger(__name__)

# 延迟导入 qwen_asr，避免启动时加载
_qwen_asr_module = None

def _get_qwen_asr_model():
    global _qwen_asr_module
    if _qwen_asr_module is None:
        try:
            from qwen_asr import Qwen3ASRModel
            _qwen_asr_module = Qwen3ASRModel
        except ImportError:
            _qwen_asr_module = None
            logger.warning("qwen-asr 未安装，Qwen3-ASR 功能不可用")
    return _qwen_asr_module


class Qwen3ASREngine(BaseASREngine):
    """
    Qwen3-ASR 语音识别引擎

    使用 vLLM 后端进行高效推理，内置 VAD、标点、语言识别
    支持字级时间戳（通过 Qwen3-ForcedAligner）
    """

    model: "Any"

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
        """
        初始化 Qwen3-ASR 引擎

        Args:
            model_path: 模型路径或 HuggingFace/ModelScope ID
            device: 推理设备（auto/cuda:0/cpu）
            gpu_memory_utilization: GPU 显存使用率（0-1），None表示自动根据模型大小设置
            forced_aligner_path: 时间戳对齐模型路径（可选）
            max_inference_batch_size: 最大批处理大小
            max_new_tokens: 最大生成 token 数
            max_model_len: 最大模型序列长度（用于限制 KV 缓存）
        """
        Qwen3ASRModel = _get_qwen_asr_model()
        if Qwen3ASRModel is None:
            raise ImportError(
                "qwen-asr 未安装，请运行: pip install qwen-asr[vllm]"
            )

        self.model_path = model_path
        self._device = self._detect_device(device)

        # 根据模型大小自动设置显存使用率（如果未显式指定）
        if gpu_memory_utilization is None:
            if "0.6B" in model_path:
                gpu_memory_utilization = 0.3
                logger.info("检测到 0.6B 模型，自动设置显存使用率为 0.3")
            elif "1.7B" in model_path:
                gpu_memory_utilization = 0.4
                logger.info("检测到 1.7B 模型，自动设置显存使用率为 0.4")
            else:
                gpu_memory_utilization = 0.4  # 默认
                logger.info(f"未识别模型大小，默认显存使用率 0.4 (路径: {model_path})")

        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_inference_batch_size = max_inference_batch_size
        self.max_new_tokens = max_new_tokens

        # 构建 forced_aligner_kwargs（仅在需要时）
        forced_aligner_kwargs = None
        if forced_aligner_path:
            forced_aligner_kwargs = {
                "dtype": torch.bfloat16,
                "device_map": self._device if ":" in self._device else "cuda:0",
            }
            logger.info(f"启用 ForcedAligner: {forced_aligner_path}")

        logger.info(f"正在加载 Qwen3-ASR 模型: {model_path}")
        logger.info(f"设备: {self._device}, GPU 显存使用率: {gpu_memory_utilization}")
        if max_model_len:
            logger.info(f"最大序列长度限制: {max_model_len}")

        try:
            # 构建 vLLM 初始化参数
            llm_kwargs = {
                "model": model_path,
                "gpu_memory_utilization": gpu_memory_utilization,
                "max_inference_batch_size": max_inference_batch_size,
                "max_new_tokens": max_new_tokens,
            }
            # 只有当 forced_aligner_path 存在时才添加
            if forced_aligner_path:
                llm_kwargs["forced_aligner"] = forced_aligner_path
                llm_kwargs["forced_aligner_kwargs"] = forced_aligner_kwargs
            # 添加 max_model_len（如果指定）
            if max_model_len:
                llm_kwargs["max_model_len"] = max_model_len

            # 使用 vLLM 后端初始化
            self.model = Qwen3ASRModel.LLM(**llm_kwargs)
            logger.info(f"Qwen3-ASR 模型加载成功: {model_path}")
        except Exception as e:
            logger.error(f"Qwen3-ASR 模型加载失败: {e}")
            raise DefaultServerErrorException(f"模型加载失败: {e}")

    def transcribe_file(
        self,
        audio_path: str,
        hotwords: str = "",
        enable_punctuation: bool = True,  # Qwen3 内置，此参数忽略
        enable_itn: bool = True,
        enable_vad: bool = False,  # Qwen3 内置 VAD
        sample_rate: int = 16000,
        language: Optional[str] = None,
    ) -> str:
        """
        转录音频文件

        Args:
            audio_path: 音频文件路径
            hotwords: 热词/上下文提示
            enable_punctuation: 是否启用标点（Qwen3 始终启用）
            enable_itn: 是否启用 ITN（未实现，Qwen3 内置）
            enable_vad: 是否启用 VAD（Qwen3 内置）
            sample_rate: 采样率
            language: 强制语言（如 "Chinese", "English"），None 表示自动检测

        Returns:
            识别文本
        """
        try:
            results = self.model.transcribe(
                audio=audio_path,
                context=hotwords if hotwords else "",
                language=language,
                return_time_stamps=False,
            )

            if not results:
                return ""

            return results[0].text or ""

        except Exception as e:
            logger.error(f"Qwen3-ASR 转写失败: {e}")
            raise DefaultServerErrorException(f"语音识别失败: {e}")

    def transcribe_file_with_vad(
        self,
        audio_path: str,
        hotwords: str = "",
        enable_punctuation: bool = True,
        enable_itn: bool = True,
        sample_rate: int = 16000,
        word_timestamps: bool = True,
        **kwargs,
    ) -> ASRRawResult:
        """
        使用 VAD 转录音频文件，返回带时间戳分段的结果

        Args:
            audio_path: 音频文件路径
            hotwords: 热词/上下文提示
            enable_punctuation: 是否启用标点
            enable_itn: 是否启用 ITN
            sample_rate: 采样率
            word_timestamps: 是否返回字词级时间戳（默认 True）
            **kwargs: 额外参数（兼容基类）

        Returns:
            ASRRawResult 包含文本和分段信息，每个 segment 包含 word_tokens
        """
        _ = kwargs  # 忽略额外参数
        try:
            results = self.model.transcribe(
                audio=audio_path,
                context=hotwords if hotwords else "",
                return_time_stamps=True,  # 启用时间戳
            )

            if not results:
                return ASRRawResult(text="", segments=[])

            result = results[0]
            segments = self._convert_timestamps_to_segments(
                result.text, result.time_stamps, word_level=word_timestamps
            )

            return ASRRawResult(text=result.text or "", segments=segments)

        except Exception as e:
            logger.error(f"Qwen3-ASR VAD 转写失败: {e}")
            raise DefaultServerErrorException(f"语音识别失败: {e}")

    def _convert_timestamps_to_segments(
        self, text: str, time_stamps: Any, word_level: bool = True
    ) -> List[ASRSegmentResult]:
        """
        将 Qwen3-ASR 时间戳转换为内部分段格式

        Args:
            text: 完整识别文本
            time_stamps: Qwen3-ForcedAligner 返回的时间戳对象
            word_level: 是否包含字词级时间戳（默认 True）

        Returns:
            ASRSegmentResult 列表
        """
        segments = []

        if not time_stamps or not hasattr(time_stamps, "items") or not time_stamps.items:
            # 无时间戳，返回整段
            if text:
                segments.append(
                    ASRSegmentResult(text=text, start_time=0.0, end_time=0.0)
                )
            return segments

        items = time_stamps.items
        sentence_breaks = "。！？；\n"  # 句子分隔符

        current_text = ""
        current_start = items[0].start_time if items else 0.0
        current_end = 0.0
        current_word_tokens: List[WordToken] = []

        for i, item in enumerate(items):
            char = item.text
            current_text += char
            current_end = item.end_time

            # 收集字词级时间戳
            if word_level:
                current_word_tokens.append(
                    WordToken(
                        text=char,
                        start_time=round(item.start_time, 3),
                        end_time=round(item.end_time, 3),
                    )
                )

            # 遇到句子分隔符或最后一个字，保存当前句子
            if char in sentence_breaks or i == len(items) - 1:
                if current_text.strip():
                    segment = ASRSegmentResult(
                        text=current_text.strip(),
                        start_time=round(current_start, 2),
                        end_time=round(current_end, 2),
                        word_tokens=current_word_tokens if word_level else None,
                    )
                    segments.append(segment)
                # 重置
                current_text = ""
                current_word_tokens = []
                if i < len(items) - 1:
                    current_start = items[i + 1].start_time

        return segments

    def _transcribe_batch(
        self,
        segments: List[Any],
        hotwords: str = "",
        enable_punctuation: bool = False,
        enable_itn: bool = False,
        sample_rate: int = 16000,
        word_timestamps: bool = False,
    ) -> List[ASRSegmentResult]:
        """
        批量转写多个音频片段，支持字词级时间戳

        利用 Qwen3-ASR 的原生批处理能力

        Args:
            segments: 音频片段列表（需有 temp_file 属性）
            hotwords: 热词
            enable_punctuation: 是否启用标点
            enable_itn: 是否启用 ITN
            sample_rate: 采样率
            word_timestamps: 是否返回字词级时间戳

        Returns:
            ASRSegmentResult 列表
        """
        # 过滤有效片段
        valid_paths = []
        valid_segments = []
        valid_indices = []

        for idx, seg in enumerate(segments):
            if hasattr(seg, "temp_file") and seg.temp_file:
                valid_paths.append(seg.temp_file)
                valid_segments.append(seg)
                valid_indices.append(idx)

        if not valid_paths:
            return [
                ASRSegmentResult(text="", start_time=0.0, end_time=0.0)
                for _ in segments
            ]

        try:
            logger.info(
                f"Qwen3-ASR 批量推理: {len(valid_paths)} 个片段, word_timestamps={word_timestamps}"
            )

            # 使用 Qwen3-ASR 原生批处理
            results = self.model.transcribe(
                audio=valid_paths,
                context=hotwords if hotwords else "",
                return_time_stamps=word_timestamps,
            )

            # 组装结果
            output: List[ASRSegmentResult] = [
                ASRSegmentResult(text="", start_time=0.0, end_time=0.0)
                for _ in segments
            ]

            for idx, seg, result in zip(valid_indices, valid_segments, results):
                # 转换字词级时间戳
                word_tokens = None
                if (
                    word_timestamps
                    and hasattr(result, "time_stamps")
                    and result.time_stamps
                    and hasattr(result.time_stamps, "items")
                    and result.time_stamps.items
                ):
                    word_tokens = [
                        WordToken(
                            text=item.text,
                            start_time=round(item.start_time, 3),
                            end_time=round(item.end_time, 3),
                        )
                        for item in result.time_stamps.items
                    ]

                output[idx] = ASRSegmentResult(
                    text=result.text or "",
                    start_time=round(seg.start_sec, 2),
                    end_time=round(seg.end_sec, 2),
                    speaker_id=getattr(seg, "speaker_id", None),
                    word_tokens=word_tokens,
                )

            logger.info(f"Qwen3-ASR 批量推理完成: {len(valid_paths)}/{len(segments)}")
            return output

        except Exception as e:
            logger.error(f"Qwen3-ASR 批量推理失败: {e}，降级到单条推理")
            # 降级：逐个推理
            return super()._transcribe_batch(
                segments, hotwords, enable_punctuation, enable_itn, sample_rate, word_timestamps
            )

    def is_model_loaded(self) -> bool:
        """检查模型是否已加载"""
        return self.model is not None

    @property
    def device(self) -> str:
        """获取设备信息"""
        return self._device

    @property
    def supports_realtime(self) -> bool:
        """
        是否支持实时识别

        Qwen3-ASR vLLM 后端支持流式识别（通过累积重推理机制）
        """
        return True

    def init_streaming_state(
        self,
        context: str = "",
        language: Optional[str] = None,
        unfixed_chunk_num: int = 2,
        unfixed_token_num: int = 5,
        chunk_size_sec: float = 2.0,
    ) -> Qwen3StreamingState:
        """
        初始化流式识别状态

        Args:
            context: 上下文/热词提示
            language: 强制语言（如 "Chinese", "English"），None 表示自动检测
            unfixed_chunk_num: 前 N 个 chunk 不使用 prefix 提示
            unfixed_token_num: 回滚 token 数，用于减少边界抖动
            chunk_size_sec: 每个 chunk 的音频长度（秒）

        Returns:
            Qwen3StreamingState: 流式状态对象
        """
        if self.model is None:
            raise DefaultServerErrorException("模型未加载")

        try:
            internal_state = self.model.init_streaming_state(
                context=context,
                language=language,
                unfixed_chunk_num=unfixed_chunk_num,
                unfixed_token_num=unfixed_token_num,
                chunk_size_sec=chunk_size_sec,
            )
            return Qwen3StreamingState(
                internal_state=internal_state,
                chunk_count=0,
                last_text="",
                last_language="",
            )
        except Exception as e:
            logger.error(f"初始化流式状态失败: {e}")
            raise DefaultServerErrorException(f"初始化流式状态失败: {e}")

    def streaming_transcribe(
        self,
        pcm16k: np.ndarray,
        state: Qwen3StreamingState,
    ) -> Qwen3StreamingState:
        """
        流式识别音频块

        Args:
            pcm16k: 16kHz 单声道音频数据（float32 numpy 数组）
            state: 流式状态对象

        Returns:
            Qwen3StreamingState: 更新后的状态对象
        """
        if self.model is None:
            raise DefaultServerErrorException("模型未加载")

        try:
            # 确保音频是 float32
            if pcm16k.dtype == np.int16:
                pcm16k = (pcm16k.astype(np.float32) / 32768.0)
            else:
                pcm16k = pcm16k.astype(np.float32, copy=False)

            # 调用 Qwen3-ASR 的流式识别
            self.model.streaming_transcribe(pcm16k, state.internal_state)

            # 更新包装器状态
            state.chunk_count += 1
            state.last_text = state.internal_state.text
            state.last_language = state.internal_state.language

            return state
        except Exception as e:
            logger.error(f"流式识别失败: {e}")
            raise DefaultServerErrorException(f"流式识别失败: {e}")

    def finish_streaming_transcribe(
        self,
        state: Qwen3StreamingState,
    ) -> Qwen3StreamingState:
        """
        结束流式识别，处理剩余音频

        Args:
            state: 流式状态对象

        Returns:
            Qwen3StreamingState: 最终结果状态
        """
        if self.model is None:
            raise DefaultServerErrorException("模型未加载")

        try:
            self.model.finish_streaming_transcribe(state.internal_state)

            # 更新包装器状态
            state.last_text = state.internal_state.text
            state.last_language = state.internal_state.language

            return state
        except Exception as e:
            logger.error(f"结束流式识别失败: {e}")
            raise DefaultServerErrorException(f"结束流式识别失败: {e}")


# 自动注册 Qwen3 引擎（由 manager.py 显式触发）
def _register_qwen3_engine(register_func, model_config_cls):
    """注册 Qwen3 引擎到引擎注册表

    Args:
        register_func: register_engine 函数
        model_config_cls: ModelConfig 类
    """
    from app.core.config import settings

    def _create_qwen3_engine(config) -> "Qwen3ASREngine":
        # 使用 extra_kwargs 中的配置，允许 models.json 覆盖默认行为
        # 过滤掉 None 值，避免覆盖默认参数
        extra_kwargs = {k: v for k, v in config.extra_kwargs.items() if v is not None}

        # 获取模型 ID（如 "Qwen/Qwen3-ASR-0.6B"）
        model_id = config.models.get("offline")

        # 确保模型在 HuggingFace 缓存中（从 ModelScope 复制）
        # vLLM 需要 HF 格式的缓存才能正确加载
        _ensure_model_in_hf_cache(model_id)

        # 同样处理 forced_aligner_path
        forced_aligner_path = extra_kwargs.get("forced_aligner_path")
        if forced_aligner_path:
            _ensure_model_in_hf_cache(forced_aligner_path)

        return Qwen3ASREngine(
            model_path=model_id,
            device=settings.DEVICE,
            **extra_kwargs
        )

    register_func("qwen3", _create_qwen3_engine)


def _ensure_model_in_hf_cache(model_id: str) -> str:
    """确保模型在 HuggingFace 缓存格式中（从 ModelScope 创建符号链接）

    vLLM 需要 HuggingFace 格式的缓存结构：
    - HF: ~/.cache/huggingface/hub/models--{org}--{model}/snapshots/{hash}/
    - MS: ~/.cache/modelscope/hub/models/{model_id}/

    Args:
        model_id: 模型 ID（如 "Qwen/Qwen3-ASR-0.6B"）

    Returns:
        原始 model_id
    """
    if not model_id or os.path.isabs(model_id):
        return model_id

    from pathlib import Path

    # ModelScope 缓存路径
    ms_cache_path = Path.home() / ".cache" / "modelscope" / "hub" / "models" / model_id

    if not ms_cache_path.exists():
        logger.warning(f"模型 {model_id} 本地缓存不存在，将尝试在线下载")
        return model_id

    # HuggingFace 缓存路径
    parts = model_id.split("/")
    if len(parts) != 2:
        return model_id

    org, model = parts
    hf_cache_name = f"models--{org}--{model}"
    hf_cache_path = Path.home() / ".cache" / "huggingface" / "hub" / hf_cache_name

    # 如果 HF 缓存已存在，无需处理
    if hf_cache_path.exists():
        return model_id

    # 创建 HF 格式的符号链接
    logger.info(f"创建 HF 缓存符号链接: {model_id}")
    try:
        hf_cache_path.mkdir(parents=True, exist_ok=True)

        # 创建 snapshots 目录
        snapshots_path = hf_cache_path / "snapshots"
        snapshots_path.mkdir(exist_ok=True)

        # 使用固定 hash（model_id 的 hash）
        import hashlib
        snapshot_hash = hashlib.sha256(model_id.encode()).hexdigest()[:12]
        snapshot_path = snapshots_path / snapshot_hash

        # 创建符号链接指向 ModelScope 缓存
        if not snapshot_path.exists():
            snapshot_path.symlink_to(ms_cache_path, target_is_directory=True)

        # 创建 refs/main 指向该 hash
        refs_path = hf_cache_path / "refs"
        refs_path.mkdir(exist_ok=True)
        (refs_path / "main").write_text(snapshot_hash)

        logger.info(f"HF 缓存符号链接创建完成: {snapshot_path} -> {ms_cache_path}")

    except Exception as e:
        logger.error(f"创建 HF 缓存符号链接失败: {e}")

    return model_id
