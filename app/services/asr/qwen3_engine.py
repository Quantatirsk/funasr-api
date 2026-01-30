# -*- coding: utf-8 -*-
"""
Qwen3-ASR 引擎 - 内嵌 vLLM 后端

支持 Qwen3-ASR-0.6B 和 Qwen3-ASR-1.7B 模型
通过 Qwen3ASRModel.LLM() 直接初始化 vLLM
"""

import logging
from typing import Optional, List, Any, Tuple

import torch
import numpy as np

from .engine import BaseASREngine, ASRRawResult, ASRSegmentResult, ASRFullResult, WordToken
from ...core.config import settings
from ...core.exceptions import DefaultServerErrorException
from ...utils.audio import get_audio_duration

logger = logging.getLogger(__name__)

# 延迟导入 qwen_asr，避免启动时加载
try:
    from qwen_asr import Qwen3ASRModel
except ImportError:
    Qwen3ASRModel = None
    logger.warning("qwen-asr 未安装，Qwen3-ASR 功能不可用")


class Qwen3ASREngine(BaseASREngine):
    """
    Qwen3-ASR 语音识别引擎

    使用 vLLM 后端进行高效推理，内置 VAD、标点、语言识别
    支持字级时间戳（通过 Qwen3-ForcedAligner）
    """

    def __init__(
        self,
        model_path: str = "Qwen/Qwen3-ASR-1.7B",
        device: str = "auto",
        gpu_memory_utilization: float = 0.8,
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
            gpu_memory_utilization: GPU 显存使用率（0-1）
            forced_aligner_path: 时间戳对齐模型路径（可选）
            max_inference_batch_size: 最大批处理大小
            max_new_tokens: 最大生成 token 数
            max_model_len: 最大模型序列长度（用于限制 KV 缓存）
        """
        if Qwen3ASRModel is None:
            raise ImportError(
                "qwen-asr 未安装，请运行: pip install qwen-asr[vllm]"
            )

        self.model_path = model_path
        self._device = self._detect_device(device)
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_inference_batch_size = max_inference_batch_size
        self.max_new_tokens = max_new_tokens

        # 构建 forced_aligner_kwargs
        forced_aligner_kwargs = None
        if forced_aligner_path:
            forced_aligner_kwargs = {
                "dtype": torch.bfloat16,
                "device_map": self._device if ":" in self._device else "cuda:0",
            }

        logger.info(f"正在加载 Qwen3-ASR 模型: {model_path}")
        logger.info(f"设备: {self._device}, GPU 显存使用率: {gpu_memory_utilization}")
        if max_model_len:
            logger.info(f"最大序列长度限制: {max_model_len}")

        try:
            # 构建 vLLM 初始化参数
            llm_kwargs = {
                "model": model_path,
                "gpu_memory_utilization": gpu_memory_utilization,
                "forced_aligner": forced_aligner_path,
                "forced_aligner_kwargs": forced_aligner_kwargs,
                "max_inference_batch_size": max_inference_batch_size,
                "max_new_tokens": max_new_tokens,
            }
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

        Returns:
            ASRRawResult 包含文本和分段信息，每个 segment 包含 word_tokens
        """
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

        Note: Qwen3-ASR vLLM 后端暂不支持 WebSocket 流式
        流式识别建议使用 qwen-asr-serve 或 vllm serve 单独部署
        """
        return False
