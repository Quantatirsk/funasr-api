# -*- coding: utf-8 -*-
"""
ASR引擎模块 - 支持多种ASR引擎
"""

import torch
import logging
import threading
from typing import Optional, Dict, List, Any, cast
from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass
from funasr import AutoModel

from ...core.config import settings
from ...core.exceptions import DefaultServerErrorException
from ...utils.audio import get_audio_duration
from ...utils.text_processing import apply_itn_to_text
from .loaders import ModelLoaderFactory, BaseModelLoader


class TempAutoModelWrapper:
    """临时AutoModel包装器，用于动态组合VAD/PUNC模型"""

    def __init__(self) -> None:
        self.model: Any = None
        self.kwargs: Any = {}
        self.model_path: Any = ""
        self.spk_model: Any = None
        self.vad_model: Any = None
        self.vad_kwargs: Any = {}
        self.punc_model: Any = None
        self.punc_kwargs: Any = {}

    def inference(self, *args: Any, **kwargs: Any) -> Any:
        """调用AutoModel.inference"""
        return AutoModel.inference(cast(Any, self), *args, **kwargs)

    def inference_with_vad(self, *args: Any, **kwargs: Any) -> Any:
        """调用AutoModel.inference_with_vad"""
        return AutoModel.inference_with_vad(cast(Any, self), *args, **kwargs)

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        """调用AutoModel.generate"""
        return AutoModel.generate(cast(Any, self), *args, **kwargs)


@dataclass
class ASRSegmentResult:
    """ASR 分段识别结果"""

    text: str  # 该段识别文本
    start_time: float  # 开始时间（秒）
    end_time: float  # 结束时间（秒）
    speaker_id: Optional[str] = None  # 说话人ID（多说话人模式）


@dataclass
class ASRFullResult:
    """ASR 完整识别结果（支持长音频）"""

    text: str  # 完整识别文本
    segments: List[ASRSegmentResult]  # 分段结果
    duration: float  # 音频总时长（秒）


@dataclass
class ASRRawResult:
    """ASR 原始识别结果（包含时间戳）"""

    text: str  # 完整识别文本
    segments: List[ASRSegmentResult]  # 分段结果（从 VAD 时间戳解析）


logger = logging.getLogger(__name__)


def resolve_model_path(model_id: Optional[str]) -> str:
    """将模型 ID 解析为本地缓存路径（如果存在）

    FunASR/ModelScope 的缓存目录结构:
    ~/.cache/modelscope/hub/{model_id}/

    如果本地缓存存在，返回本地路径；否则返回原始 model_id
    """
    import os
    from pathlib import Path

    if not model_id:
        raise ValueError("model_id 不能为空")

    # 获取 ModelScope 缓存目录
    cache_dir = os.environ.get("MODELSCOPE_CACHE", os.path.expanduser("~/.cache/modelscope"))

    # ModelScope 有两种可能的缓存路径结构
    possible_paths = [
        Path(cache_dir) / "hub" / model_id,
        Path(cache_dir) / "models" / model_id,
    ]

    # 检查哪个路径存在
    for local_path in possible_paths:
        if local_path.exists() and local_path.is_dir():
            resolved = str(local_path)
            logger.info(f"模型 {model_id} 使用本地缓存: {resolved}")
            return resolved

    # 都不存在，返回模型ID（运行时会自动下载）
    logger.warning(f"模型 {model_id} 本地缓存不存在，将在运行时下载")
    return model_id


class ModelType(Enum):
    """模型类型枚举"""

    OFFLINE = "offline"
    REALTIME = "realtime"


class BaseASREngine(ABC):
    """基础ASR引擎抽象基类"""

    # 默认最大音频时长限制（秒）
    MAX_AUDIO_DURATION_SEC = 60.0

    @abstractmethod
    def transcribe_file(
        self,
        audio_path: str,
        hotwords: str = "",
        enable_punctuation: bool = False,
        enable_itn: bool = False,
        enable_vad: bool = False,
        sample_rate: int = 16000,
    ) -> str:
        """转录音频文件"""
        pass

    @abstractmethod
    def transcribe_file_with_vad(
        self,
        audio_path: str,
        hotwords: str = "",
        enable_punctuation: bool = True,
        enable_itn: bool = True,
        sample_rate: int = 16000,
    ) -> ASRRawResult:
        """使用 VAD 转录音频文件，返回带时间戳分段的结果"""
        pass

    def transcribe_long_audio(
        self,
        audio_path: str,
        hotwords: str = "",
        enable_punctuation: bool = False,
        enable_itn: bool = False,
        sample_rate: int = 16000,
        max_segment_sec: float = 55.0,
        enable_speaker_diarization: bool = True,
    ) -> ASRFullResult:
        """转录长音频文件（自动分段）

        Args:
            audio_path: 音频文件路径
            hotwords: 热词
            enable_punctuation: 是否启用标点
            enable_itn: 是否启用 ITN
            sample_rate: 采样率
            max_segment_sec: 每段最大时长（秒）
            enable_speaker_diarization: 是否启用说话人分离

        Returns:
            ASRFullResult: 包含完整文本、分段结果和时长的结果
        """
        from ...utils.audio_splitter import AudioSplitter

        logger.info(f"[transcribe_long_audio] 音频: {audio_path}, speaker_diarization={enable_speaker_diarization}")

        try:
            # 获取音频时长
            duration = get_audio_duration(audio_path)
            logger.info(f"[transcribe_long_audio] 音频时长: {duration:.2f}秒")

            # 短音频且不启用说话人分离：直接使用 VAD 识别
            if duration <= self.MAX_AUDIO_DURATION_SEC and not enable_speaker_diarization:
                raw_result = self.transcribe_file_with_vad(
                    audio_path=audio_path,
                    hotwords=hotwords,
                    enable_punctuation=enable_punctuation,
                    enable_itn=enable_itn,
                    sample_rate=sample_rate,
                )

                segments = raw_result.segments
                if not segments:
                    segments = [
                        ASRSegmentResult(
                            text=raw_result.text,
                            start_time=0.0,
                            end_time=duration
                        )
                    ]

                return ASRFullResult(
                    text=raw_result.text,
                    segments=segments,
                    duration=duration,
                )

            # 长音频或多说话人：需要分段
            speaker_segments = None
            audio_segments = None

            if enable_speaker_diarization:
                # 多说话人：使用说话人分离
                from ...utils.speaker_diarizer import SpeakerDiarizer

                logger.info("使用说话人分离模式")
                diarizer = SpeakerDiarizer(max_segment_sec=max_segment_sec)
                speaker_segments = diarizer.split_audio_by_speakers(audio_path)

                if not speaker_segments:
                    logger.warning("说话人分离未检测到片段，fallback 到 VAD 分割")

            if not speaker_segments:
                # 单说话人：使用 VAD 分割
                logger.info("使用 VAD 分割模式")
                splitter = AudioSplitter(
                    max_segment_sec=max_segment_sec, device=self.device
                )
                audio_segments = splitter.split_audio_file(audio_path)

            # 选择要处理的片段
            segments_to_process = speaker_segments if speaker_segments else audio_segments
            if not segments_to_process:
                raise DefaultServerErrorException("音频分割失败：未生成任何片段")

            logger.info(f"音频已分割为 {len(segments_to_process)} 段")

            # 批处理推理配置
            batch_size = settings.ASR_BATCH_SIZE
            logger.info(f"使用批处理推理，batch_size={batch_size}")

            results: List[ASRSegmentResult] = []
            all_texts: List[str] = []

            # 批处理推理
            for batch_start in range(0, len(segments_to_process), batch_size):
                batch_end = min(batch_start + batch_size, len(segments_to_process))
                batch_segments = segments_to_process[batch_start:batch_end]

                logger.info(
                    f"推理批次 {batch_start//batch_size + 1}/{(len(segments_to_process) + batch_size - 1)//batch_size}: "
                    f"片段 {batch_start+1}-{batch_end}/{len(segments_to_process)}"
                )

                try:
                    # 批量推理
                    batch_texts = self._transcribe_batch(
                        segments=batch_segments,
                        hotwords=hotwords,
                        enable_punctuation=enable_punctuation,
                        enable_itn=enable_itn,
                        sample_rate=sample_rate,
                    )

                    # 组装结果
                    for seg, text in zip(batch_segments, batch_texts):
                        if text:
                            speaker_id = getattr(seg, 'speaker_id', None)
                            results.append(
                                ASRSegmentResult(
                                    text=text,
                                    start_time=seg.start_sec,
                                    end_time=seg.end_sec,
                                    speaker_id=speaker_id,
                                )
                            )
                            all_texts.append(text)

                    logger.info(f"批次推理完成，有效片段: {len(batch_texts)}")

                except Exception as e:
                    logger.error(f"批次推理失败: {e}, 跳过该批次")

                except Exception as e:
                    logger.error(f"批次推理失败: {e}, 跳过该批次")

            # 清理临时文件
            try:
                if enable_speaker_diarization and speaker_segments:
                    from ...utils.speaker_diarizer import SpeakerDiarizer
                    SpeakerDiarizer.cleanup_segments(speaker_segments)
                elif audio_segments:
                    AudioSplitter.cleanup_segments(audio_segments)
            except Exception as e:
                logger.warning(f"清理临时文件时出错: {e}")

            full_text = "\n".join(all_texts)

            logger.info(
                f"长音频识别完成，共 {len(results)} 个有效分段，"
                f"总字符数: {len(full_text)}"
            )

            return ASRFullResult(
                text=full_text,
                segments=results,
                duration=duration,
            )

        except Exception as e:
            logger.error(f"长音频识别失败: {e}")
            raise DefaultServerErrorException(f"长音频识别失败: {str(e)}")

    @abstractmethod
    def is_model_loaded(self) -> bool:
        """检查模型是否已加载"""
        pass

    @property
    @abstractmethod
    def device(self) -> str:
        """获取设备信息"""
        pass

    @property
    @abstractmethod
    def supports_realtime(self) -> bool:
        """是否支持实时识别"""
        pass

    def _transcribe_batch(
        self,
        segments: List[Any],
        hotwords: str = "",
        enable_punctuation: bool = False,
        enable_itn: bool = False,
        sample_rate: int = 16000,
    ) -> List[str]:
        """批量推理多个音频片段

        Args:
            segments: 音频片段列表（每个片段需要有 temp_file 属性）
            hotwords: 热词
            enable_punctuation: 是否启用标点
            enable_itn: 是否启用 ITN
            sample_rate: 采样率

        Returns:
            识别文本列表，与输入片段一一对应
        """
        # 默认实现：逐个推理（子类可以重写实现真正的批处理）
        results = []
        for idx, seg in enumerate(segments):
            try:
                if not seg.temp_file:
                    logger.warning(f"批处理片段 {idx + 1} 临时文件不存在，跳过")
                    results.append("")
                    continue

                text = self.transcribe_file(
                    audio_path=seg.temp_file,
                    hotwords=hotwords,
                    enable_punctuation=enable_punctuation,
                    enable_itn=enable_itn,
                    enable_vad=False,
                    sample_rate=sample_rate,
                )
                results.append(text)
            except Exception as e:
                logger.error(f"批处理片段 {idx + 1} 推理失败: {e}")
                results.append("")

        return results

    def _detect_device(self, device: str = "auto") -> str:
        """检测可用设备"""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda:0"
            else:
                return "cpu"
        return device


class RealTimeASREngine(BaseASREngine):
    """实时ASR引擎抽象基类"""

    @property
    def supports_realtime(self) -> bool:
        """支持实时识别"""
        return True

    @abstractmethod
    def transcribe_websocket(
        self,
        audio_chunk: bytes,
        cache: Optional[Dict] = None,
        is_final: bool = False,
        **kwargs,
    ) -> str:
        """WebSocket流式语音识别"""
        pass


class FunASREngine(RealTimeASREngine):
    """FunASR语音识别引擎 - 使用模块化加载器架构"""

    def __init__(
        self,
        offline_model_path: Optional[str] = None,
        realtime_model_path: Optional[str] = None,
        device: str = "auto",
        vad_model: Optional[str] = None,
        punc_model: Optional[str] = None,
        punc_realtime_model: Optional[str] = None,
        enable_lm: bool = True,
        extra_model_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.offline_model: Optional[AutoModel] = None
        self.realtime_model: Optional[AutoModel] = None
        self.punc_model_instance: Optional[AutoModel] = None
        self.punc_realtime_model_instance: Optional[AutoModel] = None
        self._device: str = self._detect_device(device)

        # 模型路径配置
        self.offline_model_path = offline_model_path
        self.realtime_model_path = realtime_model_path

        # 辅助模型配置
        self.vad_model = vad_model or settings.VAD_MODEL
        self.punc_model = punc_model or settings.PUNC_MODEL
        self.punc_realtime_model = punc_realtime_model or settings.PUNC_REALTIME_MODEL

        # 语言模型配置
        self.enable_lm = enable_lm and settings.ASR_ENABLE_LM
        self.lm_model = settings.LM_MODEL if self.enable_lm else None
        self.lm_weight = settings.LM_WEIGHT
        self.lm_beam_size = settings.LM_BEAM_SIZE

        # 额外的模型加载参数
        self.extra_model_kwargs = extra_model_kwargs or {}

        # 模型加载器（由 _load_offline_model 创建）
        self._offline_loader: Optional[BaseModelLoader] = None

        self._load_models_based_on_mode()

    def _load_models_based_on_mode(self) -> None:
        """根据ASR_MODEL_MODE加载对应的模型"""
        mode = settings.ASR_MODEL_MODE.lower()

        if mode == "all":
            if self.offline_model_path:
                self._load_offline_model()
            if self.realtime_model_path:
                self._load_realtime_model()
        elif mode == "offline":
            if self.offline_model_path:
                self._load_offline_model()
            else:
                logger.warning("ASR_MODEL_MODE设置为offline，但未提供离线模型路径")
        elif mode == "realtime":
            if self.realtime_model_path:
                self._load_realtime_model()
            else:
                logger.warning("ASR_MODEL_MODE设置为realtime，但未提供实时模型路径")
        else:
            raise DefaultServerErrorException(f"不支持的ASR_MODEL_MODE: {mode}")

    def _load_offline_model(self) -> None:
        """加载离线模型 - 使用模块化加载器"""
        if not self.offline_model_path:
            raise DefaultServerErrorException("未提供离线模型路径")

        try:
            # 使用工厂创建对应的加载器
            self._offline_loader = ModelLoaderFactory.create_loader(
                model_path=self.offline_model_path,
                device=self._device,
                extra_kwargs=self.extra_model_kwargs,
                enable_lm=self.enable_lm,
                lm_model=self.lm_model,
                lm_weight=self.lm_weight,
                lm_beam_size=self.lm_beam_size,
            )

            # 使用加载器加载模型
            self.offline_model = self._offline_loader.load()
            logger.info(f"离线模型加载成功（类型: {self._offline_loader.model_type}）")

        except Exception as e:
            raise DefaultServerErrorException(f"离线FunASR模型加载失败: {str(e)}")

    def _load_realtime_model(self) -> None:
        """加载实时FunASR模型（不再内嵌PUNC，改用全局实例）"""
        try:
            # 解析模型路径：优先使用本地缓存
            resolved_model_path = resolve_model_path(self.realtime_model_path)
            logger.info(f"正在加载实时FunASR模型: {resolved_model_path}")

            model_kwargs = {
                "model": resolved_model_path,
                "device": self._device,
                **settings.FUNASR_AUTOMODEL_KWARGS,
            }

            self.realtime_model = AutoModel(**model_kwargs)
            logger.info("实时FunASR模型加载成功（PUNC将按需使用全局实例）")

        except Exception as e:
            raise DefaultServerErrorException(f"实时FunASR模型加载失败: {str(e)}")

    def transcribe_file(
        self,
        audio_path: str,
        hotwords: str = "",
        enable_punctuation: bool = False,
        enable_itn: bool = False,
        enable_vad: bool = False,
        sample_rate: int = 16000,
        language: Optional[str] = None,
    ) -> str:
        """使用FunASR转录音频文件

        使用模块化加载器处理不同模型类型的推理逻辑：
        1. Fun-ASR-Nano（端到端 Audio-LLM）：直接调用，不使用外部 VAD/PUNC
        2. Paraformer（传统模型）：支持动态组合 VAD/PUNC/LM
        """
        _ = sample_rate  # 当前未使用
        if not self.offline_model or not self._offline_loader:
            raise DefaultServerErrorException(
                "离线模型未加载，无法进行文件识别。"
                "请将 ASR_MODEL_MODE 设置为 offline 或 all"
            )

        try:
            # 使用加载器准备推理参数
            generate_kwargs = self._offline_loader.prepare_generate_kwargs(
                audio_path=audio_path,
                hotwords=hotwords,
                enable_punctuation=enable_punctuation,
                enable_itn=enable_itn,
                language=language,
            )

            # 根据加载器特性决定如何处理 VAD/PUNC
            if self._offline_loader.supports_external_vad and enable_vad:
                # 传统模型：使用外部 VAD + PUNC
                result = self._transcribe_with_vad(
                    audio_path, generate_kwargs, enable_punctuation
                )
            else:
                # 端到端模型：直接推理
                logger.debug(f"使用 {self._offline_loader.model_type} 进行端到端推理")
                result = self.offline_model.generate(**generate_kwargs)

                # 对于传统模型，如果没有 VAD 但需要 PUNC，手动添加
                if (
                    self._offline_loader.supports_external_punc
                    and not enable_vad
                    and enable_punctuation
                ):
                    result = self._apply_punc_to_result(result)

            # 提取识别结果
            if result and len(result) > 0:
                text = result[0].get("text", "")
                text = text.strip()

                # 应用ITN处理
                if enable_itn and text:
                    logger.debug(f"应用ITN处理前: {text}")
                    text = apply_itn_to_text(text)
                    logger.debug(f"应用ITN处理后: {text}")

                return text
            else:
                return ""

        except Exception as e:
            raise DefaultServerErrorException(f"语音识别失败: {str(e)}")

    def transcribe_file_with_vad(
        self,
        audio_path: str,
        hotwords: str = "",
        enable_punctuation: bool = True,
        enable_itn: bool = True,
        sample_rate: int = 16000,
    ) -> ASRRawResult:
        """使用 VAD 转录音频文件，返回带时间戳分段的结果

        Args:
            audio_path: 音频文件路径
            hotwords: 热词
            enable_punctuation: 是否启用标点
            enable_itn: 是否启用 ITN
            sample_rate: 采样率

        Returns:
            ASRRawResult: 包含文本和分段时间戳的结果
        """
        _ = sample_rate  # 当前未使用

        if not self.offline_model or not self._offline_loader:
            raise DefaultServerErrorException(
                "离线模型未加载，无法进行文件识别。"
            )

        try:
            # 使用加载器准备推理参数
            generate_kwargs = self._offline_loader.prepare_generate_kwargs(
                audio_path=audio_path,
                hotwords=hotwords,
                enable_punctuation=enable_punctuation,
                enable_itn=enable_itn,
            )

            # 根据加载器特性决定如何处理
            if self._offline_loader.supports_external_vad:
                # 传统模型：使用外部 VAD
                result = self._transcribe_with_vad(
                    audio_path, generate_kwargs, enable_punctuation
                )
            else:
                # 端到端模型：直接推理
                logger.debug(f"使用 {self._offline_loader.model_type} 进行端到端推理")
                result = self.offline_model.generate(**generate_kwargs)

            # 解析结果
            segments: List[ASRSegmentResult] = []
            full_text = ""

            if result and len(result) > 0:
                full_text = result[0].get("text", "").strip()

                # 解析时间戳
                # FunASR 返回格式可能是:
                # 1. {"sentence_info": [[start_ms, end_ms, "text"], ...]}
                # 2. {"timestamp": [[start_ms, end_ms], ...]}
                sentence_info = result[0].get("sentence_info", [])

                if sentence_info and isinstance(sentence_info, list):
                    for sent in sentence_info:
                        try:
                            if isinstance(sent, dict):
                                # 格式: {"start": ms, "end": ms, "text": "..."}
                                start_ms = sent.get("start", 0)
                                end_ms = sent.get("end", 0)
                                text = sent.get("text", "")
                            elif isinstance(sent, (list, tuple)) and len(sent) >= 3:
                                # 格式: [start_ms, end_ms, "text"]
                                start_ms = sent[0]
                                end_ms = sent[1]
                                text = sent[2] if len(sent) > 2 else ""
                            else:
                                continue

                            segments.append(ASRSegmentResult(
                                text=str(text),
                                start_time=start_ms / 1000.0,
                                end_time=end_ms / 1000.0,
                            ))
                        except (IndexError, TypeError, KeyError) as e:
                            logger.warning(f"解析 sentence_info 项失败: {e}")

                # 如果没有解析到分段信息，尝试从 timestamp 字段解析
                if not segments:
                    timestamp = result[0].get("timestamp", [])
                    if timestamp and isinstance(timestamp, list) and len(timestamp) > 0 and full_text:
                        try:
                            # timestamp 格式: [[start_ms, end_ms], ...]
                            first_ts = timestamp[0]
                            last_ts = timestamp[-1]
                            if isinstance(first_ts, (list, tuple)) and len(first_ts) >= 2:
                                start_ms = first_ts[0]
                                end_ms = last_ts[1] if isinstance(last_ts, (list, tuple)) and len(last_ts) >= 2 else first_ts[1]
                                segments.append(ASRSegmentResult(
                                    text=full_text,
                                    start_time=start_ms / 1000.0,
                                    end_time=end_ms / 1000.0,
                                ))
                        except (IndexError, TypeError) as e:
                            logger.warning(f"解析 timestamp 失败: {e}")

                # 应用 ITN 处理
                if enable_itn and full_text:
                    full_text = apply_itn_to_text(full_text)
                    # 同时对分段文本应用 ITN
                    for seg in segments:
                        seg.text = apply_itn_to_text(seg.text)

            return ASRRawResult(text=full_text, segments=segments)

        except Exception as e:
            raise DefaultServerErrorException(f"语音识别失败: {str(e)}")

    def transcribe_websocket(
        self,
        audio_chunk: bytes,
        cache: Optional[Dict] = None,
        is_final: bool = False,
        **kwargs: Any,
    ) -> str:
        """WebSocket流式语音识别（未实现）"""
        # 忽略未使用的参数（功能尚未实现）
        _ = (audio_chunk, cache, is_final, kwargs)
        if not self.realtime_model:
            raise DefaultServerErrorException(
                "实时模型未加载，无法进行WebSocket流式识别。"
                "请将 ASR_MODEL_MODE 设置为 realtime 或 all"
            )

        logger.warning("WebSocket流式识别功能尚未实现")
        return ""

    def _transcribe_with_vad(
        self,
        audio_path: str,
        generate_kwargs: Dict[str, Any],
        enable_punctuation: bool,
    ) -> List[Dict[str, Any]]:
        """使用 VAD 进行转录（传统模型专用）"""
        logger.debug("使用VAD进行分段识别")
        vad_model_instance = get_global_vad_model(self._device)

        punc_model_instance = None
        if enable_punctuation:
            logger.debug("预加载全局PUNC模型")
            punc_model_instance = get_global_punc_model(self._device)

        # 创建临时AutoModel包装器
        if self.offline_model is None:
            raise DefaultServerErrorException("离线模型未加载")
        temp_automodel = TempAutoModelWrapper()
        temp_automodel.model = self.offline_model.model
        temp_automodel.kwargs = self.offline_model.kwargs
        temp_automodel.model_path = self.offline_model.model_path

        # 设置VAD
        temp_automodel.vad_model = vad_model_instance.model
        temp_automodel.vad_kwargs = vad_model_instance.kwargs

        # 设置PUNC
        if punc_model_instance:
            temp_automodel.punc_model = punc_model_instance.model
            temp_automodel.punc_kwargs = punc_model_instance.kwargs

        return temp_automodel.generate(**generate_kwargs)

    def _apply_punc_to_text(self, text: str) -> str:
        """手动应用标点符号到文本

        Args:
            text: 无标点的识别文本

        Returns:
            添加标点后的文本
        """
        if not text:
            return text

        try:
            logger.debug(f"手动应用PUNC模型: {text[:50]}...")
            punc_model_instance = get_global_punc_model(self._device)
            punc_result = punc_model_instance.generate(
                input=text,
                cache={},
            )
            if punc_result and len(punc_result) > 0:
                text_with_punc = punc_result[0].get("text", text)
                logger.debug(f"标点添加完成: {text_with_punc[:50]}...")
                return text_with_punc
        except Exception as e:
            logger.warning(f"PUNC模型应用失败: {e}, 返回原文本")

        return text

    def _apply_punc_to_result(
        self, result: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """手动应用标点符号到识别结果"""
        if not result or len(result) == 0:
            return result

        text = result[0].get("text", "").strip()
        if not text:
            return result

        text_with_punc = self._apply_punc_to_text(text)
        result[0]["text"] = text_with_punc

        return result

    def _transcribe_batch(
        self,
        segments: List[Any],
        hotwords: str = "",
        enable_punctuation: bool = False,
        enable_itn: bool = False,
        sample_rate: int = 16000,
    ) -> List[str]:
        """批量推理多个音频片段（FunASR 优化版）

        利用 FunASR 的批量推理能力，比逐个推理快 2-3 倍
        """
        if not self.offline_model or not self._offline_loader:
            logger.warning("离线模型未加载，使用默认批处理实现")
            return super()._transcribe_batch(segments, hotwords, enable_punctuation, enable_itn, sample_rate)

        # 过滤有效片段
        valid_segments = [(idx, seg) for idx, seg in enumerate(segments) if seg.temp_file]
        if not valid_segments:
            return [""] * len(segments)

        try:
            import librosa

            # 加载音频数据（批量输入）
            logger.info(f"加载 {len(valid_segments)} 个音频片段到内存...")
            audio_inputs = []
            for idx, seg in valid_segments:
                try:
                    # 加载音频数据为numpy数组
                    audio_data, sr = librosa.load(seg.temp_file, sr=sample_rate)
                    audio_inputs.append(audio_data)
                except Exception as e:
                    logger.error(f"加载音频片段 {idx + 1} 失败: {e}")
                    audio_inputs.append(None)

            # 过滤加载成功的音频
            valid_inputs = [
                (idx, audio) for (idx, _), audio in zip(valid_segments, audio_inputs) if audio is not None
            ]

            if not valid_inputs:
                logger.warning("没有成功加载的音频片段")
                return [""] * len(segments)

            logger.info(f"FunASR 批量推理: {len(valid_inputs)} 个片段")

            # 准备批量推理参数
            batch_audio_data = [audio for _, audio in valid_inputs]

            # 使用加载器准备推理参数（注意：传入音频数据而不是路径）
            generate_kwargs = self._offline_loader.prepare_generate_kwargs(
                audio_path=None,  # 不使用路径
                hotwords=hotwords,
                enable_punctuation=enable_punctuation,
                enable_itn=enable_itn,
            )

            # 覆盖input参数为批量音频数据
            generate_kwargs['input'] = batch_audio_data
            generate_kwargs['batch_size'] = len(batch_audio_data)

            # 批量推理
            batch_results = self.offline_model.generate(**generate_kwargs)

            # 解析批量结果
            batch_texts = []
            if batch_results and isinstance(batch_results, list):
                for res in batch_results:
                    if isinstance(res, dict):
                        text = res.get("text", "").strip()

                        # 应用PUNC模型（Paraformer需要手动添加标点）
                        if enable_punctuation and text and self._offline_loader.supports_external_punc:
                            text = self._apply_punc_to_text(text)

                        # 应用ITN处理
                        if enable_itn and text:
                            text = apply_itn_to_text(text)

                        batch_texts.append(text)
                    else:
                        batch_texts.append("")
            else:
                logger.warning("批量推理返回结果格式异常")
                batch_texts = [""] * len(valid_inputs)

            # 将结果映射回原始顺序（包括跳过的片段）
            results = [""] * len(segments)
            for (idx, _), text in zip(valid_inputs, batch_texts):
                results[idx] = text

            logger.info(f"FunASR 批量推理完成，有效结果: {sum(1 for t in results if t)}/{len(results)}")
            return results

        except Exception as e:
            logger.error(f"FunASR 批量推理失败: {e}，fallback 到逐个推理")
            # fallback 到父类的逐个推理实现
            return super()._transcribe_batch(segments, hotwords, enable_punctuation, enable_itn, sample_rate)

    def is_model_loaded(self) -> bool:
        """检查模型是否已加载"""
        return self.offline_model is not None or self.realtime_model is not None

    @property
    def device(self) -> str:
        """获取设备信息"""
        return self._device


# 全局ASR引擎实例缓存
_asr_engine: Optional[BaseASREngine] = None

# 全局语音活动检测(VAD)模型缓存（避免重复加载）
_global_vad_model = None
_vad_model_lock = threading.Lock()

# 全局标点符号模型缓存（避免重复加载）
_global_punc_model = None
_punc_model_lock = threading.Lock()

# 全局实时标点符号模型缓存（避免重复加载）
_global_punc_realtime_model = None
_punc_realtime_model_lock = threading.Lock()


def get_global_vad_model(device: str):
    """获取全局语音活动检测(VAD)模型实例"""
    global _global_vad_model

    with _vad_model_lock:
        if _global_vad_model is None:
            try:
                # 解析模型路径：优先使用本地缓存
                resolved_vad_path = resolve_model_path(settings.VAD_MODEL)
                logger.info(f"正在加载全局语音活动检测(VAD)模型: {resolved_vad_path}")

                _global_vad_model = AutoModel(
                    model=resolved_vad_path,
                    device=device,
                    **settings.FUNASR_AUTOMODEL_KWARGS,
                )
                logger.info("全局语音活动检测(VAD)模型加载成功")
            except Exception as e:
                logger.error(f"全局语音活动检测(VAD)模型加载失败: {str(e)}")
                _global_vad_model = None
                raise

    return _global_vad_model


def clear_global_vad_model():
    """清理全局语音活动检测(VAD)模型缓存"""
    global _global_vad_model

    with _vad_model_lock:
        if _global_vad_model is not None:
            del _global_vad_model
            _global_vad_model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("全局语音活动检测(VAD)模型缓存已清理")


def get_global_punc_model(device: str):
    """获取全局标点符号模型实例（离线版）"""
    global _global_punc_model

    with _punc_model_lock:
        if _global_punc_model is None:
            try:
                # 解析模型路径：优先使用本地缓存
                resolved_punc_path = resolve_model_path(settings.PUNC_MODEL)
                logger.info(f"正在加载全局标点符号模型（离线）: {resolved_punc_path}")

                _global_punc_model = AutoModel(
                    model=resolved_punc_path,
                    device=device,
                    **settings.FUNASR_AUTOMODEL_KWARGS,
                )
                logger.info("全局标点符号模型（离线）加载成功")
            except Exception as e:
                logger.error(f"全局标点符号模型（离线）加载失败: {str(e)}")
                _global_punc_model = None
                raise

    return _global_punc_model


def clear_global_punc_model():
    """清理全局标点符号模型缓存"""
    global _global_punc_model

    with _punc_model_lock:
        if _global_punc_model is not None:
            del _global_punc_model
            _global_punc_model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("全局标点符号模型（离线）缓存已清理")


def get_global_punc_realtime_model(device: str):
    """获取全局实时标点符号模型实例"""
    global _global_punc_realtime_model

    with _punc_realtime_model_lock:
        if _global_punc_realtime_model is None:
            try:
                # 解析模型路径：优先使用本地缓存
                resolved_punc_realtime_path = resolve_model_path(settings.PUNC_REALTIME_MODEL)
                logger.info(f"正在加载全局标点符号模型（实时）: {resolved_punc_realtime_path}")

                _global_punc_realtime_model = AutoModel(
                    model=resolved_punc_realtime_path,
                    device=device,
                    **settings.FUNASR_AUTOMODEL_KWARGS,
                )
                logger.info("全局标点符号模型（实时）加载成功")
            except Exception as e:
                logger.error(f"全局标点符号模型（实时）加载失败: {str(e)}")
                _global_punc_realtime_model = None
                raise

    return _global_punc_realtime_model


def clear_global_punc_realtime_model():
    """清理全局实时标点符号模型缓存"""
    global _global_punc_realtime_model

    with _punc_realtime_model_lock:
        if _global_punc_realtime_model is not None:
            del _global_punc_realtime_model
            _global_punc_realtime_model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("全局标点符号模型（实时）缓存已清理")


def get_asr_engine() -> BaseASREngine:
    """获取全局ASR引擎实例"""
    global _asr_engine
    if _asr_engine is None:
        from .manager import get_model_manager

        model_manager = get_model_manager()
        _asr_engine = model_manager.get_asr_engine()
    return _asr_engine
