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

    def transcribe_long_audio(
        self,
        audio_path: str,
        hotwords: str = "",
        enable_punctuation: bool = False,
        enable_itn: bool = False,
        sample_rate: int = 16000,
        max_segment_sec: float = 55.0,
    ) -> ASRFullResult:
        """转录长音频文件（自动分段）

        Args:
            audio_path: 音频文件路径
            hotwords: 热词
            enable_punctuation: 是否启用标点
            enable_itn: 是否启用 ITN
            sample_rate: 采样率
            max_segment_sec: 每段最大时长（秒）

        Returns:
            ASRFullResult: 包含完整文本、分段结果和时长的结果
        """
        from ...utils.audio_splitter import AudioSplitter

        logger.info(f"[transcribe_long_audio] 方法被调用，音频: {audio_path}")

        try:
            # 获取音频时长
            logger.info("[transcribe_long_audio] 正在获取音频时长...")
            duration = get_audio_duration(audio_path)
            logger.info(f"[transcribe_long_audio] 音频时长: {duration:.2f}秒")

            # 检查是否需要分段
            if duration <= self.MAX_AUDIO_DURATION_SEC:
                # 短音频，使用 VAD 获取分段信息
                raw_result = self.transcribe_file_with_vad(
                    audio_path=audio_path,
                    hotwords=hotwords,
                    enable_punctuation=enable_punctuation,
                    enable_itn=enable_itn,
                    sample_rate=sample_rate,
                )

                # 如果没有分段信息，创建一个完整的分段
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

            # 长音频，需要分段
            splitter = AudioSplitter(
                max_segment_sec=max_segment_sec, device=self.device
            )
            segments = splitter.split_audio_file(audio_path)

            logger.info(f"音频已分割为 {len(segments)} 段")

            # 逐段识别，使用 try-finally 确保临时文件被清理
            results: List[ASRSegmentResult] = []
            all_texts: List[str] = []

            try:
                for idx, segment in enumerate(segments):
                    logger.info(
                        f"识别分段 {idx + 1}/{len(segments)}: "
                        f"{segment.start_sec:.2f}s - {segment.end_sec:.2f}s"
                    )

                    try:
                        # 确保临时文件存在
                        if not segment.temp_file:
                            logger.warning(f"分段 {idx + 1} 临时文件不存在，跳过")
                            continue

                        # 识别该段
                        segment_text = self.transcribe_file(
                            audio_path=segment.temp_file,
                            hotwords=hotwords,
                            enable_punctuation=enable_punctuation,
                            enable_itn=enable_itn,
                            enable_vad=False,  # 分段后不再需要 VAD
                            sample_rate=sample_rate,
                        )

                        if segment_text:
                            results.append(
                                ASRSegmentResult(
                                    text=segment_text,
                                    start_time=segment.start_sec,
                                    end_time=segment.end_sec,
                                )
                            )
                            all_texts.append(segment_text)
                            logger.debug(f"分段 {idx + 1} 识别结果: {segment_text[:50]}...")

                    except Exception as e:
                        logger.error(f"分段 {idx + 1} 识别失败: {e}")
                        # 继续处理下一段
            finally:
                # 确保临时文件被清理，即使发生异常
                AudioSplitter.cleanup_segments(segments)

            # 合并结果
            full_text = "".join(all_texts)

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
    """FunASR语音识别引擎"""

    def __init__(
        self,
        offline_model_path: Optional[str] = None,
        realtime_model_path: Optional[str] = None,
        device: str = "auto",
        vad_model: Optional[str] = None,
        vad_model_revision: str = "v2.0.4",
        punc_model: Optional[str] = None,
        punc_model_revision: str = "v2.0.4",
        punc_realtime_model: Optional[str] = None,
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
        self.vad_model_revision = vad_model_revision
        self.punc_model = punc_model or settings.PUNC_MODEL
        self.punc_model_revision = punc_model_revision
        self.punc_realtime_model = punc_realtime_model or settings.PUNC_REALTIME_MODEL

        self._load_models_based_on_mode()

    def _load_models_based_on_mode(self) -> None:
        """根据ASR_MODEL_MODE加载对应的模型"""
        mode = settings.ASR_MODEL_MODE.lower()

        if mode == "all":
            # 加载所有可用模型
            if self.offline_model_path:
                self._load_offline_model()
            if self.realtime_model_path:
                self._load_realtime_model()
        elif mode == "offline":
            # 只加载离线模型
            if self.offline_model_path:
                self._load_offline_model()
            else:
                logger.warning("ASR_MODEL_MODE设置为offline，但未提供离线模型路径")
        elif mode == "realtime":
            # 只加载实时模型
            if self.realtime_model_path:
                self._load_realtime_model()
            else:
                logger.warning("ASR_MODEL_MODE设置为realtime，但未提供实时模型路径")
        else:
            raise DefaultServerErrorException(f"不支持的ASR_MODEL_MODE: {mode}")

    def _load_offline_model(self) -> None:
        """加载离线FunASR模型（不再内嵌VAD/PUNC，改用全局实例）"""
        try:
            logger.info(f"正在加载离线FunASR模型: {self.offline_model_path}")

            model_kwargs = {
                "model": self.offline_model_path,
                "device": self._device,
                **settings.FUNASR_AUTOMODEL_KWARGS,
            }

            self.offline_model = AutoModel(**model_kwargs)
            logger.info("离线FunASR模型加载成功（VAD/PUNC将按需使用全局实例）")

        except Exception as e:
            raise DefaultServerErrorException(f"离线FunASR模型加载失败: {str(e)}")

    def _load_realtime_model(self) -> None:
        """加载实时FunASR模型（不再内嵌PUNC，改用全局实例）"""
        try:
            logger.info(f"正在加载实时FunASR模型: {self.realtime_model_path}")

            model_kwargs = {
                "model": self.realtime_model_path,
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
    ) -> str:
        """使用FunASR转录音频文件（支持动态启用VAD和PUNC）

        根据参数组合采用不同策略：
        1. 只PUNC：手动后处理
        2. 有VAD：利用全局实例直接构造临时AutoModel（复用已加载模型）
        """
        _ = sample_rate  # 当前未使用
        # 优先使用离线模型进行文件识别
        if not self.offline_model:
            raise DefaultServerErrorException(
                "离线模型未加载，无法进行文件识别。"
                "请将 ASR_MODEL_MODE 设置为 offline 或 all"
            )

        try:
            # 根据参数决定是否需要VAD/PUNC
            need_vad = enable_vad
            need_punc = enable_punctuation

            if need_vad:
                # 使用VAD时，需要构建临时AutoModel包装器
                # 预加载全局VAD和PUNC实例
                logger.debug("启用VAD，预加载全局VAD模型")
                vad_model_instance = get_global_vad_model(self._device)

                punc_model_instance = None
                if need_punc:
                    logger.debug("预加载全局PUNC模型")
                    punc_model_instance = get_global_punc_model(self._device)

                # 创建临时AutoModel包装器（复用已加载的模型）
                temp_automodel = TempAutoModelWrapper()
                temp_automodel.model = self.offline_model.model
                temp_automodel.kwargs = self.offline_model.kwargs
                temp_automodel.model_path = self.offline_model.model_path

                # 设置VAD（使用全局实例）
                temp_automodel.vad_model = vad_model_instance.model
                temp_automodel.vad_kwargs = vad_model_instance.kwargs

                # 设置PUNC（使用全局实例）
                if punc_model_instance:
                    temp_automodel.punc_model = punc_model_instance.model
                    temp_automodel.punc_kwargs = punc_model_instance.kwargs

                logger.debug("临时AutoModel构建完成，调用generate")
                result = temp_automodel.generate(
                    input=audio_path,
                    hotword=hotwords if hotwords else None,
                    cache={},
                )
            else:
                # 不使用VAD，直接识别
                result = self.offline_model.generate(
                    input=audio_path,
                    hotword=hotwords if hotwords else None,
                    cache={},
                )

                # 如果启用了PUNC但没有VAD，需要手动应用PUNC
                if need_punc and result and len(result) > 0:
                    text = result[0].get("text", "").strip()
                    if text:
                        logger.debug("手动应用PUNC模型（因为未启用VAD）")
                        punc_model_instance = get_global_punc_model(self._device)
                        punc_result = punc_model_instance.generate(
                            input=text,
                            cache={},
                        )
                        if punc_result and len(punc_result) > 0:
                            result[0]["text"] = punc_result[0].get("text", text)
                            logger.debug("标点符号添加完成")

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

        if not self.offline_model:
            raise DefaultServerErrorException(
                "离线模型未加载，无法进行文件识别。"
            )

        try:
            # 预加载全局 VAD 和 PUNC 模型
            logger.debug("启用 VAD 进行分段识别")
            vad_model_instance = get_global_vad_model(self._device)

            punc_model_instance = None
            if enable_punctuation:
                punc_model_instance = get_global_punc_model(self._device)

            # 创建临时 AutoModel 包装器
            temp_automodel = TempAutoModelWrapper()
            temp_automodel.model = self.offline_model.model
            temp_automodel.kwargs = self.offline_model.kwargs
            temp_automodel.model_path = self.offline_model.model_path

            # 设置 VAD
            temp_automodel.vad_model = vad_model_instance.model
            temp_automodel.vad_kwargs = vad_model_instance.kwargs

            # 设置 PUNC
            if punc_model_instance:
                temp_automodel.punc_model = punc_model_instance.model
                temp_automodel.punc_kwargs = punc_model_instance.kwargs

            # 调用识别
            result = temp_automodel.generate(
                input=audio_path,
                hotword=hotwords if hotwords else None,
                cache={},
            )

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

    def is_model_loaded(self) -> bool:
        """检查模型是否已加载"""
        return self.offline_model is not None or self.realtime_model is not None

    @property
    def device(self) -> str:
        """获取设备信息"""
        return self._device


# 全局ASR引擎实例缓存
_asr_engine: Optional[BaseASREngine] = None

# 全局VAD模型缓存（避免重复加载）
_global_vad_model = None
_vad_model_lock = threading.Lock()

# 全局标点符号模型缓存（避免重复加载）
_global_punc_model = None
_punc_model_lock = threading.Lock()

# 全局实时标点符号模型缓存（避免重复加载）
_global_punc_realtime_model = None
_punc_realtime_model_lock = threading.Lock()


def get_global_vad_model(device: str):
    """获取全局VAD模型实例"""
    global _global_vad_model

    with _vad_model_lock:
        if _global_vad_model is None:
            try:
                logger.info("正在加载全局VAD模型...")

                _global_vad_model = AutoModel(
                    model=settings.VAD_MODEL,
                    model_revision=settings.VAD_MODEL_REVISION,
                    device=device,
                    **settings.FUNASR_AUTOMODEL_KWARGS,
                )
                logger.info("全局VAD模型加载成功")
            except Exception as e:
                logger.error(f"全局VAD模型加载失败: {str(e)}")
                _global_vad_model = None
                raise

    return _global_vad_model


def clear_global_vad_model():
    """清理全局VAD模型缓存"""
    global _global_vad_model

    with _vad_model_lock:
        if _global_vad_model is not None:
            del _global_vad_model
            _global_vad_model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("全局VAD模型缓存已清理")


def get_global_punc_model(device: str):
    """获取全局标点符号模型实例（离线版）"""
    global _global_punc_model

    with _punc_model_lock:
        if _global_punc_model is None:
            try:
                logger.info("正在加载全局标点符号模型（离线）...")

                _global_punc_model = AutoModel(
                    model=settings.PUNC_MODEL,
                    model_revision=settings.PUNC_MODEL_REVISION,
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
                logger.info("正在加载全局标点符号模型（实时）...")

                _global_punc_realtime_model = AutoModel(
                    model=settings.PUNC_REALTIME_MODEL,
                    model_revision=settings.PUNC_MODEL_REVISION,
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
