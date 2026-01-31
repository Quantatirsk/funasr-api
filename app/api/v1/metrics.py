# -*- coding: utf-8 -*-
"""
Prometheus 指标端点
提供 ASR 服务的性能指标和监控数据
"""

import time
import logging
from typing import Any, Callable, TypeVar, cast
from functools import wraps
from contextlib import contextmanager

from fastapi import APIRouter
from fastapi.responses import Response

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Metrics"])

# 尝试导入 prometheus_client
try:
    import prometheus_client

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("prometheus_client 未安装，使用模拟指标实现")
    prometheus_client = None  # type: ignore


# 指标类型定义
class _MockMetric:
    """模拟指标基类"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._val = 0.0
        self._labels: dict[str, Any] = {}

    def labels(self, **kwargs: Any) -> "_MockMetric":
        """获取带标签的指标"""
        return self

    def inc(self, amount: float = 1) -> None:
        """增加计数"""
        self._val += amount

    def dec(self, amount: float = 1) -> None:
        """减少计数"""
        self._val -= amount

    def observe(self, value: float) -> None:
        """观察值"""
        self._val = value

    def set(self, value: float) -> None:
        """设置值"""
        self._val = value


class _MockInfo:
    """模拟信息指标"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._info: dict[str, str] = {}

    def info(self, data: dict[str, str]) -> None:
        """设置信息"""
        self._info.update(data)


# 根据是否安装 prometheus_client 选择实现
if PROMETHEUS_AVAILABLE and prometheus_client:
    _Counter = prometheus_client.Counter
    _Histogram = prometheus_client.Histogram
    _Gauge = prometheus_client.Gauge
    _Info = prometheus_client.Info
    _CollectorRegistry = prometheus_client.CollectorRegistry
    _generate_latest = prometheus_client.generate_latest
    _CONTENT_TYPE_LATEST = prometheus_client.CONTENT_TYPE_LATEST
else:
    _Counter = _MockMetric  # type: ignore
    _Histogram = _MockMetric  # type: ignore
    _Gauge = _MockMetric  # type: ignore
    _Info = _MockInfo  # type: ignore
    _CollectorRegistry = object  # type: ignore

    def _generate_latest(registry: Any = None) -> bytes:
        return b"# Prometheus metrics (prometheus_client not installed)\n"

    _CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"


# 创建指标注册表
REGISTRY: Any = _CollectorRegistry() if PROMETHEUS_AVAILABLE else None

# ============ ASR 推理指标 ============

TRANSCRIPTION_COUNT: Any = _Counter(
    "asr_transcription_total",
    "Total number of transcriptions",
    ["model_id", "status"],
    registry=REGISTRY,
)

TRANSCRIPTION_DURATION: Any = _Histogram(
    "asr_transcription_duration_seconds",
    "Transcription duration in seconds",
    ["model_id"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0],
    registry=REGISTRY,
)

AUDIO_DURATION: Any = _Histogram(
    "asr_audio_duration_seconds",
    "Audio duration in seconds",
    ["model_id"],
    buckets=[1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0, 1800.0, 3600.0],
    registry=REGISTRY,
)

TRANSCRIPTION_RTF: Any = _Histogram(
    "asr_transcription_rtf",
    "Real-time factor (processing time / audio duration)",
    ["model_id"],
    buckets=[0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0],
    registry=REGISTRY,
)

# ============ 批处理指标 ============

BATCH_SIZE: Any = _Histogram(
    "asr_batch_size",
    "Batch size for transcription",
    buckets=[1, 2, 4, 8, 16, 32],
    registry=REGISTRY,
)

BATCH_INFERENCE_DURATION: Any = _Histogram(
    "asr_batch_inference_duration_seconds",
    "Batch inference duration in seconds",
    ["model_id"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
    registry=REGISTRY,
)

# ============ 系统指标 ============

ACTIVE_REQUESTS: Any = _Gauge(
    "asr_active_requests",
    "Number of active transcription requests",
    registry=REGISTRY,
)

QUEUE_SIZE: Any = _Gauge(
    "asr_queue_size",
    "Current queue size (if using queue-based processing)",
    registry=REGISTRY,
)

MODEL_LOADED: Any = _Gauge(
    "asr_model_loaded",
    "Whether the model is loaded (1=yes, 0=no)",
    ["model_id"],
    registry=REGISTRY,
)

# ============ 应用信息 ============

APP_INFO: Any = _Info(
    "asr_app",
    "ASR application information",
    registry=REGISTRY,
)


def init_app_info(version: str, model_mode: str) -> None:
    """初始化应用信息"""
    APP_INFO.info({
        "version": version,
        "model_mode": model_mode,
    })


# ============ 指标记录辅助函数 ============

def record_transcription_metrics(
    model_id: str,
    status: str,
    duration_sec: float,
    audio_duration_sec: float,
    batch_size: int = 1,
) -> None:
    """记录转写指标

    Args:
        model_id: 模型ID
        status: 状态（success/error）
        duration_sec: 处理耗时（秒）
        audio_duration_sec: 音频时长（秒）
        batch_size: 批处理大小
    """
    TRANSCRIPTION_COUNT.labels(model_id=model_id, status=status).inc()
    TRANSCRIPTION_DURATION.labels(model_id=model_id).observe(duration_sec)
    AUDIO_DURATION.labels(model_id=model_id).observe(audio_duration_sec)

    if audio_duration_sec > 0:
        rtf = duration_sec / audio_duration_sec
        TRANSCRIPTION_RTF.labels(model_id=model_id).observe(rtf)

    if batch_size > 1:
        BATCH_SIZE.observe(batch_size)
        BATCH_INFERENCE_DURATION.labels(model_id=model_id).observe(duration_sec)


def record_model_loaded(model_id: str, loaded: bool) -> None:
    """记录模型加载状态

    Args:
        model_id: 模型ID
        loaded: 是否已加载
    """
    MODEL_LOADED.labels(model_id=model_id).set(1 if loaded else 0)


@contextmanager
def active_request_counter():
    """上下文管理器：统计活跃请求数"""
    ACTIVE_REQUESTS.inc()
    try:
        yield
    finally:
        ACTIVE_REQUESTS.dec()


F = TypeVar("F", bound=Callable[..., Any])


def time_transcription(model_id: str) -> Callable[[F], F]:
    """装饰器：计时转写操作并记录指标

    示例:
        @time_transcription("qwen3-asr-1.7b")
        def transcribe(audio_path):
            ...
    """

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.time()
            status = "success"
            try:
                result = func(*args, **kwargs)
                return result
            except Exception:
                status = "error"
                raise
            finally:
                duration = time.time() - start_time
                TRANSCRIPTION_DURATION.labels(model_id=model_id).observe(duration)
                TRANSCRIPTION_COUNT.labels(model_id=model_id, status=status).inc()

        return cast(F, wrapper)

    return decorator


# ============ API 端点 ============


@router.get("/metrics")
async def metrics():
    """Prometheus 指标端点

    返回 Prometheus 格式的指标数据，可用于监控和告警。

    可用指标：
    - asr_transcription_total: 总转写次数
    - asr_transcription_duration_seconds: 转写耗时分布
    - asr_audio_duration_seconds: 音频时长分布
    - asr_transcription_rtf: 实时率（RTF）分布
    - asr_active_requests: 当前活跃请求数
    - asr_model_loaded: 模型加载状态
    """
    return Response(
        content=_generate_latest(REGISTRY),
        media_type=_CONTENT_TYPE_LATEST,
    )


@router.get("/metrics/health")
async def metrics_health():
    """指标系统健康检查"""
    return {
        "status": "healthy",
        "prometheus_available": PROMETHEUS_AVAILABLE,
        "metrics_enabled": True,
    }
