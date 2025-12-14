# -*- coding: utf-8 -*-
"""
统计计算模块
"""

from typing import List, Union
import numpy as np

from .models import ASRMetrics, TTSMetrics, AggregatedMetrics


def calculate_percentile(values: List[float], percentile: float) -> float:
    """
    计算百分位数

    Args:
        values: 数值列表
        percentile: 百分位 (0-100)

    Returns:
        百分位值
    """
    if not values:
        return 0.0
    return float(np.percentile(values, percentile))


def calculate_asr_statistics(
    metrics_list: List[ASRMetrics],
    concurrency_level: int,
    total_test_time: float,
) -> AggregatedMetrics:
    """
    计算 ASR 指标统计

    Args:
        metrics_list: ASR 指标列表
        concurrency_level: 并发级别
        total_test_time: 总测试时间 (秒)

    Returns:
        聚合后的指标
    """
    successful = [m for m in metrics_list if m.success]
    failed = [m for m in metrics_list if not m.success]

    # 提取各项指标值
    first_latencies = [
        m.first_result_latency_ms for m in successful if m.first_result_latency_ms is not None
    ]
    total_times = [
        m.total_processing_time_ms for m in successful if m.total_processing_time_ms is not None
    ]
    rtfs = [m.rtf for m in successful if m.rtf is not None]

    return AggregatedMetrics(
        test_type="asr",
        concurrency_level=concurrency_level,
        total_requests=len(metrics_list),
        successful_requests=len(successful),
        failed_requests=len(failed),
        total_test_time_seconds=total_test_time,
        # 首次延迟
        first_latency_avg=np.mean(first_latencies) if first_latencies else 0.0,
        first_latency_p50=calculate_percentile(first_latencies, 50),
        first_latency_p95=calculate_percentile(first_latencies, 95),
        first_latency_p99=calculate_percentile(first_latencies, 99),
        first_latency_max=max(first_latencies) if first_latencies else 0.0,
        # 总时间
        total_time_avg=np.mean(total_times) if total_times else 0.0,
        total_time_p50=calculate_percentile(total_times, 50),
        total_time_p95=calculate_percentile(total_times, 95),
        total_time_p99=calculate_percentile(total_times, 99),
        total_time_max=max(total_times) if total_times else 0.0,
        # RTF
        rtf_avg=np.mean(rtfs) if rtfs else 0.0,
        rtf_p50=calculate_percentile(rtfs, 50),
        rtf_p95=calculate_percentile(rtfs, 95),
        rtf_p99=calculate_percentile(rtfs, 99),
        rtf_max=max(rtfs) if rtfs else 0.0,
    )


def calculate_tts_statistics(
    metrics_list: List[TTSMetrics],
    concurrency_level: int,
    total_test_time: float,
) -> AggregatedMetrics:
    """
    计算 TTS 指标统计

    Args:
        metrics_list: TTS 指标列表
        concurrency_level: 并发级别
        total_test_time: 总测试时间 (秒)

    Returns:
        聚合后的指标
    """
    successful = [m for m in metrics_list if m.success]
    failed = [m for m in metrics_list if not m.success]

    # 提取各项指标值
    first_latencies = [
        m.first_chunk_latency_ms for m in successful if m.first_chunk_latency_ms is not None
    ]
    total_times = [
        m.total_synthesis_time_ms for m in successful if m.total_synthesis_time_ms is not None
    ]
    rtfs = [m.rtf for m in successful if m.rtf is not None]

    return AggregatedMetrics(
        test_type="tts",
        concurrency_level=concurrency_level,
        total_requests=len(metrics_list),
        successful_requests=len(successful),
        failed_requests=len(failed),
        total_test_time_seconds=total_test_time,
        # 首包延迟
        first_latency_avg=np.mean(first_latencies) if first_latencies else 0.0,
        first_latency_p50=calculate_percentile(first_latencies, 50),
        first_latency_p95=calculate_percentile(first_latencies, 95),
        first_latency_p99=calculate_percentile(first_latencies, 99),
        first_latency_max=max(first_latencies) if first_latencies else 0.0,
        # 总时间
        total_time_avg=np.mean(total_times) if total_times else 0.0,
        total_time_p50=calculate_percentile(total_times, 50),
        total_time_p95=calculate_percentile(total_times, 95),
        total_time_p99=calculate_percentile(total_times, 99),
        total_time_max=max(total_times) if total_times else 0.0,
        # RTF
        rtf_avg=np.mean(rtfs) if rtfs else 0.0,
        rtf_p50=calculate_percentile(rtfs, 50),
        rtf_p95=calculate_percentile(rtfs, 95),
        rtf_p99=calculate_percentile(rtfs, 99),
        rtf_max=max(rtfs) if rtfs else 0.0,
    )


def calculate_statistics(
    metrics_list: Union[List[ASRMetrics], List[TTSMetrics]],
    concurrency_level: int,
    total_test_time: float,
) -> AggregatedMetrics:
    """
    通用统计计算函数

    Args:
        metrics_list: 指标列表 (ASR 或 TTS)
        concurrency_level: 并发级别
        total_test_time: 总测试时间 (秒)

    Returns:
        聚合后的指标
    """
    if not metrics_list:
        raise ValueError("指标列表不能为空")

    if isinstance(metrics_list[0], ASRMetrics):
        return calculate_asr_statistics(metrics_list, concurrency_level, total_test_time)
    elif isinstance(metrics_list[0], TTSMetrics):
        return calculate_tts_statistics(metrics_list, concurrency_level, total_test_time)
    else:
        raise TypeError(f"不支持的指标类型: {type(metrics_list[0])}")
