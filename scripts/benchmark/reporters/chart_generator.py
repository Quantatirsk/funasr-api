# -*- coding: utf-8 -*-
"""
Matplotlib 图表生成器
"""

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

from ..metrics.models import AggregatedMetrics

# 设置中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


class ChartGenerator:
    """图表生成器"""

    def __init__(self):
        self.colors = {
            "asr": "#4CAF50",  # 绿色
            "tts": "#2196F3",  # 蓝色
        }

    def generate_all_charts(
        self,
        asr_results: List[AggregatedMetrics],
        tts_results: List[AggregatedMetrics],
        output_dir: Path,
        timestamp: str,
    ) -> List[Path]:
        """
        生成所有图表

        Args:
            asr_results: ASR 测试结果
            tts_results: TTS 测试结果
            output_dir: 输出目录
            timestamp: 时间戳

        Returns:
            生成的图表文件路径列表
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        generated_files = []

        # 1. 首次延迟对比图
        if asr_results or tts_results:
            path = output_dir / f"first_latency_{timestamp}.png"
            self._generate_first_latency_chart(asr_results, tts_results, path)
            generated_files.append(path)

        # 2. RTF 对比图
        if asr_results or tts_results:
            path = output_dir / f"rtf_{timestamp}.png"
            self._generate_rtf_chart(asr_results, tts_results, path)
            generated_files.append(path)

        # 3. 吞吐量对比图
        if asr_results or tts_results:
            path = output_dir / f"throughput_{timestamp}.png"
            self._generate_throughput_chart(asr_results, tts_results, path)
            generated_files.append(path)

        # 4. 总时间对比图
        if asr_results or tts_results:
            path = output_dir / f"total_time_{timestamp}.png"
            self._generate_total_time_chart(asr_results, tts_results, path)
            generated_files.append(path)

        return generated_files

    def _generate_first_latency_chart(
        self,
        asr_results: List[AggregatedMetrics],
        tts_results: List[AggregatedMetrics],
        output_path: Path,
    ) -> None:
        """生成首次延迟对比图"""
        _fig, ax = plt.subplots(figsize=(10, 6))

        levels = []
        if asr_results:
            levels = [r.concurrency_level for r in asr_results]
            avg_values = [r.first_latency_avg for r in asr_results]
            p95_values = [r.first_latency_p95 for r in asr_results]

            ax.plot(levels, avg_values, 'o-', color=self.colors["asr"],
                   label='ASR 首次响应 (Avg)', linewidth=2, markersize=8)
            ax.plot(levels, p95_values, 's--', color=self.colors["asr"],
                   label='ASR 首次响应 (P95)', linewidth=1.5, markersize=6, alpha=0.7)

        if tts_results:
            levels = [r.concurrency_level for r in tts_results]
            avg_values = [r.first_latency_avg for r in tts_results]
            p95_values = [r.first_latency_p95 for r in tts_results]

            ax.plot(levels, avg_values, 'o-', color=self.colors["tts"],
                   label='TTS 首包延迟 (Avg)', linewidth=2, markersize=8)
            ax.plot(levels, p95_values, 's--', color=self.colors["tts"],
                   label='TTS 首包延迟 (P95)', linewidth=1.5, markersize=6, alpha=0.7)

        ax.set_xlabel('并发数', fontsize=12)
        ax.set_ylabel('延迟 (ms)', fontsize=12)
        ax.set_title('首次响应延迟 vs 并发数', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        if levels:
            ax.set_xticks(levels)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()

    def _generate_rtf_chart(
        self,
        asr_results: List[AggregatedMetrics],
        tts_results: List[AggregatedMetrics],
        output_path: Path,
    ) -> None:
        """生成 RTF 对比图"""
        _fig, ax = plt.subplots(figsize=(10, 6))

        if asr_results:
            levels = [r.concurrency_level for r in asr_results]
            avg_values = [r.rtf_avg for r in asr_results]
            p95_values = [r.rtf_p95 for r in asr_results]

            ax.plot(levels, avg_values, 'o-', color=self.colors["asr"],
                   label='ASR RTF (Avg)', linewidth=2, markersize=8)
            ax.plot(levels, p95_values, 's--', color=self.colors["asr"],
                   label='ASR RTF (P95)', linewidth=1.5, markersize=6, alpha=0.7)

        if tts_results:
            levels = [r.concurrency_level for r in tts_results]
            avg_values = [r.rtf_avg for r in tts_results]
            p95_values = [r.rtf_p95 for r in tts_results]

            ax.plot(levels, avg_values, 'o-', color=self.colors["tts"],
                   label='TTS RTF (Avg)', linewidth=2, markersize=8)
            ax.plot(levels, p95_values, 's--', color=self.colors["tts"],
                   label='TTS RTF (P95)', linewidth=1.5, markersize=6, alpha=0.7)

        # 添加 RTF=1.0 参考线
        all_levels = set()
        if asr_results:
            all_levels.update(r.concurrency_level for r in asr_results)
        if tts_results:
            all_levels.update(r.concurrency_level for r in tts_results)
        if all_levels:
            ax.axhline(y=1.0, color='red', linestyle=':', linewidth=1.5,
                      label='RTF = 1.0 (实时)')

        ax.set_xlabel('并发数', fontsize=12)
        ax.set_ylabel('RTF', fontsize=12)
        ax.set_title('RTF vs 并发数', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()

    def _generate_throughput_chart(
        self,
        asr_results: List[AggregatedMetrics],
        tts_results: List[AggregatedMetrics],
        output_path: Path,
    ) -> None:
        """生成吞吐量柱状图"""
        _fig, ax = plt.subplots(figsize=(10, 6))

        all_levels = sorted(set(
            [r.concurrency_level for r in asr_results] +
            [r.concurrency_level for r in tts_results]
        ))

        x = np.arange(len(all_levels))
        width = 0.35

        if asr_results:
            asr_throughput = []
            for level in all_levels:
                r = next((r for r in asr_results if r.concurrency_level == level), None)
                asr_throughput.append(r.throughput if r else 0)
            ax.bar(x - width/2, asr_throughput, width, label='ASR',
                  color=self.colors["asr"], alpha=0.8)

        if tts_results:
            tts_throughput = []
            for level in all_levels:
                r = next((r for r in tts_results if r.concurrency_level == level), None)
                tts_throughput.append(r.throughput if r else 0)
            ax.bar(x + width/2, tts_throughput, width, label='TTS',
                  color=self.colors["tts"], alpha=0.8)

        ax.set_xlabel('并发数', fontsize=12)
        ax.set_ylabel('吞吐量 (req/s)', fontsize=12)
        ax.set_title('吞吐量 vs 并发数', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([str(level) for level in all_levels])
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()

    def _generate_total_time_chart(
        self,
        asr_results: List[AggregatedMetrics],
        tts_results: List[AggregatedMetrics],
        output_path: Path,
    ) -> None:
        """生成总时间对比图"""
        _fig, ax = plt.subplots(figsize=(10, 6))

        if asr_results:
            levels = [r.concurrency_level for r in asr_results]
            avg_values = [r.total_time_avg for r in asr_results]
            p95_values = [r.total_time_p95 for r in asr_results]

            ax.plot(levels, avg_values, 'o-', color=self.colors["asr"],
                   label='ASR 总时间 (Avg)', linewidth=2, markersize=8)
            ax.plot(levels, p95_values, 's--', color=self.colors["asr"],
                   label='ASR 总时间 (P95)', linewidth=1.5, markersize=6, alpha=0.7)

        if tts_results:
            levels = [r.concurrency_level for r in tts_results]
            avg_values = [r.total_time_avg for r in tts_results]
            p95_values = [r.total_time_p95 for r in tts_results]

            ax.plot(levels, avg_values, 'o-', color=self.colors["tts"],
                   label='TTS 总时间 (Avg)', linewidth=2, markersize=8)
            ax.plot(levels, p95_values, 's--', color=self.colors["tts"],
                   label='TTS 总时间 (P95)', linewidth=1.5, markersize=6, alpha=0.7)

        ax.set_xlabel('并发数', fontsize=12)
        ax.set_ylabel('时间 (ms)', fontsize=12)
        ax.set_title('总处理时间 vs 并发数', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
