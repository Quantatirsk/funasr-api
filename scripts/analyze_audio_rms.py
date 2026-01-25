#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
音频RMS时序分析工具

用于分析音频���件的RMS能量，帮助确定远场过滤的阈值。
支持立体声、左声道、右声道选择。
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from typing import Optional

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_audio(file_path: str, channel: str = 'stereo') -> tuple:
    """加载音频文件

    Args:
        file_path: 音频文件路径
        channel: 声道选择 ('stereo', 'left', 'right')

    Returns:
        (audio_data, sample_rate): 音频数据和采样率
    """
    file_ext = Path(file_path).suffix.lower()

    if file_ext == '.wav':
        import wave
        with wave.open(file_path, 'rb') as wav_file:
            sample_rate = wav_file.getframerate()
            n_channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            n_frames = wav_file.getnframes()

            # 读取音频数据
            audio_bytes = wav_file.readframes(n_frames)

            # 转换为numpy数组
            if sample_width == 2:  # 16-bit
                audio_int = np.frombuffer(audio_bytes, dtype=np.int16)
            elif sample_width == 4:  # 32-bit
                audio_int = np.frombuffer(audio_bytes, dtype=np.int32)
            else:
                raise ValueError(f"不支持的采样位深: {sample_width}")

            # 转换为float32 (-1.0 to 1.0)
            audio_float = audio_int.astype(np.float32) / (2 ** (8 * sample_width - 1))

            # 处理多声道
            if n_channels > 1:
                audio_float = audio_float.reshape(-1, n_channels)
                if channel == 'left':
                    audio_float = audio_float[:, 0]
                    print(f"✓ 使用左声道")
                elif channel == 'right':
                    audio_float = audio_float[:, 1]
                    print(f"✓ 使用右声道")
                else:  # stereo - 平均
                    audio_float = np.mean(audio_float, axis=1)
                    print(f"✓ 使用立体声（双声道平均）")
            else:
                print(f"✓ 使用单声道")

            return audio_float, sample_rate

    else:
        # 尝试使用 soundfile 或 librosa
        try:
            import soundfile as sf
            audio_float, sample_rate = sf.read(file_path)

            if len(audio_float.shape) > 1:  # 多声道
                if channel == 'left':
                    audio_float = audio_float[:, 0]
                    print(f"✓ 使用左声道")
                elif channel == 'right':
                    audio_float = audio_float[:, 1]
                    print(f"✓ 使用右声道")
                else:
                    audio_float = np.mean(audio_float, axis=1)
                    print(f"✓ 使用立体声（双声道平均）")
            else:
                print(f"✓ 使用单声道")

            return audio_float, sample_rate

        except ImportError:
            print("错误: 请安装 soundfile 库: pip install soundfile")
            sys.exit(1)


def calculate_rms_energy(audio_array: np.ndarray) -> float:
    """计算音频RMS能量

    Args:
        audio_array: float32音频数组，范围-1.0到1.0

    Returns:
        RMS能量值
    """
    if len(audio_array) == 0:
        return 0.0
    return float(np.sqrt(np.mean(audio_array ** 2)))


def analyze_rms_timeline(audio_data: np.ndarray, sample_rate: int,
                         chunk_size_ms: int = 240) -> tuple:
    """分析音频的RMS时序

    Args:
        audio_data: 音频数据
        sample_rate: 采样率
        chunk_size_ms: 分块大小（毫秒）

    Returns:
        (time_points, rms_values): 时间点和对应的RMS值
    """
    chunk_samples = int(sample_rate * chunk_size_ms / 1000)
    n_chunks = len(audio_data) // chunk_samples

    time_points = []
    rms_values = []

    for i in range(n_chunks):
        start_idx = i * chunk_samples
        end_idx = start_idx + chunk_samples
        chunk = audio_data[start_idx:end_idx]

        rms = calculate_rms_energy(chunk)
        time_s = (start_idx + chunk_samples / 2) / sample_rate

        time_points.append(time_s)
        rms_values.append(rms)

    return np.array(time_points), np.array(rms_values)


def print_statistics(rms_values: np.ndarray, threshold: float = 0.01):
    """打印RMS统计信息

    Args:
        rms_values: RMS值数组
        threshold: 阈值
    """
    print("\n" + "="*60)
    print("RMS 统计分析")
    print("="*60)

    print(f"\n📊 基础统计:")
    print(f"  - 最小值: {np.min(rms_values):.6f}")
    print(f"  - 最大值: {np.max(rms_values):.6f}")
    print(f"  - 平均值: {np.mean(rms_values):.6f}")
    print(f"  - 中位数: {np.median(rms_values):.6f}")
    print(f"  - 标准差: {np.std(rms_values):.6f}")

    print(f"\n📈 百分位数:")
    for p in [10, 25, 50, 75, 90, 95, 99]:
        value = np.percentile(rms_values, p)
        print(f"  - P{p:2d}: {value:.6f}")

    print(f"\n🎯 阈值分析 (当前阈值: {threshold:.6f}):")
    above_threshold = np.sum(rms_values >= threshold)
    below_threshold = np.sum(rms_values < threshold)
    total = len(rms_values)

    print(f"  - 超过阈值的帧数: {above_threshold} ({above_threshold/total*100:.1f}%)")
    print(f"  - 低于阈值的帧数: {below_threshold} ({below_threshold/total*100:.1f}%)")

    print(f"\n💡 建议的阈值范围:")
    # 基于非零RMS值的统计
    non_zero_rms = rms_values[rms_values > 0.001]
    if len(non_zero_rms) > 0:
        p10 = np.percentile(non_zero_rms, 10)
        p25 = np.percentile(non_zero_rms, 25)
        mean = np.mean(non_zero_rms)

        print(f"  - 保守模式 (高灵敏度): {p10:.6f} (P10)")
        print(f"  - 宽松模式 (推荐):     {p25:.6f} (P25)")
        print(f"  - 严格模式 (低误触):   {mean*0.5:.6f} (平均值的50%)")

    print("="*60 + "\n")


def plot_rms_timeline(time_points: np.ndarray, rms_values: np.ndarray,
                      threshold: float = 0.01, save_path: Optional[str] = None):
    """绘制RMS时序图

    Args:
        time_points: 时间点数组
        rms_values: RMS值数组
        threshold: 阈值线
        save_path: 保存路径
    """
    _, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # 上图: RMS时序
    ax1.plot(time_points, rms_values, linewidth=1, label='RMS Energy', color='steelblue')
    ax1.axhline(y=threshold, color='red', linestyle='--', linewidth=2,
                label=f'阈值 = {threshold:.6f}')

    # 标记超过阈值的区域
    above_threshold = rms_values >= threshold
    ax1.fill_between(time_points, 0, rms_values, where=above_threshold,
                     alpha=0.3, color='green', label='近场音频 (>= 阈值)')
    ax1.fill_between(time_points, 0, rms_values, where=~above_threshold,
                     alpha=0.3, color='red', label='远场音频 (< 阈值)')

    ax1.set_xlabel('时间 (秒)', fontsize=12)
    ax1.set_ylabel('RMS 能量', fontsize=12)
    ax1.set_title('音频 RMS 能量时序分析', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)

    # 下图: RMS分布直方图
    ax2.hist(rms_values, bins=100, color='steelblue', alpha=0.7, edgecolor='black')
    ax2.axvline(x=threshold, color='red', linestyle='--', linewidth=2,
                label=f'阈值 = {threshold:.6f}')
    ax2.axvline(x=np.mean(rms_values), color='orange', linestyle=':', linewidth=2,
                label=f'平均值 = {np.mean(rms_values):.6f}')
    ax2.axvline(x=np.median(rms_values), color='green', linestyle=':', linewidth=2,
                label=f'中位数 = {np.median(rms_values):.6f}')

    ax2.set_xlabel('RMS 能量', fontsize=12)
    ax2.set_ylabel('帧数', fontsize=12)
    ax2.set_title('RMS 能量分布直方图', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ 图表已保存到: {save_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='音频RMS时序分析工具 - 帮助确定远场过滤阈值',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 分析立体声音频（默认）
  python analyze_audio_rms.py audio.wav

  # 仅分析左声道
  python analyze_audio_rms.py audio.wav --channel left

  # 仅分析右声道
  python analyze_audio_rms.py audio.wav --channel right

  # 自定义阈值和分块大小
  python analyze_audio_rms.py audio.wav --threshold 0.015 --chunk-size 160

  # 保存图表
  python analyze_audio_rms.py audio.wav --output rms_analysis.png
        """
    )

    parser.add_argument('audio_file', type=str,
                       help='音频文件路径 (支持 WAV, MP3, FLAC 等格式)')
    parser.add_argument('--channel', type=str, choices=['stereo', 'left', 'right'],
                       default='stereo',
                       help='声道选择: stereo(立体声平均), left(左声道), right(右声道) [默认: stereo]')
    parser.add_argument('--threshold', type=float, default=0.01,
                       help='RMS能量阈值 [默认: 0.01]')
    parser.add_argument('--chunk-size', type=int, default=240,
                       help='分块大小(毫秒) [默认: 240ms，与流式ASR一致]')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='保存图表的路径 (例如: output.png)')
    parser.add_argument('--no-plot', action='store_true',
                       help='不显示图表，仅输出统计信息')

    args = parser.parse_args()

    # 检查文件是否存在
    if not Path(args.audio_file).exists():
        print(f"错误: 文件不存在: {args.audio_file}")
        sys.exit(1)

    print("="*60)
    print("音频 RMS 时序分析工具")
    print("="*60)
    print(f"\n📁 文件: {args.audio_file}")
    print(f"🎚️  声道: {args.channel}")
    print(f"📊 分块大小: {args.chunk_size}ms")
    print(f"🎯 阈值: {args.threshold:.6f}")
    print()

    # 加载音频
    print("正在加载音频...")
    audio_data, sample_rate = load_audio(args.audio_file, args.channel)
    duration = len(audio_data) / sample_rate

    print(f"✓ 采样率: {sample_rate} Hz")
    print(f"✓ 时长: {duration:.2f} 秒")
    print(f"✓ 样本数: {len(audio_data)}")

    # 分析RMS时序
    print(f"\n正在分析 RMS 时序 (分块大小: {args.chunk_size}ms)...")
    time_points, rms_values = analyze_rms_timeline(audio_data, sample_rate, args.chunk_size)
    print(f"✓ 分析了 {len(rms_values)} 个音频块")

    # 打印统计信息
    print_statistics(rms_values, args.threshold)

    # 绘制图表
    if not args.no_plot:
        print("正在生成图表...")
        plot_rms_timeline(time_points, rms_values, args.threshold, args.output)
    elif args.output:
        print("正在保存图表...")
        plot_rms_timeline(time_points, rms_values, args.threshold, args.output)
        # 关闭显示窗口
        plt.close()


if __name__ == '__main__':
    main()
