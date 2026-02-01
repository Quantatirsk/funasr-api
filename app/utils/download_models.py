#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型预下载脚本
用于构建 Docker 镜像时预下载所有模型

所有模型统一从 ModelScope 下载，使用默认缓存路径 ~/.cache/modelscope
"""

import os
from pathlib import Path

# === Qwen3-ASR 模型选择 ===
# auto = 检测显存自动选择 (<48G用0.6B, >=48G用1.7B)
# Qwen3-ASR-1.7B = 强制使用 1.7B
# Qwen3-ASR-0.6B = 强制使用 0.6B
QWEN_ASR_MODEL = os.getenv("QWEN_ASR_MODEL", "auto")


def _get_qwen_models() -> list[tuple[str, str]]:
    """根据配置返回要下载的 Qwen3-ASR 模型列表"""
    model_config = QWEN_ASR_MODEL

    # 强制指定模型
    if model_config == "Qwen3-ASR-1.7B":
        return [
            ("Qwen/Qwen3-ASR-1.7B", "Qwen3-ASR 1.7B (vLLM 后端，强制指定)"),
            ("Qwen/Qwen3-ForcedAligner-0.6B", "Qwen3-ForcedAligner 0.6B (时间戳对齐)"),
        ]
    elif model_config == "Qwen3-ASR-0.6B":
        return [
            ("Qwen/Qwen3-ASR-0.6B", "Qwen3-ASR 0.6B (vLLM 后端，轻量版，强制指定)"),
            ("Qwen/Qwen3-ForcedAligner-0.6B", "Qwen3-ForcedAligner 0.6B (时间戳对齐)"),
        ]
    else:  # auto 或其他值
        try:
            import torch

            if torch.cuda.is_available():
                total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                if total_vram >= 48:
                    print(f"检测到显存 {total_vram:.1f}GB >= 48GB，加载 Qwen3-ASR-1.7B")
                    return [
                        ("Qwen/Qwen3-ASR-1.7B", "Qwen3-ASR 1.7B (vLLM 后端，自动选择)"),
                        ("Qwen/Qwen3-ForcedAligner-0.6B", "Qwen3-ForcedAligner 0.6B (时间戳对齐)"),
                    ]
                else:
                    print(f"检测到显存 {total_vram:.1f}GB < 48GB，加载 Qwen3-ASR-0.6B")
                    return [
                        ("Qwen/Qwen3-ASR-0.6B", "Qwen3-ASR 0.6B (vLLM 后端，轻量版，自动选择)"),
                        ("Qwen/Qwen3-ForcedAligner-0.6B", "Qwen3-ForcedAligner 0.6B (时间戳对齐)"),
                    ]
            else:
                print("无 CUDA 设备，下载 Qwen3-ASR-0.6B (轻量版)")
                return [
                    ("Qwen/Qwen3-ASR-0.6B", "Qwen3-ASR 0.6B (vLLM 后端，轻量版，无GPU)"),
                    ("Qwen/Qwen3-ForcedAligner-0.6B", "Qwen3-ForcedAligner 0.6B (时间戳对齐)"),
                ]
        except ImportError:
            print("无法检测显存，默认下载 Qwen3-ASR-1.7B")
            return [
                ("Qwen/Qwen3-ASR-1.7B", "Qwen3-ASR 1.7B (vLLM 后端，默认)"),
                ("Qwen/Qwen3-ForcedAligner-0.6B", "Qwen3-ForcedAligner 0.6B (时间戳对齐)"),
            ]


# === 所有模型统一从 ModelScope 下载 ===
# 标准缓存路径: ~/.cache/modelscope/hub/models/{model_id}/
ALL_MODELS = [
    # Paraformer 模型
    ("iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch", "Paraformer Large 离线模型(VAD+标点)"),
    ("iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online", "Paraformer Large 实时模型"),
    ("iic/speech_fsmn_vad_zh-cn-16k-common-pytorch", "语音活动检测模型(VAD) - iic"),
    ("damo/speech_fsmn_vad_zh-cn-16k-common-pytorch", "语音活动检测模型(VAD) - damo"),
    ("iic/speech_campplus_speaker-diarization_common", "说话人分离模型(CAM++)"),
    ("damo/speech_campplus_sv_zh-cn_16k-common", "声纹识别模型(CAM++依赖)"),
    ("damo/speech_campplus-transformer_scl_zh-cn_16k-common", "CAM++ transformer模型(说话人分离依赖)"),
    ("iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch", "标点符号模型(离线)"),
    ("iic/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727", "标点符号模型(实时)"),
    ("iic/speech_ngram_lm_zh-cn-ai-wesp-fst", "语言模型(N-gram LM)"),
] + _get_qwen_models()


def check_model_exists(model_id: str) -> tuple[bool, str]:
    """检查模型是否已存在于本地缓存

    标准路径: ~/.cache/modelscope/hub/models/{model_id}/
    """
    from pathlib import Path

    try:
        cache_dir = Path.home() / ".cache" / "modelscope"
        model_path = cache_dir / "hub" / "models" / model_id

        if model_path.exists() and model_path.is_dir():
            if any(model_path.iterdir()):
                return True, str(model_path)
    except Exception:
        pass

    return False, ""


def check_all_models() -> list[str]:
    """检查所有模型是否存在

    Returns:
        缺失的模型ID列表
    """
    missing = []
    for model_id, _ in ALL_MODELS:
        exists, _ = check_model_exists(model_id)
        if not exists:
            missing.append(model_id)

    return missing


def download_models(auto_mode: bool = False) -> bool:
    """下载所有需要的模型

    Args:
        auto_mode: 如果为True，表示自动模式（从start.py调用），会简化输出

    Returns:
        是否全部下载成功
    """
    from modelscope.hub.snapshot_download import snapshot_download

    # 检查缺失的模型
    missing = check_all_models()

    if not missing:
        if not auto_mode:
            print("✅ 所有模型已存在，无需下载")
        return True

    cache_dir = Path.home() / ".cache" / "modelscope"

    if auto_mode:
        print(f"📦 检测到 {len(missing)} 个模型需要下载...")
    else:
        print("=" * 60)
        print("FunASR-API 模型预下载")
        print("=" * 60)
        print(f"ModelScope 缓存: {cache_dir}")
        print(f"待下载模型: {len(missing)} 个")
        print("=" * 60)

    failed = []
    downloaded = []

    # 下载所有模型（统一从 ModelScope）
    if missing:
        if not auto_mode:
            print("\n📦 开始下载 ModelScope 模型...")
            print("-" * 60)

        for i, (model_id, desc) in enumerate(ALL_MODELS, 1):
            if model_id not in missing:
                continue

            if not auto_mode:
                print(f"\n[{i}/{len(ALL_MODELS)}] {desc}")
                print(f"    模型ID: {model_id}")
                print(f"    📥 开始下载...", end="")

            try:
                # 使用 ModelScope 默认缓存路径
                path = snapshot_download(model_id)
                if not auto_mode:
                    print(f" ✅ 完成: {path}")
                downloaded.append(model_id)
            except Exception as e:
                if not auto_mode:
                    print(f" ❌ 失败: {e}")
                failed.append((model_id, str(e)))

    if not auto_mode:
        print("\n" + "=" * 60)
        print("📊 下载统计:")
        print(f"  ✅ 已下载: {len(downloaded)} 个")
        print(f"  ❌ 失败: {len(failed)} 个")
        print("=" * 60)

        if failed:
            print(f"\n失败的模型:")
            for model_id, err in failed:
                print(f"  - {model_id}: {err}")
            return False
        else:
            print("\n✅ 所有模型准备就绪!")
            print("=" * 60)

    return len(failed) == 0


if __name__ == "__main__":
    download_models()
