#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型预下载脚本
用于构建 Docker 镜像时预下载所有模型

- Paraformer 模型从 ModelScope 下载
- Qwen3-ASR 模型从 HuggingFace 下载 (vLLM 要求)
"""

import os
from pathlib import Path
from typing import Optional

from huggingface_hub import snapshot_download as hf_snapshot_download
from modelscope.hub.snapshot_download import snapshot_download as ms_snapshot_download

# === Qwen3-ASR 模型选择 ===
QWEN_ASR_MODEL = os.getenv("QWEN_ASR_MODEL", "auto")


def _get_qwen_models() -> list[tuple[str, str]]:
    """返回要从 HuggingFace 下载的 Qwen3-ASR 模型列表"""
    config = QWEN_ASR_MODEL

    # 强制指定模型
    if config == "Qwen3-ASR-1.7B":
        return [
            ("Qwen/Qwen3-ASR-1.7B", "Qwen3-ASR 1.7B"),
            ("Qwen/Qwen3-ForcedAligner-0.6B", "Qwen3-ForcedAligner"),
        ]
    elif config == "Qwen3-ASR-0.6B":
        return [
            ("Qwen/Qwen3-ASR-0.6B", "Qwen3-ASR 0.6B"),
            ("Qwen/Qwen3-ForcedAligner-0.6B", "Qwen3-ForcedAligner"),
        ]

    # auto 模式
    try:
        import torch
        if torch.cuda.is_available():
            vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if vram >= 32:
                print(f"显存 {vram:.1f}GB >= 32GB，加载 Qwen3-ASR-1.7B")
                return [("Qwen/Qwen3-ASR-1.7B", "Qwen3-ASR 1.7B"), ("Qwen/Qwen3-ForcedAligner-0.6B", "Qwen3-ForcedAligner")]
            else:
                print(f"显存 {vram:.1f}GB < 32GB，加载 Qwen3-ASR-0.6B")
                return [("Qwen/Qwen3-ASR-0.6B", "Qwen3-ASR 0.6B"), ("Qwen/Qwen3-ForcedAligner-0.6B", "Qwen3-ForcedAligner")]
        else:
            print("无 CUDA，跳过 Qwen3-ASR（vLLM 不支持 CPU）")
            return []  # CPU 模式下不下载 Qwen 模型
    except ImportError:
        print("默认下载 Qwen3-ASR-1.7B")
        return [("Qwen/Qwen3-ASR-1.7B", "Qwen3-ASR 1.7B"), ("Qwen/Qwen3-ForcedAligner-0.6B", "Qwen3-ForcedAligner")]


# === ModelScope 模型 (Paraformer) ===
# 注意：仅列出代码中实际使用的模型，避免重复下载
# 同一模型的不同命名（如 iic/ vs damo/）只保留实际使用的版本
# 格式: (model_id, description, revision)  revision 为 None 表示使用最新版本
MODELSCOPE_MODELS = [
    ("iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch", "Paraformer Large", None),
    ("iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online", "Paraformer Online", None),
    # VAD 模型: config.py 中使用的是 damo/ 版本
    # CAM++ 依赖 v2.0.2 版本，不是最新版
    ("damo/speech_fsmn_vad_zh-cn-16k-common-pytorch", "VAD", "v2.0.2"),
    # CAM++ 说话人分离: speaker_diarizer.py 中使用的是 iic/speech_campplus_speaker-diarization_common
    # 注意：以下两个 damo/ 模型是 CAM++ 的隐式依赖，FunASR 会自动下载，需预置避免运行时下载
    ("iic/speech_campplus_speaker-diarization_common", "CAM++", None),
    ("damo/speech_campplus_sv_zh-cn_16k-common", "CAM++ SV (隐式依赖)", None),
    ("damo/speech_campplus-transformer_scl_zh-cn_16k-common", "CAM++ Transformer (隐式依赖)", None),
    ("iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch", "标点模型", None),
    ("iic/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727", "标点模型(实时)", None),
    ("iic/speech_ngram_lm_zh-cn-ai-wesp-fst", "N-gram LM", None),
]

# === HuggingFace 模型 (Qwen3-ASR) ===
HF_MODELS = _get_qwen_models()

# === 合并所有模型（用于检查）===
ALL_MODELS = MODELSCOPE_MODELS + HF_MODELS


def _get_cache_path(model_id: str, source: str = "modelscope") -> Path:
    """获取模型缓存路径"""
    if source == "huggingface":
        # HF 缓存格式: ~/.cache/huggingface/hub/models--{org}--{model}/
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        org, model = model_id.split("/", 1)
        model_path = cache_dir / f"models--{org}--{model}"
    else:
        cache_dir = Path.home() / ".cache" / "modelscope"
        model_path = cache_dir / "hub" / "models" / model_id
    return model_path


def check_model_exists(model_id: str, source: str = "modelscope") -> tuple[bool, str]:
    """检查模型是否已存在于本地缓存"""
    try:
        model_path = _get_cache_path(model_id, source)

        if model_path.exists() and model_path.is_dir():
            if any(model_path.iterdir()):
                return True, str(model_path)
    except Exception:
        pass

    return False, ""


def check_all_models() -> list[tuple[str, str, str, Optional[str]]]:
    """检查所有模型是否存在

    Returns:
        缺失的模型列表，每个元素为 (model_id, description, source, revision)
    """
    missing = []

    # 检查 ModelScope 模型
    for item in MODELSCOPE_MODELS:
        model_id, desc, revision = item
        exists, _ = check_model_exists(model_id, source="modelscope")
        if not exists:
            missing.append((model_id, desc, "modelscope", revision))

    # 检查 HuggingFace 模型 (HF 模型暂不支持指定版本)
    for model_id, desc in HF_MODELS:
        exists, _ = check_model_exists(model_id, source="huggingface")
        if not exists:
            missing.append((model_id, desc, "huggingface", None))

    return missing


def download_models(auto_mode: bool = False) -> bool:
    """下载所有需要的模型

    Args:
        auto_mode: 如果为True，表示自动模式（从start.py调用），会简化输出

    Returns:
        是否全部下载成功
    """
    # 检查缺失的模型
    missing = check_all_models()

    if not missing:
        if not auto_mode:
            print("✅ 所有模型已存在，无需下载")
        return True

    ms_cache_dir = Path.home() / ".cache" / "modelscope"
    hf_cache_dir = Path.home() / ".cache" / "huggingface"

    if auto_mode:
        print(f"📦 检测到 {len(missing)} 个模型需要下载...")
    else:
        print("=" * 60)
        print("FunASR-API 模型预下载")
        print("=" * 60)
        print(f"ModelScope 缓存: {ms_cache_dir}")
        print(f"HuggingFace 缓存: {hf_cache_dir}")
        print(f"待下载模型: {len(missing)} 个")
        print("=" * 60)

    failed = []
    downloaded = []

    # 下载 ModelScope 模型 (Paraformer)
    ms_missing = [(mid, desc, rev) for mid, desc, src, rev in missing if src == "modelscope"]
    if ms_missing:
        if not auto_mode:
            print("\n📦 开始下载 ModelScope 模型 (Paraformer)...")
            print("-" * 60)

        for i, (model_id, desc, revision) in enumerate(ms_missing, 1):
            if not auto_mode:
                print(f"\n[{i}/{len(ms_missing)}] {desc}")
                print(f"    模型ID: {model_id}")
                if revision:
                    print(f"    版本: {revision}")
                print(f"    📥 开始下载...", end="")

            try:
                # 传递版本参数，如果指定了版本
                if revision:
                    path = ms_snapshot_download(model_id, revision=revision)
                else:
                    path = ms_snapshot_download(model_id)
                if not auto_mode:
                    print(f" ✅ 完成: {path}")
                downloaded.append(model_id)
            except Exception as e:
                if not auto_mode:
                    print(f" ❌ 失败: {e}")
                failed.append((model_id, str(e)))

    # 下载 HuggingFace 模型 (Qwen3-ASR)
    hf_missing = [(mid, desc, rev) for mid, desc, src, rev in missing if src == "huggingface"]
    if hf_missing:
        if not auto_mode:
            print("\n📦 开始下载 HuggingFace 模型 (Qwen3-ASR)...")
            print("-" * 60)

        for i, (model_id, desc, _) in enumerate(hf_missing, 1):
            if not auto_mode:
                print(f"\n[{i}/{len(hf_missing)}] {desc}")
                print(f"    模型ID: {model_id}")
                print(f"    📥 开始下载...", end="")

            try:
                path = hf_snapshot_download(model_id)
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
