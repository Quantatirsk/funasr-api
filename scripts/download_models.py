#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型预下载脚本
用于构建 Docker 镜像时预下载所有模型
"""

import os
import sys
import urllib.request

# 设置环境变量，避免不必要的输出
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "0"  # 下载时显示进度

# 设置 ModelScope 缓存目录（如果未设置）
if "MODELSCOPE_CACHE" not in os.environ:
    # 根据系统环境自动选择缓存目录
    # Docker 容器内默认 /root，本地环境默认用户目录
    default_cache = os.path.expanduser("~/.cache/modelscope")
    os.environ["MODELSCOPE_CACHE"] = default_cache

# 需要额外下载远程代码的模型（ModelScope 不包含 model.py）
REMOTE_CODE_MODELS = {
    "FunAudioLLM/Fun-ASR-Nano-2512": {
        "url": "https://raw.githubusercontent.com/FunAudioLLM/Fun-ASR/main/model.py",
        "filename": "model.py",
    }
}


def download_remote_code(model_id: str, model_path: str) -> bool:
    """下载模型的远程代码文件（如 model.py）"""
    if model_id not in REMOTE_CODE_MODELS:
        return True

    config = REMOTE_CODE_MODELS[model_id]
    url = config["url"]
    filename = config["filename"]
    target_path = os.path.join(model_path, filename)

    # 如果文件已存在，跳过下载
    if os.path.exists(target_path):
        print(f"    ℹ️  {filename} 已存在，跳过下载")
        return True

    print(f"    📥 下载远程代码: {filename}")
    try:
        urllib.request.urlretrieve(url, target_path)
        print(f"    ✅ 远程代码下载完成: {target_path}")
        return True
    except Exception as e:
        print(f"    ❌ 远程代码下载失败: {e}")
        return False


def check_model_exists(model_id: str, cache_dir: str) -> tuple[bool, str]:
    """检查模型是否已存在于本地缓存

    Args:
        model_id: 模型ID
        cache_dir: 缓存目录

    Returns:
        (是否存在, 模型路径)
    """
    from pathlib import Path

    # ModelScope 的缓存结构有两种可能：
    # 1. cache_dir/hub/model_id/
    # 2. cache_dir/models/model_id/
    possible_paths = [
        Path(cache_dir) / "hub" / model_id,
        Path(cache_dir) / "models" / model_id,
    ]

    for model_path in possible_paths:
        if model_path.exists() and model_path.is_dir():
            # 检查是否有实际内容（至少有一个文件）
            if any(model_path.iterdir()):
                return True, str(model_path)

    return False, ""


def download_models():
    """下载所有需要的模型"""
    from modelscope.hub.snapshot_download import snapshot_download

    # 所有需要下载的模型列表 (ModelScope)
    models = [
        # === 1. 核心 ASR 模型 ===
        # Paraformer Large (默认模型) - 一体化版本，内置VAD+标点+时间戳
        ("iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch", "Paraformer Large 离线模型(VAD+标点)"),
        ("iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online", "Paraformer Large 实时模型"),
        # Fun-ASR-Nano - 轻量级多语言ASR，支持31种语言和中文方言
        ("FunAudioLLM/Fun-ASR-Nano-2512", "Fun-ASR-Nano(多语言+方言)"),

        # === 2. 音频预处理模型 ===
        # 语音活动检测(VAD)模型 - 检测语音段落
        ("iic/speech_fsmn_vad_zh-cn-16k-common-pytorch", "语音活动检测模型(VAD)"),
        # 说话人分离模型 (CAM++) - 多说话人场景
        ("iic/speech_campplus_speaker-diarization_common", "说话人分离模型(CAM++)"),
        # CAM++ 依赖的声纹识别模型
        ("damo/speech_campplus_sv_zh-cn_16k-common", "声纹识别模型(CAM++依赖)"),

        # === 3. 后处理模型 ===
        # 标点模型
        ("iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch", "标点符号模型(离线)"),
        ("iic/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727", "标点符号模型(实时)"),
        # 语言模型 (LM) - 用于提升识别准确率
        ("iic/speech_ngram_lm_zh-cn-ai-wesp-fst", "语言模型(N-gram LM)"),
    ]

    cache_dir = os.environ.get('MODELSCOPE_CACHE', os.path.expanduser('~/.cache/modelscope'))

    print("=" * 60)
    print("FunASR-API 模型预下载")
    print("=" * 60)
    print(f"模型缓存目录: {cache_dir}")
    print(f"待检查模型数: {len(models)}")
    print("=" * 60)

    failed = []
    skipped = []
    downloaded = []

    for i, (model_id, desc) in enumerate(models, 1):
        print(f"\n[{i}/{len(models)}] {desc}")
        print(f"    模型ID: {model_id}")

        # 检查模型是否已存在
        exists, existing_path = check_model_exists(model_id, cache_dir)
        if exists:
            print(f"    ⏭️  已存在，跳过下载: {existing_path}")
            skipped.append(model_id)

            # 仍然检查远程代码
            if not download_remote_code(model_id, existing_path):
                failed.append((model_id, "远程代码下载失败"))
            continue

        # 模型不存在，开始下载
        print(f"    📥 开始下载...")
        try:
            path = snapshot_download(model_id)
            print(f"    ✅ 下载完成: {path}")
            downloaded.append(model_id)

            # 下载远程代码（如果需要）
            if not download_remote_code(model_id, path):
                failed.append((model_id, "远程代码下载失败"))
        except Exception as e:
            print(f"    ❌ 下载失败: {e}")
            failed.append((model_id, str(e)))

    print("\n" + "=" * 60)
    print("📊 下载统计:")
    print(f"  ✅ 已下载: {len(downloaded)} 个")
    print(f"  ⏭️  已跳过: {len(skipped)} 个")
    print(f"  ❌ 失败: {len(failed)} 个")
    print("=" * 60)

    if failed:
        print(f"\n失败的模型:")
        for model_id, err in failed:
            print(f"  - {model_id}: {err}")
        sys.exit(1)
    else:
        print("\n✅ 所有模型准备就绪!")
    print("=" * 60)


if __name__ == "__main__":
    download_models()
