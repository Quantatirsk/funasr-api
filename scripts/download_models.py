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
os.environ["MODELSCOPE_CACHE"] = "/root/.cache/modelscope"

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


def download_models():
    """下载所有需要的模型"""
    from modelscope.hub.snapshot_download import snapshot_download

    # 所有需要下载的模型列表 (ModelScope)
    models = [
        # === 1. 核心 ASR 模型 ===
        # Paraformer Large (默认模型) - 一体化版本，内置VAD+标点+时间戳
        ("iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch", "Paraformer Large Offline (VAD+PUNC)"),
        ("iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online", "Paraformer Large Online/Realtime"),
        # Fun-ASR-Nano - 轻量级多语言ASR，支持31种语言和中文方言
        ("FunAudioLLM/Fun-ASR-Nano-2512", "Fun-ASR-Nano (多语言+方言)"),

        # === 2. 音频预处理模型 ===
        # VAD 模型 - 语音活动检测
        ("iic/speech_fsmn_vad_zh-cn-16k-common-pytorch", "VAD Model"),
        # 说话人分离模型 (CAM++) - 多说话人场景
        ("iic/speech_campplus_speaker-diarization_common", "Speaker Diarization Model (CAM++)"),

        # === 3. 后处理模型 ===
        # 标点模型
        ("iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch", "Punctuation Model"),
        ("iic/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727", "Realtime Punctuation Model"),
        # 语言模型 (LM) - 用于提升识别准确率
        ("iic/speech_ngram_lm_zh-cn-ai-wesp-fst", "Language Model (N-gram LM)"),
    ]

    print("=" * 60)
    print("FunASR-API 模型预下载")
    print("=" * 60)
    print(f"模型缓存目录: {os.environ['MODELSCOPE_CACHE']}")
    print(f"待下载模型数: {len(models)}")
    print("=" * 60)

    failed = []
    for i, (model_id, desc) in enumerate(models, 1):
        print(f"\n[{i}/{len(models)}] 下载: {desc}")
        print(f"    模型ID: {model_id}")
        try:
            path = snapshot_download(model_id)
            print(f"    ✅ 完成: {path}")

            # 下载远程代码（如果需要）
            if not download_remote_code(model_id, path):
                failed.append((model_id, "远程代码下载失败"))
        except Exception as e:
            print(f"    ❌ 失败: {e}")
            failed.append((model_id, str(e)))

    print("\n" + "=" * 60)
    if failed:
        print(f"下载完成，{len(failed)} 个模型失败:")
        for model_id, err in failed:
            print(f"  - {model_id}: {err}")
        sys.exit(1)
    else:
        print("✅ 所有模型下载完成!")
    print("=" * 60)


if __name__ == "__main__":
    download_models()
