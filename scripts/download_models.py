#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型预下载脚本
用于构建 Docker 镜像时预下载所有模型
"""

import os
import sys

# 设置环境变量，避免不必要的输出
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "0"  # 下载时显示进度
os.environ["MODELSCOPE_CACHE"] = "/root/.cache/modelscope"


def download_models():
    """下载所有需要的模型"""
    from modelscope.hub.snapshot_download import snapshot_download

    # 所有需要下载的模型列表
    models = [
        # Paraformer Large (默认模型)
        ("iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch", "Paraformer Large Offline"),
        ("iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online", "Paraformer Large Online/Realtime"),
        # SenseVoice Small
        ("iic/SenseVoiceSmall", "SenseVoice Small"),
        # VAD 模型
        ("iic/speech_fsmn_vad_zh-cn-16k-common-pytorch", "VAD Model"),
        # 标点模型
        ("iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch", "Punctuation Model"),
        ("iic/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727", "Realtime Punctuation Model"),
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
