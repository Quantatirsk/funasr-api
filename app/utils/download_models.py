#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型预下载脚本
用于构建 Docker 镜像时预下载所有模型
"""

import os

# 强制使用统一的模型缓存路径，避免 MODELSCOPE_CACHE 环境变量干扰
# 标准路径: ~/.cache/modelscope/hub/models/{model_id}/
MODELSCOPE_BASE_PATH = os.path.expanduser("~/.cache/modelscope")

# 设置 HuggingFace 缓存目录（如果未设置）
if "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = os.path.expanduser("~/.cache/huggingface")


# 模型版本控制已移除，全部使用 ModelScope 默认版本
MODEL_REVISIONS = {}

# === ModelScope 模型列表 ===
MODELSCOPE_MODELS = [
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
]

# === HuggingFace 模型列表 ===
HUGGINGFACE_MODELS = [
    ("Qwen/Qwen3-ASR-1.7B", "Qwen3-ASR 1.7B (vLLM 后端)"),
]


def check_model_exists(model_id: str, cache_dir: str) -> tuple[bool, str]:
    """检查 ModelScope 模型是否已存在于本地缓存

    标准路径: ~/.cache/modelscope/hub/models/{model_id}/
    """
    from pathlib import Path

    model_path = Path(cache_dir) / "hub" / "models" / model_id

    if model_path.exists() and model_path.is_dir():
        if any(model_path.iterdir()):
            return True, str(model_path)

    return False, ""


def check_hf_model_exists(model_id: str, cache_dir: str) -> tuple[bool, str]:
    """检查 HuggingFace 模型是否已存在于本地缓存"""
    from pathlib import Path

    org, name = model_id.split("/")
    model_path = Path(cache_dir) / "hub" / f"models--{org}--{name}"

    if model_path.exists() and model_path.is_dir():
        snapshots_dir = model_path / "snapshots"
        if snapshots_dir.exists() and any(snapshots_dir.iterdir()):
            return True, str(model_path)

    return False, ""


def check_all_models() -> tuple[list[str], list[str]]:
    """检查所有模型是否存在

    Returns:
        (missing_ms_models, missing_hf_models) - 缺失的模型ID列表
    """
    cache_dir = MODELSCOPE_BASE_PATH
    hf_cache_dir = os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface'))

    missing_ms = []
    for model_id, _ in MODELSCOPE_MODELS:
        exists, _ = check_model_exists(model_id, cache_dir)
        if not exists:
            missing_ms.append(model_id)

    missing_hf = []
    for model_id, _ in HUGGINGFACE_MODELS:
        exists, _ = check_hf_model_exists(model_id, hf_cache_dir)
        if not exists:
            missing_hf.append(model_id)

    return missing_ms, missing_hf


def download_models(auto_mode: bool = False) -> bool:
    """下载所有需要的模型

    Args:
        auto_mode: 如果为True，表示自动模式（从start.py调用），会简化输出

    Returns:
        是否全部下载成功
    """
    from modelscope.hub.snapshot_download import snapshot_download

    cache_dir = MODELSCOPE_BASE_PATH
    hf_cache_dir = os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface'))

    # 检查缺失的模型
    missing_ms, missing_hf = check_all_models()

    if not missing_ms and not missing_hf:
        if not auto_mode:
            print("✅ 所有模型已存在，无需下载")
        return True

    if auto_mode:
        print(f"📦 检测到 {len(missing_ms)} 个 ModelScope 模型、{len(missing_hf)} 个 HuggingFace 模型需要下载...")
    else:
        print("=" * 60)
        print("FunASR-API 模型预下载")
        print("=" * 60)
        print(f"ModelScope 缓存: {cache_dir}")
        print(f"HuggingFace 缓存: {hf_cache_dir}")
        print(f"待下载 ModelScope 模型: {len(missing_ms)} 个")
        print(f"待下载 HuggingFace 模型: {len(missing_hf)} 个")
        print("=" * 60)

    failed = []
    skipped = []
    downloaded = []

    # 下载 ModelScope 模型
    if missing_ms:
        if not auto_mode:
            print("\n📦 开始下载 ModelScope 模型...")
            print("-" * 60)

        for i, (model_id, desc) in enumerate(MODELSCOPE_MODELS, 1):
            if model_id not in missing_ms:
                continue

            if not auto_mode:
                print(f"\n[{i}/{len(MODELSCOPE_MODELS)}] {desc}")
                print(f"    模型ID: {model_id}")
                print(f"    📥 开始下载...", end="")

            try:
                # 显式指定缓存目录，确保下载到标准路径
                path = snapshot_download(model_id, cache_dir=MODELSCOPE_BASE_PATH)
                if not auto_mode:
                    print(f" ✅ 完成: {path}")
                downloaded.append(f"MS:{model_id}")
            except Exception as e:
                if not auto_mode:
                    print(f" ❌ 失败: {e}")
                failed.append((f"MS:{model_id}", str(e)))

    # 下载 HuggingFace 模型
    if missing_hf:
        if not auto_mode:
            print("\n📦 开始下载 HuggingFace 模型...")
            print("-" * 60)

        try:
            from huggingface_hub import snapshot_download as hf_snapshot_download
        except ImportError:
            print("⚠️  huggingface_hub 未安装，跳过 HuggingFace 模型下载")
            print("    如需下载，请运行: pip install huggingface_hub")
            hf_snapshot_download = None

        if hf_snapshot_download:
            for model_id, desc in HUGGINGFACE_MODELS:
                if model_id not in missing_hf:
                    continue

                if not auto_mode:
                    print(f"\n{desc}")
                    print(f"    模型ID: {model_id}")
                    print(f"    📥 开始下载...", end="")

                try:
                    path = hf_snapshot_download(model_id)
                    if not auto_mode:
                        print(f" ✅ 完成: {path}")
                    downloaded.append(f"HF:{model_id}")
                except Exception as e:
                    if not auto_mode:
                        print(f" ❌ 失败: {e}")
                    failed.append((f"HF:{model_id}", str(e)))

    if not auto_mode:
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
            return False
        else:
            print("\n✅ 所有模型准备就绪!")
            print("=" * 60)

    return len(failed) == 0


if __name__ == "__main__":
    download_models()
