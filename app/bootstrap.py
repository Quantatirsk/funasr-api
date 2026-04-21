# -*- coding: utf-8 -*-
"""Shared bootstrap helpers for process startup."""

from __future__ import annotations

import sys


def ensure_models_downloaded(interactive: bool) -> bool:
    """Ensure declared deployment models already exist locally."""
    try:
        from app.utils.download_models import check_all_models

        missing = check_all_models()
        if not missing:
            return True

        print(f"\n⚠️  检测到 {len(missing)} 个模型未下载")
        for model_id, *_ in missing:
            print(f"  - {model_id}")

        print(
            "\n当前默认启用 HF_HUB_LOCAL_FILES_ONLY=1，启动前不会再自动联网下载模型。"
        )
        if interactive:
            print("请先运行以下命令准备模型：")
            print("  ./scripts/prepare-models.sh")
            print("或：")
            print("  uv run python -m app.utils.download_models")
        else:
            print("非交互式终端下请预先准备模型缓存后再启动。")
            print("推荐命令：uv run python -m app.utils.download_models")
        return False
    except Exception as exc:
        print(f"⚠️  模型检查失败: {exc}")
        return False


def run_cli_preflight() -> bool:
    """Preflight checks for the CLI entrypoint."""
    return ensure_models_downloaded(interactive=sys.stdin.isatty())
