#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FunASR-API Server 启动脚本"""

import sys
import os

# 强制离线模式，必须在任何 HF/transformers 导入前设置
# 注意：不要设置 HF_HUB_OFFLINE=1，否则 vLLM 会把 model_id 替换为绝对路径
os.environ.setdefault("HF_HUB_LOCAL_FILES_ONLY", "1")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()


def check_and_download_models() -> bool:
    """检查并下载缺失的模型"""
    try:
        from app.utils.download_models import check_all_models, download_models

        missing = check_all_models()
        if not missing:
            return True

        print(f"\n⚠️  检测到 {len(missing)} 个模型未下载")
        for mid in missing:
            print(f"  - {mid}")

        # 检测是否为交互式终端（Docker 环境跳过询问）
        if not sys.stdin.isatty():
            print("\n非交互式终端，自动下载模型...")
            return download_models(auto_mode=True)

        response = input("\n自动下载? [Y/n] ").strip().lower()
        if response in ("", "y", "yes"):
            success = download_models(auto_mode=True)
            print("✅ 下载完成" if success else "❌ 下载失败")
            return success
        else:
            print("⚠️  跳过下载，将在使用时下载")
            return False

    except Exception as e:
        print(f"⚠️  模型检查失败: {e}")
        return False


def main() -> None:
    """主入口"""
    from app.core.config import settings
    import uvicorn

    workers = int(os.getenv("WORKERS", "1"))

    print(f"🚀 FunASR-API | http://{settings.HOST}:{settings.PORT} | {settings.DEVICE}")

    if workers == 1:
        check_and_download_models()

        try:
            from app.utils.model_loader import (
                preload_models,
                print_model_statistics,
                verify_required_models_integrity,
            )

            integrity_result = verify_required_models_integrity(use_logger=False)
            invalid_models = integrity_result["invalid_models"]
            if invalid_models:
                print("❌ 检测到残缺模型，已阻止服务启动。")
                print("请修复模型缓存后重试。")
                sys.exit(1)

            result = preload_models()
            print_model_statistics(result, use_logger=False)
        except Exception as e:
            print(f"⚠️  预加载失败: {e}")
    else:
        print(f"多Worker模式({workers})，模型延迟加载")

    try:
        uvicorn.run(
            "app.main:app",
            host=settings.HOST,
            port=settings.PORT,
            workers=workers,
            reload=settings.DEBUG if workers == 1 else False,
            log_level="debug" if settings.DEBUG else settings.LOG_LEVEL.lower(),
            access_log=True,
        )
    except KeyboardInterrupt:
        print("\n已停止")
        sys.exit(0)
    except Exception as e:
        print(f"启动失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
