#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FunASR-API Server 启动脚本
简化的启动方式，适用于开发和测试
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 加载 .env 文件（如果存在）
from dotenv import load_dotenv
load_dotenv()

if __name__ == "__main__":
    from app.core.config import settings
    import uvicorn

    # 读取并发配置
    workers = int(os.getenv("WORKERS", "1"))
    thread_pool_size = os.getenv("INFERENCE_THREAD_POOL_SIZE", "auto")

    print("=" * 60)
    print("FunASR-API Server")
    print("=" * 60)
    print(f"服务地址: http://{settings.HOST}:{settings.PORT}")
    print(f"设备配置: {settings.DEVICE}")
    print(f"ASR模型模式: {settings.ASR_MODEL_MODE}")
    print(f"Worker进程数: {workers}")
    print(f"推理线程池: {thread_pool_size}")
    print(f"API文档: http://{settings.HOST}:{settings.PORT}/docs")
    print(
        f"ASR健康检查: http://{settings.HOST}:{settings.PORT}/stream/v1/asr/health"
    )
    print("=" * 60)

    # 预加载所有模型（仅在单worker或主进程时执行）
    # 多worker模式下，每个worker会在fork后自动加载模型
    if workers == 1:
        try:
            from app.utils.model_loader import preload_models, print_model_statistics

            preload_result = preload_models()

            # 打印详细的加载统计到控制台
            print()
            print_model_statistics(preload_result, use_logger=False)
            print()

        except Exception as e:
            print(f"\n模型预加载失败: {e}")
            print("服务将继续启动，模型将在首次使用时加载\n")
    else:
        print(f"\n多Worker模式({workers}个进程)，模型将在每个Worker启动时加载")
        print("提示: 多Worker会占用更多内存/显存，请确保资源充足\n")

    print("=" * 60)
    print("正在启动API服务器...")
    print("=" * 60)

    try:
        uvicorn.run(
            "app.main:app",
            host=settings.HOST,
            port=settings.PORT,
            workers=workers,
            reload=settings.DEBUG if workers == 1 else False,  # 多worker时禁用reload
            log_level="debug" if settings.DEBUG else settings.LOG_LEVEL.lower(),
            access_log=True,
        )
    except KeyboardInterrupt:
        print("\n服务已停止")
    except Exception as e:
        print(f"启动失败: {e}")
        sys.exit(1)
