# -*- coding: utf-8 -*-
"""
日志配置模块
统一的日志配置和管理，支持多 Worker 模式
"""

import logging
import logging.handlers
import sys
import os
from typing import Optional
from pathlib import Path
from .config import settings


def get_worker_id() -> str:
    """获取当前 Worker ID

    Returns:
        Worker 标识符，格式为 'worker-{pid}' 或 'main'
    """
    # 检查是否在多 worker 模式下
    workers = int(os.getenv("WORKERS", "1"))
    if workers > 1:
        return f"worker-{os.getpid()}"
    return "main"


def setup_logging(
    level: Optional[str] = None,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None,
    max_bytes: Optional[int] = None,
    backup_count: Optional[int] = None,
    worker_id: Optional[str] = None,
) -> None:
    """设置应用日志配置

    Args:
        level: 日志级别
        log_file: 日志文件路径
        format_string: 日志格式字符串
        max_bytes: 单个日志文件最大大小（字节）
        backup_count: 保留的备份文件数量
        worker_id: Worker 标识符（多 Worker 模式下使用）
    """
    # 使用传入的参数或配置文件中的设置
    log_level = level or settings.LOG_LEVEL
    log_file_path = log_file or settings.LOG_FILE
    max_file_size = max_bytes or settings.LOG_MAX_BYTES
    backup_files = backup_count or settings.LOG_BACKUP_COUNT

    # 获取 Worker ID
    current_worker_id = worker_id or get_worker_id()
    workers = int(os.getenv("WORKERS", "1"))
    worker_log_path: Optional[Path] = None

    # 多 Worker 模式下，日志格式包含 Worker ID
    if workers > 1:
        log_format = format_string or f"%(asctime)s - [{current_worker_id}] - %(name)s - %(levelname)s - %(message)s"
    else:
        log_format = format_string or "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # 创建处理器列表
    handlers = [logging.StreamHandler(sys.stdout)]

    # 确定日志文件路径
    if log_file_path:
        log_path = Path(log_file_path)
    else:
        log_path = Path("logs/funasr-api.log")

    # 确保日志目录存在
    log_dir = log_path.parent
    log_dir.mkdir(parents=True, exist_ok=True)

    # 多 Worker 模式下，每个 Worker 使用独立的日志文件
    if workers > 1:
        # 生成 worker 专属日志文件名: funasr-api.log -> funasr-api.worker-12345.log
        worker_log_path = log_dir / f"{log_path.stem}.{current_worker_id}{log_path.suffix}"

        # Worker 专属日志文件
        worker_file_handler = logging.handlers.RotatingFileHandler(
            worker_log_path,
            maxBytes=max_file_size,
            backupCount=backup_files,
            encoding="utf-8",
        )
        handlers.append(worker_file_handler)

        # 同时也写入主日志文件（汇总所有 Worker 的日志）
        main_file_handler = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=max_file_size,
            backupCount=backup_files,
            encoding="utf-8",
        )
        handlers.append(main_file_handler)
    else:
        # 单 Worker 模式，只写入主日志文件
        file_handler = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=max_file_size,
            backupCount=backup_files,
            encoding="utf-8",
        )
        handlers.append(file_handler)

    # 配置根日志记录器
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=handlers,
        force=True,  # 强制重新配置
    )

    # 设置第三方库的日志级别（由LOG_LEVEL控制）
    third_party_level = getattr(logging, log_level.upper())
    logging.getLogger("urllib3").setLevel(third_party_level)
    logging.getLogger("requests").setLevel(third_party_level)
    logging.getLogger("httpx").setLevel(third_party_level)
    logging.getLogger("httpcore").setLevel(third_party_level)

    # 始终禁用噪音特别大的库
    logging.getLogger("numba").setLevel(logging.WARNING)
    logging.getLogger("numba.core").setLevel(logging.WARNING)
    logging.getLogger("numba.core.ssa").setLevel(logging.WARNING)

    # 多 Worker 模式下记录启动日志
    if workers > 1 and worker_log_path:
        logger = logging.getLogger(__name__)
        logger.info(f"Worker {current_worker_id} 日志系统已初始化，日志文件: {worker_log_path}")


