# -*- coding: utf-8 -*-
"""
WebSocket ASR API路由
"""

import logging
import time
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from ...services.websocket_asr import get_aliyun_websocket_asr_service

logger = logging.getLogger(__name__)

# 创建路由器
router = APIRouter(prefix="/ws/v1/asr", tags=["WebSocket ASR"])


@router.websocket("")
async def aliyun_websocket_asr_endpoint(websocket: WebSocket):
    """阿里云WebSocket实时ASR端点"""
    await websocket.accept()
    service = get_aliyun_websocket_asr_service()
    task_id = f"aliyun_ws_asr_{int(time.time())}_{id(websocket)}"

    try:
        await service._process_websocket_connection(websocket, task_id)
    except WebSocketDisconnect:
        logger.info(f"[{task_id}] 客户端断开连接")
    except Exception as e:
        logger.error(f"[{task_id}] 连接处理异常: {e}")
    finally:
        try:
            await websocket.close()
        except:
            pass


@router.get("/test", response_class=HTMLResponse)
async def websocket_asr_test_page():
    """阿里云WebSocket ASR测试页面"""
    # 从模板文件读取HTML内容
    import os
    template_path = os.path.join(os.path.dirname(__file__), "..", "..", "templates", "aliyun_asr_test.html")
    with open(template_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)
