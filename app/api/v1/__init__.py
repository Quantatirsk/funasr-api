# -*- coding: utf-8 -*-
"""API v1版本路由"""

from fastapi import APIRouter
from .asr import router as asr_router
from .websocket_asr import router as websocket_asr_router
from .websocket_qwen3_asr import router as qwen3_websocket_router
from .openai_compatible import router as openai_router

api_router = APIRouter()

# 原有 API (阿里云兼容)
api_router.include_router(asr_router)
api_router.include_router(websocket_asr_router)

# Qwen3-ASR 专用 WebSocket 流式端点 (POC)
api_router.include_router(qwen3_websocket_router)

# OpenAI 兼容 API
api_router.include_router(openai_router)
