# -*- coding: utf-8 -*-
"""
WebSocket 客户端基类
"""

import json
import uuid
import logging
from abc import ABC, abstractmethod
from typing import Optional, Any, Dict

from websockets.legacy.client import connect, WebSocketClientProtocol  # type: ignore

logger = logging.getLogger(__name__)


class BaseWebSocketClient(ABC):
    """WebSocket 测试客户端基类"""

    def __init__(self, ws_url: str, timeout: float = 120.0):
        """
        初始化客户端

        Args:
            ws_url: WebSocket URL
            timeout: 超时时间 (秒)
        """
        self.ws_url = ws_url
        self.timeout = timeout
        self.websocket: Optional[WebSocketClientProtocol] = None
        self.task_id = self._generate_id()

    @staticmethod
    def _generate_id() -> str:
        """生成 32 位唯一 ID"""
        return str(uuid.uuid4()).replace("-", "")[:32]

    async def connect(self) -> None:
        """建立 WebSocket 连接"""
        self.websocket = await connect(
            self.ws_url,
            ping_interval=None,
            ping_timeout=None,
            max_size=10 * 1024 * 1024,  # 10MB
        )

    async def close(self) -> None:
        """关闭 WebSocket 连接"""
        if self.websocket:
            try:
                await self.websocket.close()
            except Exception:
                pass
            self.websocket = None

    async def send_json(self, data: Dict[str, Any]) -> None:
        """发送 JSON 消息"""
        if self.websocket:
            await self.websocket.send(json.dumps(data, ensure_ascii=False))

    async def send_bytes(self, data: bytes) -> None:
        """发送二进制数据"""
        if self.websocket:
            await self.websocket.send(data)

    async def receive(self) -> Any:
        """接收消息 (JSON 或二进制)"""
        if self.websocket:
            return await self.websocket.recv()
        return None

    async def receive_json(self) -> Optional[Dict[str, Any]]:
        """接收 JSON 消息"""
        data = await self.receive()
        if isinstance(data, str):
            return json.loads(data)
        return None

    async def wait_for_message(self, expected_name: str) -> Dict[str, Any]:
        """
        等待指定名称的消息

        Args:
            expected_name: 期望的消息名称

        Returns:
            消息数据

        Raises:
            Exception: 收到 TaskFailed 消息
        """
        while True:
            response = await self.receive()
            if isinstance(response, str):
                data = json.loads(response)
                header = data.get("header", {})
                name = header.get("name", "")

                if name == expected_name:
                    return data
                elif name == "TaskFailed":
                    status_text = header.get("status_text", "Unknown error")
                    raise Exception(f"TaskFailed: {status_text}")

    def _create_header(self, name: str, namespace: str) -> Dict[str, Any]:
        """
        创建消息头部

        Args:
            name: 消息名称
            namespace: 命名空间

        Returns:
            头部字典
        """
        return {
            "message_id": self._generate_id(),
            "task_id": self.task_id,
            "namespace": namespace,
            "name": name,
        }

    @abstractmethod
    async def run_test(self) -> Any:
        """
        执行测试

        Returns:
            测试指标
        """
        pass
