# -*- coding: utf-8 -*-
"""
统一异常处理模块
定义所有自定义异常类和错误处理函数
"""

from fastapi import Request
from fastapi.responses import JSONResponse
import logging

logger = logging.getLogger(__name__)


class APIException(Exception):
    """API基础异常类"""

    def __init__(
        self, status_code: int, message: str, task_id: str = "", error_code: str = ""
    ):
        self.status_code = status_code
        self.message = message
        self.task_id = task_id
        self.error_code = error_code or self._get_error_code(status_code)
        super().__init__(self.message)

    def _get_error_code(self, status_code: int) -> str:
        """根据状态码获取错误代码"""
        code_mapping = {
            20000000: "SUCCESS",
            40000000: "DEFAULT_CLIENT_ERROR",
            40000001: "AUTHENTICATION_FAILED",
            40000002: "INVALID_MESSAGE",
            40000003: "INVALID_PARAMETER",
            40000004: "IDLE_TIMEOUT",
            40000005: "TOO_MANY_REQUESTS",
            40000010: "TRIAL_EXPIRED",
            41010101: "UNSUPPORTED_SAMPLE_RATE",
            50000000: "DEFAULT_SERVER_ERROR",
            50000001: "INTERNAL_GRPC_ERROR",
        }
        return code_mapping.get(status_code, "UNKNOWN_ERROR")


# 标准异常类
class AuthenticationException(APIException):
    """身份认证异常"""

    def __init__(self, message: str, task_id: str = ""):
        super().__init__(40000001, message, task_id)


class InvalidMessageException(APIException):
    """无效消息异常"""

    def __init__(self, message: str, task_id: str = ""):
        super().__init__(40000002, message, task_id)


class InvalidParameterException(APIException):
    """无效参数异常"""

    def __init__(self, message: str, task_id: str = ""):
        super().__init__(40000003, message, task_id)


class UnsupportedSampleRateException(APIException):
    """不支持的采样率异常"""

    def __init__(self, message: str, task_id: str = ""):
        super().__init__(41010101, message, task_id)


class DefaultServerErrorException(APIException):
    """默认服务端错误异常"""

    def __init__(self, message: str, task_id: str = ""):
        super().__init__(50000000, message, task_id)


# 异常处理器
async def api_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """API异常处理器"""
    # FastAPI 只会在抛出 APIException 时调用此处理器
    api_exc = exc if isinstance(exc, APIException) else APIException(50000000, str(exc))
    logger.error(f"[{api_exc.task_id}] API异常: {api_exc.message}")

    response_data = {
        "task_id": api_exc.task_id,
        "result": "",
        "status": api_exc.status_code,
        "message": api_exc.message,
    }

    return JSONResponse(
        content=response_data,
        headers={"task_id": api_exc.task_id} if api_exc.task_id else {},
        status_code=400 if api_exc.status_code >= 40000000 else 500,
    )


async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """通用异常处理器"""
    logger.error(f"未处理的异常: {str(exc)}", exc_info=True)

    response_data = {
        "task_id": "",
        "result": "",
        "status": 50000000,
        "message": f"内部服务错误: {str(exc)}",
    }

    return JSONResponse(content=response_data, status_code=500)
