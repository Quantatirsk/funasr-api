# -*- coding: utf-8 -*-
"""
OpenAI compatible API response streaming tests.
"""

import asyncio
import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace


def _install_dependency_stubs() -> None:
    class DummyRouter:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def get(self, *args, **kwargs):
            def decorator(func):
                return func

            return decorator

        def post(self, *args, **kwargs):
            def decorator(func):
                return func

            return decorator

    class DummyHTTPException(Exception):
        def __init__(self, status_code: int = 500, detail: str = "") -> None:
            super().__init__(detail)
            self.status_code = status_code
            self.detail = detail

    class DummyStreamingResponse:
        def __init__(self, body_iterator, media_type=None, headers=None) -> None:
            self.body_iterator = body_iterator
            self.media_type = media_type
            self.headers = headers or {}

    class DummyJSONResponse:
        def __init__(self, content=None, status_code: int = 200, media_type=None) -> None:
            self.content = content
            self.status_code = status_code
            self.media_type = media_type

    class DummyPlainTextResponse(DummyJSONResponse):
        pass

    class DummyBaseModel:
        def __init__(self, **kwargs) -> None:
            for key, value in kwargs.items():
                setattr(self, key, value)

        def model_dump(self):
            return self.__dict__.copy()

    def dummy_field(default=None, **kwargs):
        if "default_factory" in kwargs:
            return kwargs["default_factory"]()
        return default

    fastapi_module = types.ModuleType("fastapi")
    fastapi_module.APIRouter = DummyRouter
    fastapi_module.File = lambda default=None, **kwargs: default
    fastapi_module.Form = lambda default=None, **kwargs: default
    fastapi_module.UploadFile = type("UploadFile", (), {})
    fastapi_module.Request = type("Request", (), {})
    fastapi_module.HTTPException = DummyHTTPException
    sys.modules["fastapi"] = fastapi_module

    fastapi_responses_module = types.ModuleType("fastapi.responses")
    fastapi_responses_module.JSONResponse = DummyJSONResponse
    fastapi_responses_module.PlainTextResponse = DummyPlainTextResponse
    fastapi_responses_module.StreamingResponse = DummyStreamingResponse
    sys.modules["fastapi.responses"] = fastapi_responses_module

    pydantic_module = types.ModuleType("pydantic")
    pydantic_module.BaseModel = DummyBaseModel
    pydantic_module.Field = dummy_field
    sys.modules["pydantic"] = pydantic_module

    package_names = [
        "app",
        "app.api",
        "app.api.v1",
        "app.core",
        "app.services",
        "app.services.asr",
        "app.services.audio",
    ]
    for package_name in package_names:
        package_module = types.ModuleType(package_name)
        package_module.__path__ = []  # type: ignore[attr-defined]
        sys.modules.setdefault(package_name, package_module)

    config_module = types.ModuleType("app.core.config")
    config_module.settings = SimpleNamespace(MAX_AUDIO_SIZE=25 * 1024 * 1024)
    sys.modules["app.core.config"] = config_module

    executor_module = types.ModuleType("app.core.executor")
    executor_module.run_sync = lambda func, *args, **kwargs: func(*args, **kwargs)
    sys.modules["app.core.executor"] = executor_module

    security_module = types.ModuleType("app.core.security")
    security_module.validate_openai_token = lambda request: (True, None)
    sys.modules["app.core.security"] = security_module

    exceptions_module = types.ModuleType("app.core.exceptions")
    exceptions_module.InvalidParameterException = type(
        "InvalidParameterException", (Exception,), {}
    )
    exceptions_module.create_error_response = lambda **kwargs: kwargs
    sys.modules["app.core.exceptions"] = exceptions_module

    manager_module = types.ModuleType("app.services.asr.manager")
    manager_module.get_model_manager = lambda: None
    sys.modules["app.services.asr.manager"] = manager_module

    validators_module = types.ModuleType("app.services.asr.validators")
    validators_module._get_default_model = lambda: "paraformer-large"
    validators_module._get_dynamic_model_list = lambda: []
    sys.modules["app.services.asr.validators"] = validators_module

    audio_module = types.ModuleType("app.services.audio")
    audio_module.get_audio_service = lambda: None
    sys.modules["app.services.audio"] = audio_module


def _load_openai_compatible_module():
    _install_dependency_stubs()
    module_path = (
        Path(__file__).resolve().parents[1]
        / "app"
        / "api"
        / "v1"
        / "openai_compatible.py"
    )
    spec = importlib.util.spec_from_file_location(
        "app.api.v1.openai_compatible",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


async def _collect_stream_chunks(response) -> list[bytes]:
    chunks: list[bytes] = []
    async for chunk in response.body_iterator:
        chunks.append(chunk)
    return chunks


def test_heartbeat_stream_returns_immediately_when_inference_is_done() -> None:
    openai_compatible = _load_openai_compatible_module()
    cleanup_calls = 0

    async def inference_coro():
        return SimpleNamespace(text="hello", segments=[])

    def cleanup_callback() -> None:
        nonlocal cleanup_calls
        cleanup_calls += 1

    original_interval = openai_compatible.HEARTBEAT_INTERVAL_SECONDS
    openai_compatible.HEARTBEAT_INTERVAL_SECONDS = 10.0

    try:
        response = openai_compatible.create_heartbeat_streaming_response(
            response_format=openai_compatible.ResponseFormat.JSON,
            inference_coro=inference_coro(),
            audio_duration=1.0,
            language=None,
            cleanup_callback=cleanup_callback,
        )

        chunks = asyncio.run(_collect_stream_chunks(response))
    finally:
        openai_compatible.HEARTBEAT_INTERVAL_SECONDS = original_interval

    assert cleanup_calls == 1
    assert chunks == [json.dumps({"text": "hello"}, separators=(",", ":")).encode("utf-8")]


def test_heartbeat_stream_sends_heartbeat_while_waiting() -> None:
    openai_compatible = _load_openai_compatible_module()
    cleanup_calls = 0

    async def inference_coro():
        await asyncio.sleep(0.03)
        return SimpleNamespace(text="hello", segments=[])

    def cleanup_callback() -> None:
        nonlocal cleanup_calls
        cleanup_calls += 1

    original_interval = openai_compatible.HEARTBEAT_INTERVAL_SECONDS
    openai_compatible.HEARTBEAT_INTERVAL_SECONDS = 0.01

    try:
        response = openai_compatible.create_heartbeat_streaming_response(
            response_format=openai_compatible.ResponseFormat.JSON,
            inference_coro=inference_coro(),
            audio_duration=1.0,
            language=None,
            cleanup_callback=cleanup_callback,
        )

        chunks = asyncio.run(_collect_stream_chunks(response))
    finally:
        openai_compatible.HEARTBEAT_INTERVAL_SECONDS = original_interval

    assert cleanup_calls == 1
    assert any(chunk == b" \n" for chunk in chunks[:-1])
    assert chunks[-1] == json.dumps({"text": "hello"}, separators=(",", ":")).encode(
        "utf-8"
    )
