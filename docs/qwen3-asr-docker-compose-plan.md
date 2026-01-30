# Qwen3-ASR Docker Compose 部署方案

## 架构变更

### 原方案（内嵌 vLLM）
```
┌─────────────────────────────────────┐
│         funasr-api 容器              │
│  ┌───────────────────────────────┐  │
│  │   Qwen3ASREngine              │  │
│  │   └── Qwen3ASRModel.LLM()     │  │  ← vLLM 在进程内启动
│  └───────────────────────────────┘  │
└─────────────────────────────────────┘
```

### Docker Compose 方案（独立 vLLM）
```
┌─────────────────────┐      HTTP API      ┌─────────────────────────┐
│   funasr-api 容器    │ ◄────────────────► │   vllm-qwen3 容器       │
│  ┌───────────────┐  │   OpenAI API格式   │  ┌───────────────────┐  │
│  │ Qwen3ASREngine│  │                    │  │  vllm serve       │  │
│  │ └── HTTP 客户端│  │                    │  │  Qwen3-ASR-1.7B   │  │
│  └───────────────┘  │                    │  └───────────────────┘  │
└─────────────────────┘                    └─────────────────────────┘
```

## 方案对比

| 特性 | 内嵌 vLLM | 独立 vLLM (Docker Compose) |
|------|----------|---------------------------|
| 部署复杂度 | 低（单容器） | 中（多容器编排） |
| 扩展性 | 难（绑定在一起） | 易（vLLM 可独立扩缩容） |
| 资源隔离 | 无 | GPU 资源单独分配 |
| 故障恢复 | 需重启整个服务 | vLLM 可独立重启 |
| 版本升级 | 需重新构建镜像 | 独立更新 vLLM 镜像 |
| 适合场景 | 开发/小规模 | 生产/高并发 |

## 实现方案

### 1. docker-compose.yml

```yaml
version: '3.8'

services:
  # 主 API 服务
  funasr-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DEFAULT_ASR_MODEL=qwen3-asr-1.7b
      # vLLM 服务地址
      - QWEN3_VLLM_HOST=vllm-qwen3-1.7b
      - QWEN3_VLLM_PORT=8000
      # 可选：0.6B 版本
      - QWEN3_VLLM_HOST_0_6B=vllm-qwen3-0.6b
      - QWEN3_VLLM_PORT_0_6B=8000
    depends_on:
      - vllm-qwen3-1.7b
      - vllm-qwen3-0.6b
    networks:
      - asr-network

  # Qwen3-ASR-1.7B vLLM 服务
  vllm-qwen3-1.7b:
    image: vllm/vllm-openai:latest
    runtime: nvidia
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - NVIDIA_VISIBLE_DEVICES=0
    volumes:
      # 挂载模型缓存目录（避免重复下载）
      - ~/.cache/huggingface:/root/.cache/huggingface
      - ~/.cache/modelscope:/root/.cache/modelscope
    command: >
      --model Qwen/Qwen3-ASR-1.7B
      --gpu-memory-utilization 0.8
      --max-model-len 32768
      --port 8000
      --host 0.0.0.0
    ports:
      - "8001:8000"  # 可选：暴露给外部调试
    networks:
      - asr-network
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  # Qwen3-ASR-0.6B vLLM 服务（轻量版）
  vllm-qwen3-0.6b:
    image: vllm/vllm-openai:latest
    runtime: nvidia
    environment:
      - CUDA_VISIBLE_DEVICES=1  # 可分配到不同 GPU
      - NVIDIA_VISIBLE_DEVICES=1
    volumes:
      - ~/.cache/huggingface:/root/.cache/huggingface
      - ~/.cache/modelscope:/root/.cache/modelscope
    command: >
      --model Qwen/Qwen3-ASR-0.6B
      --gpu-memory-utilization 0.6
      --max-model-len 32768
      --port 8000
      --host 0.0.0.0
    networks:
      - asr-network
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

networks:
  asr-network:
    driver: bridge
```

### 2. Qwen3ASREngine（HTTP 客户端版）

```python
# app/services/asr/qwen3_http_engine.py
import os
import base64
import logging
from typing import Optional, List, Any
from urllib.parse import urljoin

import requests
import numpy as np

from .engine import BaseASREngine, ASRRawResult, ASRSegmentResult, ASRFullResult
from ...core.exceptions import DefaultServerErrorException

logger = logging.getLogger(__name__)


class Qwen3HTTPASREngine(BaseASREngine):
    """
    Qwen3-ASR HTTP 客户端引擎

    通过 OpenAI API 调用独立的 vLLM 服务
    """

    def __init__(
        self,
        model_path: str = "Qwen/Qwen3-ASR-1.7B",
        vllm_host: Optional[str] = None,
        vllm_port: int = 8000,
        api_key: str = "EMPTY",
        timeout: int = 300,
    ):
        self.model_path = model_path
        self.vllm_base_url = f"http://{vllm_host}:{vllm_port}/v1"
        self.api_key = api_key
        self.timeout = timeout
        self._device = "cuda"  # vLLM 服务端处理，客户端只标记

        # 测试连接
        self._health_check()

    def _health_check(self):
        """检查 vLLM 服务是否可用"""
        try:
            resp = requests.get(f"{self.vllm_base_url}/models", timeout=5)
            resp.raise_for_status()
            models = resp.json().get("data", [])
            model_ids = [m["id"] for m in models]
            logger.info(f"vLLM 服务可用，模型列表: {model_ids}")
        except Exception as e:
            logger.error(f"vLLM 服务连接失败: {e}")
            raise DefaultServerErrorException(f"无法连接到 vLLM 服务: {e}")

    def _prepare_audio(self, audio_input) -> dict:
        """
        准备音频数据格式

        支持: 文件路径 / URL / base64 / numpy 数组
        """
        # 文件路径
        if isinstance(audio_input, str):
            if audio_input.startswith("http://") or audio_input.startswith("https://"):
                # URL - 直接使用 audio_url
                return {"type": "audio_url", "audio_url": {"url": audio_input}}
            elif audio_input.startswith("data:"):
                # 已经是 base64 data URL
                return {"type": "audio_url", "audio_url": {"url": audio_input}}
            else:
                # 本地文件路径 - 转换为 base64
                with open(audio_input, "rb") as f:
                    audio_bytes = f.read()
                b64 = base64.b64encode(audio_bytes).decode("utf-8")
                data_url = f"data:audio/wav;base64,{b64}"
                return {"type": "audio_url", "audio_url": {"url": data_url}}

        # numpy 数组 (wav_data, sr)
        elif isinstance(audio_input, tuple) and len(audio_input) == 2:
            wav, sr = audio_input
            import io
            import soundfile as sf
            buffer = io.BytesIO()
            sf.write(buffer, wav, sr, format="WAV")
            b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
            data_url = f"data:audio/wav;base64,{b64}"
            return {"type": "audio_url", "audio_url": {"url": data_url}}

        else:
            raise ValueError(f"不支持的音频输入类型: {type(audio_input)}")

    def _call_vllm(self, audio_input, language: Optional[str] = None) -> dict:
        """
        调用 vLLM OpenAI API

        Qwen3-ASR 特殊格式:
        - 不加 language: 返回完整格式 "language XXX<asr_text>内容"
        - 加 language: 返回纯文本 "内容"
        """
        audio_content = self._prepare_audio(audio_input)

        # 构建消息
        messages = [{"role": "user", "content": [audio_content]}]

        # 如果指定了语言，添加到 prompt
        if language:
            # 在 assistant prompt 中强制语言
            # 注意：这需要特殊处理，vLLM chat template 可能不支持
            # 暂时使用通用调用，在后处理中提取
            pass

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        payload = {
            "model": self.model_path,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": 1024,
        }

        try:
            resp = requests.post(
                f"{self.vllm_base_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"vLLM API 调用失败: {e}")
            raise DefaultServerErrorException(f"ASR 服务调用失败: {e}")

    def _parse_result(self, response: dict) -> tuple[str, str]:
        """
        解析 Qwen3-ASR 输出格式

        格式: "language Chinese<asr_text>这里是识别内容"
        或: "language Chinese,English<asr_text>混合语言内容"
        """
        content = response["choices"][0]["message"]["content"]

        # 解析 language 和 text
        if "<asr_text>" in content:
            # 提取 language 部分
            lang_part = content.split("<asr_text>")[0].replace("language ", "").strip()
            text = content.split("<asr_text>")[1].strip()
            return lang_part, text
        else:
            # 纯文本输出（如果使用了 forced language）
            return "", content.strip()

    def transcribe_file(
        self,
        audio_path: str,
        hotwords: str = "",
        enable_punctuation: bool = True,
        enable_itn: bool = True,
        enable_vad: bool = False,
        sample_rate: int = 16000,
        language: Optional[str] = None,
    ) -> str:
        """单文件转写"""
        # hotwords 作为 context（Qwen3-ASR 支持）
        # 注意：vLLM OpenAI API 暂不支持 context 参数
        # 需要通过自定义 prompt 模板实现

        response = self._call_vllm(audio_path, language=language)
        _, text = self._parse_result(response)
        return text

    def transcribe_file_with_vad(
        self,
        audio_path: str,
        hotwords: str = "",
        enable_punctuation: bool = True,
        enable_itn: bool = True,
        sample_rate: int = 16000,
    ) -> ASRRawResult:
        """
        带时间戳转写

        限制：vLLM OpenAI API 不直接支持时间戳
        解决方案：
        1. 先获取 ASR 文本
        2. 调用独立的 ForcedAligner 服务（如果部署了）
        3. 或使用本地轻量级对齐模型
        """
        # 获取文本
        response = self._call_vllm(audio_path)
        lang, text = self._parse_result(response)

        # 简化处理：整段时间戳
        # 如果需要字级时间戳，需要额外部署 ForcedAligner 服务
        segments = [ASRSegmentResult(
            text=text,
            start_time=0.0,
            end_time=0.0,  # 未知，需要音频时长
        )]

        return ASRRawResult(text=text, segments=segments)

    def transcribe_long_audio(self, audio_path: str, **kwargs) -> ASRFullResult:
        """长音频转写"""
        # 复用基类的分段逻辑
        # 但每个片段通过 HTTP 调用
        return super().transcribe_long_audio(audio_path, **kwargs)

    def _transcribe_batch(
        self,
        segments: List[Any],
        **kwargs
    ) -> List[str]:
        """
        批量转写

        vLLM 支持 batch，但 OpenAI API 是逐个请求
        这里使用并发请求
        """
        import concurrent.futures

        def transcribe_single(seg):
            if not seg.temp_file:
                return ""
            try:
                return self.transcribe_file(seg.temp_file, **kwargs)
            except Exception as e:
                logger.error(f"批量转写片段失败: {e}")
                return ""

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(transcribe_single, segments))

        return results

    def is_model_loaded(self) -> bool:
        """检查 vLLM 服务是否可访问"""
        try:
            requests.get(f"{self.vllm_base_url}/models", timeout=2)
            return True
        except:
            return False

    @property
    def device(self) -> str:
        return self._device

    @property
    def supports_realtime(self) -> bool:
        return False  # HTTP 客户端不支持实时流式
```

### 3. Manager 配置

```python
# app/services/asr/manager.py

def _create_engine(self, config: ModelConfig) -> BaseASREngine:
    if config.engine == "funasr":
        return FunASREngine(...)

    elif config.engine == "qwen3":
        # 检测是 HTTP 模式还是内嵌模式
        vllm_host = config.extra_kwargs.get("vllm_host") or \
                   os.getenv(f"QWEN3_VLLM_HOST_{config.model_id.upper().replace('-', '_')}") or \
                   os.getenv("QWEN3_VLLM_HOST")

        if vllm_host:
            # Docker Compose 模式：HTTP 客户端
            from .qwen3_http_engine import Qwen3HTTPASREngine
            return Qwen3HTTPASREngine(
                model_path=config.offline_model_path,
                vllm_host=vllm_host,
                vllm_port=config.extra_kwargs.get("vllm_port", 8000),
            )
        else:
            # 内嵌模式：直接初始化 vLLM
            from .qwen3_engine import Qwen3ASREngine
            return Qwen3ASREngine(
                model_path=config.offline_model_path,
                device=settings.DEVICE,
                **config.extra_kwargs
            )
```

### 4. models.json 配置

```json
{
    "models": {
        "qwen3-asr-1.7b": {
            "name": "Qwen3-ASR-1.7B",
            "engine": "qwen3",
            "description": "Qwen3-ASR 1.7B via vLLM (Docker Compose)",
            "languages": ["zh", "en", "yue", "ja", "ko"],
            "default": false,
            "supports_realtime": false,
            "models": {
                "offline": "Qwen/Qwen3-ASR-1.7B"
            },
            "extra_kwargs": {
                "vllm_host": "vllm-qwen3-1.7b",
                "vllm_port": 8000
            }
        },
        "qwen3-asr-0.6b": {
            "name": "Qwen3-ASR-0.6B",
            "engine": "qwen3",
            "description": "Qwen3-ASR 0.6B via vLLM (Docker Compose)",
            "languages": ["zh", "en", "yue", "ja", "ko"],
            "supports_realtime": false,
            "models": {
                "offline": "Qwen/Qwen3-ASR-0.6B"
            },
            "extra_kwargs": {
                "vllm_host": "vllm-qwen3-0.6b",
                "vllm_port": 8000
            }
        }
    }
}
```

## 部署步骤

### 1. 启动服务

```bash
# 下载模型（可选，避免启动时下载）
modelscope download --model Qwen/Qwen3-ASR-1.7B --local_dir ~/.cache/modelscope/hub/Qwen/Qwen3-ASR-1.7B
modelscope download --model Qwen/Qwen3-ASR-0.6B --local_dir ~/.cache/modelscope/hub/Qwen/Qwen3-ASR-0.6B

# 启动
docker-compose up -d

# 查看日志
docker-compose logs -f vllm-qwen3-1.7b
docker-compose logs -f funasr-api
```

### 2. 验证服务

```bash
# 检查 vLLM 服务
curl http://localhost:8001/v1/models

# 检查主服务
curl http://localhost:8000/stream/v1/asr/health

# 测试 ASR
curl -X POST "http://localhost:8000/stream/v1/asr?model_id=qwen3-asr-1.7b" \
    --data-binary @test.wav
```

## 高级配置

### 多 GPU 配置

```yaml
# docker-compose.yml
services:
  vllm-qwen3-1.7b:
    image: vllm/vllm-openai:latest
    runtime: nvidia
    environment:
      - CUDA_VISIBLE_DEVICES=0,1  # 多卡
    command: >
      --model Qwen/Qwen3-ASR-1.7B
      --tensor-parallel-size 2    # TP=2
      --gpu-memory-utilization 0.9
```

### 使用官方 qwen-asr-serve

替代 `vllm serve`，使用 Qwen 官方包装器：

```yaml
services:
  vllm-qwen3-1.7b:
    image: qwenllm/qwen3-asr:latest
    command: >
      qwen-asr-serve Qwen/Qwen3-ASR-1.7B
      --gpu-memory-utilization 0.8
      --host 0.0.0.0
      --port 8000
```

### 独立 ForcedAligner 服务（时间戳）

如果需要字级时间戳，额外部署对齐服务：

```yaml
services:
  forced-aligner:
    image: qwenllm/qwen3-asr:latest
    command: >
      python -m qwen_asr.inference.qwen3_forced_aligner_server
      --model Qwen/Qwen3-ForcedAligner-0.6B
      --port 8000
```

然后在 `Qwen3HTTPASREngine` 中添加对齐调用。

## 两种模式的选择

| 场景 | 推荐方案 |
|------|----------|
| 开发/测试/小规模 | 内嵌 vLLM（简化方案） |
| 生产/高并发/多模型 | Docker Compose 独立 vLLM |
| 已有 vLLM 集群 | HTTP 客户端连接现有服务 |

## 风险与缓解

| 风险 | 缓解措施 |
|------|----------|
| 网络延迟 | 同机部署，使用 Docker 内部网络 |
| vLLM 服务故障 | 健康检查 + 自动重启策略 |
| 时间戳功能缺失 | 可选部署独立 ForcedAligner |
| 热词/context 传递 | 通过自定义 prompt 模板实现 |

## 实现任务

- [ ] `qwen3_http_engine.py` - HTTP 客户端引擎
- [ ] `docker-compose.yml` - vLLM 服务配置
- [ ] 更新 `manager.py` - 支持两种模式切换
- [ ] 更新 `models.json` - 添加 vllm_host 配置
- [ ] 健康检查和故障恢复
- [ ] 性能测试（延迟 vs 内嵌模式）
