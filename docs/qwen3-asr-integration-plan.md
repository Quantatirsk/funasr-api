# Qwen3-ASR 集成方案文档

## 1. 概述

### 1.1 目标
将 Qwen3-ASR（0.6B/1.7B）模型集成到现有 FunASR-API 项目中，支持：
- 阿里 ASR API 方案（/stream/v1/asr）
- OpenAI API 方案（/v1/audio/transcriptions）
- 通过切换模型参数使用不同后端进行 ASR 任务

### 1.2 Qwen3-ASR 特性
- **模型规格**: 0.6B 和 1.7B 两种规格
- **后端支持**: Transformers 和 vLLM（推荐 vLLM 以获得最佳性能）
- **语言能力**: 支持 52 种语言和方言的语音识别
- **推理模式**: 支持离线（Offline）和流式（Streaming）两种模式
- **时间戳**: 通过 Qwen3-ForcedAligner-0.6B 提供词/字级时间戳

## 2. 架构分析

### 2.1 当前项目架构
```
app/
├── services/asr/
│   ├── engine.py           # ASR引擎基类与FunASREngine实现
│   ├── manager.py          # 模型管理器（多模型缓存）
│   ├── loaders/
│   │   ├── base_loader.py  # 模型加载器基类
│   │   ├── loader_factory.py # 加载器工厂
│   │   ├── paraformer_loader.py
│   │   └── funasrnano_loader.py
│   └── models.json         # 模型配置文件
├── api/v1/
│   ├── asr.py              # 阿里ASR API
│   └── openai_compatible.py # OpenAI兼容API
```

### 2.2 Qwen3-ASR 架构
```
qwen_asr/
├── inference/
│   └── qwen3_asr.py        # 主要推理接口 Qwen3ASRModel
├── core/
│   ├── transformers_backend/  # Transformers后端
│   └── vllm_backend/       # vLLM后端
└── cli/
    └── serve.py            # vLLM服务启动
```

## 3. 集成方案

### 3.1 方案选择：独立 Qwen3ASREngine

**决策**: 创建新的 `Qwen3ASREngine` 类继承 `BaseASREngine`，而非复用 `FunASREngine`

**理由**:
1. Qwen3-ASR 使用 `qwen-asr` 库，与 FunASR 的 `AutoModel` 接口完全不同
2. Qwen3-ASR 内置 VAD、标点、语言识别，不需要外部模型组合
3. vLLM 后端需要特殊的初始化和推理方式
4. 保持代码清晰，避免在一个类中处理过多差异

### 3.2 类图设计

```
BaseASREngine (抽象基类)
    ├── FunASREngine (现有)
    │       └── 使用 AutoModel + 外部VAD/PUNC
    └── Qwen3ASREngine (新增)
            └── 使用 Qwen3ASRModel (vLLM/Transformers)

ModelManager
    └── 支持 FunASREngine 和 Qwen3ASREngine 的切换
```

### 3.3 核心实现类

#### 3.3.1 Qwen3ASREngine (app/services/asr/qwen3_engine.py)

```python
class Qwen3ASREngine(BaseASREngine):
    """Qwen3-ASR引擎，支持vLLM和Transformers后端"""

    def __init__(
        self,
        model_path: str,           # 如 "Qwen/Qwen3-ASR-1.7B"
        device: str = "auto",
        backend: str = "vllm",     # "vllm" 或 "transformers"
        gpu_memory_utilization: float = 0.8,
        forced_aligner_path: Optional[str] = None,  # 时间戳对齐模型
        **kwargs
    ):
        self.model = None  # Qwen3ASRModel 实例
        self.backend = backend
        # ... 初始化代码

    def transcribe_file(self, audio_path, ...) -> str:
        """单文件转写"""
        results = self.model.transcribe(
            audio=audio_path,
            language=language,  # None 表示自动检测
            return_time_stamps=False
        )
        return results[0].text

    def transcribe_file_with_vad(self, audio_path, ...) -> ASRRawResult:
        """带时间戳的转写（使用ForcedAligner）"""
        results = self.model.transcribe(
            audio=audio_path,
            return_time_stamps=True  # 启用时间戳
        )
        # 转换格式为 ASRRawResult
        # results[0].time_stamps 包含字/词级时间戳
```

#### 3.3.2 模型配置扩展 (models.json)

```json
{
    "models": {
        "qwen3-asr-1.7b": {
            "name": "Qwen3-ASR-1.7B",
            "engine": "qwen3",
            "description": "Qwen3-ASR 1.7B，支持52种语言和方言，vLLM后端",
            "languages": ["zh", "en", "yue", "ja", "ko", ...],
            "default": false,
            "supports_realtime": true,
            "models": {
                "offline": "Qwen/Qwen3-ASR-1.7B"
            },
            "extra_kwargs": {
                "backend": "vllm",
                "gpu_memory_utilization": 0.8,
                "forced_aligner": "Qwen/Qwen3-ForcedAligner-0.6B"
            }
        },
        "qwen3-asr-0.6b": {
            "name": "Qwen3-ASR-0.6B",
            "engine": "qwen3",
            "description": "Qwen3-ASR 0.6B轻量版，适合边缘部署",
            "languages": ["zh", "en", "yue", "ja", "ko", ...],
            "supports_realtime": true,
            "models": {
                "offline": "Qwen/Qwen3-ASR-0.6B"
            },
            "extra_kwargs": {
                "backend": "vllm",
                "gpu_memory_utilization": 0.6,
                "forced_aligner": "Qwen/Qwen3-ForcedAligner-0.6B"
            }
        }
    }
}
```

### 3.4 集成步骤

#### 步骤1: 添加依赖

```toml
# pyproject.toml 或 requirements.txt
[project.optional-dependencies]
qwen3 = [
    "qwen-asr[vllm]>=0.1.0",
    "flash-attn>=2.0",  # 可选，用于加速
]
```

#### 步骤2: 创建 Qwen3ASREngine

文件: `app/services/asr/qwen3_engine.py`

核心方法实现:
- `__init__`: 初始化 vLLM 或 Transformers 后端
- `transcribe_file`: 基础转写
- `transcribe_file_with_vad`: 带时间戳转写
- `transcribe_long_audio`: 长音频自动分段（复用基类或自定义）
- `transcribe_websocket`: 流式识别（vLLM backend only）

#### 步骤3: 扩展 ModelManager

修改 `app/services/asr/manager.py`:

```python
def _create_engine(self, config: ModelConfig) -> BaseASREngine:
    """根据配置创建ASR引擎"""
    engine_type = config.engine.lower()

    if engine_type == "funasr":
        return FunASREngine(...)
    elif engine_type == "qwen3":
        from .qwen3_engine import Qwen3ASREngine
        return Qwen3ASREngine(
            model_path=config.offline_model_path,
            device=settings.DEVICE,
            **config.extra_kwargs
        )
    else:
        raise InvalidParameterException(f"不支持的引擎类型: {config.engine}")
```

#### 步骤4: 更新 API 文档

- `app/api/v1/asr.py`: 更新模型列表枚举
- `app/api/v1/openai_compatible.py`: 更新模型列表

## 4. 关键实现细节

### 4.1 音频输入处理

Qwen3-ASR 支持多种音频输入格式：
- 本地文件路径
- URL
- Base64 data URL
- `(np.ndarray, sr)` 元组

我们的引擎需要统一转换为 Qwen3-ASR 支持的格式。

### 4.2 时间戳处理

Qwen3-ASR 通过 `Qwen3ForcedAligner` 提供时间戳：

```python
# 结果格式
results[0].time_stamps.items = [
    {"text": "字", "start_time": 0.0, "end_time": 0.3},
    {"text": "词", "start_time": 0.3, "end_time": 0.8},
    ...
]
```

需要转换为项目内部的 `ASRSegmentResult` 格式。

### 4.3 语言检测

Qwen3-ASR 自动检测语言，结果在 `results[0].language` 中：
- "Chinese"
- "English"
- "Chinese,English" (混合)

### 4.4 批处理优化

Qwen3-ASR vLLM 后端原生支持批处理：

```python
results = model.transcribe(
    audio=[audio1, audio2, audio3],  # 批量输入
    batch_size=32
)
```

可以利用这一点优化长音频分段处理。

### 4.5 流式识别

Qwen3-ASR 支持流式识别（仅 vLLM）：

```python
state = model.init_streaming_state(chunk_size_sec=2.0)
while has_audio:
    pcm_chunk = get_audio_chunk()
    state = model.streaming_transcribe(pcm_chunk, state)
    print(state.text)  # 当前累积结果
final_state = model.finish_streaming_transcribe(state)
```

## 5. 配置示例

### 5.1 环境变量

```bash
# 基础配置
ASR_MODEL_MODE=offline  # offline/realtime/all
DEFAULT_ASR_MODEL=qwen3-asr-1.7b

# Qwen3-ASR 特定配置
QWEN3_BACKEND=vllm  # vllm/transformers
QWEN3_GPU_MEMORY_UTILIZATION=0.8
QWEN3_FORCED_ALIGNER=Qwen/Qwen3-ForcedAligner-0.6B
```

### 5.2 Docker 部署

```dockerfile
# 基于官方vLLM镜像
FROM vllm/vllm-openai:latest

# 安装 qwen-asr
RUN pip install qwen-asr[vllm]

# 复制应用代码
COPY . /app
WORKDIR /app

# 启动命令
CMD ["python", "-m", "app.main"]
```

## 6. 使用示例

### 6.1 阿里 ASR API

```bash
# 使用 Qwen3-ASR-1.7B
curl -X POST "http://localhost:8000/stream/v1/asr?model_id=qwen3-asr-1.7b" \
    -H "X-NLS-Token: your-token" \
    -H "Content-Type: application/octet-stream" \
    --data-binary @audio.wav

# 使用 Qwen3-ASR-0.6B（轻量版）
curl -X POST "http://localhost:8000/stream/v1/asr?model_id=qwen3-asr-0.6b" \
    -H "X-NLS-Token: your-token" \
    --data-binary @audio.wav
```

### 6.2 OpenAI API

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="your-token"
)

# 使用 Qwen3-ASR
with open("audio.wav", "rb") as f:
    result = client.audio.transcriptions.create(
        model="qwen3-asr-1.7b",  # 或 qwen3-asr-0.6b
        file=f,
        response_format="verbose_json"
    )
    print(result.text)
```

## 7. 性能考虑

### 7.1 vLLM 后端优势
- **吞吐量**: 0.6B 模型在并发 128 时达到 2000x 实时率
- **延迟**: 首token延迟 < 100ms
- **显存**: 通过 gpu_memory_utilization 控制

### 7.2 模型选择建议
| 场景 | 推荐模型 | 理由 |
|------|---------|------|
| 高精度要求 | 1.7B | 最佳识别准确率 |
| 高并发/边缘 | 0.6B | 速度快，资源占用少 |
| 多语言 | 任意 | 都支持52种语言 |

## 8. 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| vLLM 版本兼容性 | 高 | 锁定 vLLM 版本，测试后再升级 |
| 显存不足 | 高 | 提供 gpu_memory_utilization 配置，支持 0.6B 降级 |
| 模型下载失败 | 中 | 预下载模型到本地缓存 |
| 与 FunASR 行为差异 | 低 | 统一接口封装，文档说明差异 |

## 9. 开发任务清单

- [ ] 创建 `Qwen3ASREngine` 类
- [ ] 实现基础转写方法 (`transcribe_file`)
- [ ] 实现带时间戳转写 (`transcribe_file_with_vad`)
- [ ] 实现长音频处理 (`transcribe_long_audio`)
- [ ] 扩展 `ModelManager` 支持 qwen3 引擎
- [ ] 更新 `models.json` 配置
- [ ] 更新 API 文档和枚举
- [ ] 添加单元测试
- [ ] 性能测试和优化
- [ ] 编写部署文档

## 10. 附录

### 10.1 参考链接
- Qwen3-ASR GitHub: https://github.com/QwenLM/Qwen3-ASR
- Qwen3-ASR 模型: https://huggingface.co/Qwen
- vLLM 文档: https://docs.vllm.ai/

### 10.2 模型ID映射

| 模型 | HuggingFace ID | ModelScope ID |
|------|---------------|---------------|
| Qwen3-ASR-1.7B | Qwen/Qwen3-ASR-1.7B | qwen/Qwen3-ASR-1.7B |
| Qwen3-ASR-0.6B | Qwen/Qwen3-ASR-0.6B | qwen/Qwen3-ASR-0.6B |
| Qwen3-ForcedAligner | Qwen/Qwen3-ForcedAligner-0.6B | qwen/Qwen3-ForcedAligner-0.6B |
