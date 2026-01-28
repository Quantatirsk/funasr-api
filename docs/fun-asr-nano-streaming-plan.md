# Fun-ASR-Nano 流式识别适配方案

基于方案 B：统一端点，参数切换模式。

## 背景

Fun-ASR 上游仓库 (FunAudioLLM/Fun-ASR) 在 2025-01-25 的提交 `c38d22f` 中添加了 Fun-ASR-Nano 模型的流式输出支持。本计划旨在将该功能适配到 funasr-api 项目中。

## 目标

- 在现有 `/ws/v1/asr` WebSocket 端点支持 Fun-ASR-Nano 流式识别
- 通过 `model_id` 参数自动选择流式策略
- 保持向后兼容，默认使用 Paraformer 模型

## 两种流式模式对比

| 特性       | Paraformer 模式    | Fun-ASR-Nano 模式        |
| ---------- | ------------------ | ------------------------ |
| 音频处理   | 分块处理（无状态） | 累积处理（有状态）       |
| 上下文传递 | 模型内部 cache     | prev_text 参数           |
| 延迟       | 低延迟 (~240ms)    | 中等延迟 (~720ms)        |
| 语言支持   | 中文为主           | 31 种语言 + 7 种中文方言 |
| 内存占用   | 低（仅当前块）     | 高（累积完整音频）       |

## 实现计划

### 阶段 1：基础设施准备

#### 1.1 加载 Fun-ASR-Nano tokenizer

**文件**: `app/services/asr/engine.py`

```python
# 新增全局 tokenizer 缓存
_global_nano_tokenizer = None
_nano_tokenizer_lock = threading.Lock()

def get_global_nano_tokenizer():
    """获取 Fun-ASR-Nano tokenizer（懒加载单例）"""
    global _global_nano_tokenizer
    with _nano_tokenizer_lock:
        if _global_nano_tokenizer is None:
            from transformers import AutoTokenizer
            _global_nano_tokenizer = AutoTokenizer.from_pretrained(
                "FunAudioLLM/Fun-ASR-Nano-2512",
                trust_remote_code=True
            )
    return _global_nano_tokenizer
```

#### 1.2 扩展 ASR 引擎接口

**文件**: `app/services/asr/engine.py`

在 `FunASREngine` 类中新增：

```python
def transcribe_streaming_nano(
    self,
    audio_array: np.ndarray,
    prev_text: str = "",
    is_final: bool = False,
) -> tuple[str, str]:
    """
    Fun-ASR-Nano 流式识别

    Args:
        audio_array: 累积音频数据（从开始到当前）
        prev_text: 上一次识别的文本（用于上下文续写）
        is_final: 是否为最后一帧

    Returns:
        (完整文本, 用于下次的 prev_text)
    """
    pass
```

### 阶段 2：WebSocket 协议扩展

#### 2.1 扩展 StartTranscription 参数

**文件**: `app/models/websocket_asr.py`

```python
# 新增支持的模型 ID
class WebSocketASRModel(str, Enum):
    PARAFORMER = "paraformer-large"
    FUN_ASR_NANO = "fun-asr-nano"
```

**文件**: `app/services/websocket_asr.py`

在 `_parse_start_transcription` 中新增：

```python
params = {
    # ... 现有参数 ...
    "model_id": payload.get("model_id", "paraformer-large"),  # 新增
}
```

#### 2.2 新增音频累积缓存

```python
# 在 _process_websocket_connection 中
cumulative_audio_buffer = np.array([], dtype=np.float32)  # Fun-ASR-Nano 累积缓存
nano_prev_text = ""  # Fun-ASR-Nano 上下文文本
```

### 阶段 3：流式策略实现

#### 3.1 策略选择逻辑

**文件**: `app/services/websocket_asr.py`

```python
async def _process_audio_for_model(
    self,
    audio_chunk: np.ndarray,
    params: dict,
    context: dict,
    task_id: str,
) -> tuple[str, bool, dict]:
    """
    根据模型选择流式策略

    Returns:
        (识别文本, 是否句子结束, 更新后的上下文)
    """
    model_id = params.get("model_id", "paraformer-large")

    if model_id == "fun-asr-nano":
        return await self._process_audio_nano(audio_chunk, params, context, task_id)
    else:
        return await self._process_audio_paraformer(audio_chunk, params, context, task_id)
```

#### 3.2 Fun-ASR-Nano 流式处理

```python
async def _process_audio_nano(
    self,
    audio_chunk: np.ndarray,
    params: dict,
    context: dict,
    task_id: str,
) -> tuple[str, bool, dict]:
    """Fun-ASR-Nano 累积式流式处理"""

    # 累积音频
    cumulative_audio = context.get("cumulative_audio", np.array([], dtype=np.float32))
    cumulative_audio = np.concatenate([cumulative_audio, audio_chunk])

    # 检查是否达到处理阈值（0.72 秒 = 11520 samples @ 16kHz）
    NANO_CHUNK_SAMPLES = 11520
    if len(cumulative_audio) < NANO_CHUNK_SAMPLES:
        context["cumulative_audio"] = cumulative_audio
        return "", False, context

    # 获取 tokenizer
    tokenizer = get_global_nano_tokenizer()
    prev_text = context.get("prev_text", "")

    # 调用 Fun-ASR-Nano 推理
    result_text = await self._inference_nano(
        cumulative_audio,
        prev_text,
        task_id
    )

    # 非最终帧：截断最后 5 个 token
    if not context.get("is_final", False):
        tokens = tokenizer.encode(result_text)
        if len(tokens) > 5:
            result_text = tokenizer.decode(tokens[:-5]).replace("�", "")

    # 更新上下文
    context["cumulative_audio"] = cumulative_audio
    context["prev_text"] = result_text

    return result_text, False, context
```

### 阶段 4：句子边界检测适配

#### 4.1 Fun-ASR-Nano 句子结束判断

Fun-ASR-Nano 是端到端模型，句子边界检测策略：

1. **静音检测**：连续 N 帧低能量音频
2. **文本稳定**：连续 M 次推理结果相同
3. **标点符号**：检测到句末标点

```python
def _detect_nano_sentence_end(
    self,
    current_text: str,
    prev_texts: list[str],
    audio_energy: float,
) -> bool:
    """Fun-ASR-Nano 句子结束检测"""

    # 方法1：文本稳定（连续3次相同）
    if len(prev_texts) >= 3:
        if all(t == current_text for t in prev_texts[-3:]):
            return True

    # 方法2：句末标点
    if current_text and current_text[-1] in "。！？.!?":
        return True

    # 方法3：静音检测
    if audio_energy < 0.001:
        return True

    return False
```

### 阶段 5：配置与环境变量

#### 5.1 新增配置项

**文件**: `app/core/config.py`

```python
# Fun-ASR-Nano 流式配置
ASR_NANO_CHUNK_DURATION: float = Field(
    default=0.72,
    description="Fun-ASR-Nano 流式处理的 chunk 时长（秒）"
)
ASR_NANO_TOKEN_TRUNCATE: int = Field(
    default=5,
    description="Fun-ASR-Nano 中间结果截断的 token 数量"
)
ASR_NANO_STABLE_COUNT: int = Field(
    default=3,
    description="Fun-ASR-Nano 文本稳定判断的连续次数"
)
```

### 阶段 6：测试与验证

#### 6.1 单元测试

- [ ] Fun-ASR-Nano tokenizer 加载测试
- [ ] 累积音频缓存测试
- [ ] prev_text 上下文传递测试
- [ ] token 截断测试

#### 6.2 集成测试

- [ ] WebSocket 连接 + Fun-ASR-Nano 模型选择
- [ ] 多语言流式识别测试（中/英/日/韩）
- [ ] 长音频流式识别稳定性测试
- [ ] 内存占用监控

#### 6.3 性能基准

| 指标     | Paraformer | Fun-ASR-Nano | 目标        |
| -------- | ---------- | ------------ | ----------- |
| 首字延迟 | ~300ms     | ~800ms       | <1s         |
| 吞吐量   | 10x RT     | 5x RT        | >3x RT      |
| 内存增长 | 稳定       | 线性增长     | <500MB/分钟 |

## 文件变更清单

| 文件                              | 变更类型 | 说明                          |
| --------------------------------- | -------- | ----------------------------- |
| `app/services/asr/engine.py`    | 修改     | 添加 tokenizer 和流式推理方法 |
| `app/services/websocket_asr.py` | 修改     | 添加模型选择和累积处理逻辑    |
| `app/models/websocket_asr.py`   | 修改     | 扩展协议参数                  |
| `app/core/config.py`            | 修改     | 新增配置项                    |
| `scripts/download_models.py`    | 修改     | 确保下载 tokenizer            |
| `tests/test_websocket_nano.py`  | 新增     | Fun-ASR-Nano 流式测试         |

## 客户端使用示例

### StartTranscription 请求

```json
{
  "header": {
    "namespace": "SpeechTranscriber",
    "name": "StartTranscription",
    "task_id": "xxx"
  },
  "payload": {
    "format": "pcm",
    "sample_rate": 16000,
    "model_id": "fun-asr-nano",
    "enable_intermediate_result": true
  }
}
```

### 响应示例

```json
{
  "header": {
    "name": "TranscriptionResultChanged",
    "task_id": "xxx"
  },
  "payload": {
    "result": "Hello, how are you",
    "model": "fun-asr-nano"
  }
}
```

## 风险与缓解

| 风险           | 影响 | 缓解措施                          |
| -------------- | ---- | --------------------------------- |
| 内存溢出       | 高   | 设置最大音频时长限制（如 5 分钟） |
| 延迟过高       | 中   | 优化 chunk 大小，支持配置调整     |
| 模型加载慢     | 中   | 启动时预加载，支持懒加载          |
| tokenizer 依赖 | 低   | 使用 transformers 库，已在依赖中  |

## 时间估算

| 阶段           | 工作内容           | 预估           |
| -------------- | ------------------ | -------------- |
| 阶段 1         | 基础设施准备       | 1 天           |
| 阶段 2         | WebSocket 协议扩展 | 0.5 天         |
| 阶段 3         | 流式策略实现       | 2 天           |
| 阶段 4         | 句子边界检测       | 1 天           |
| 阶段 5         | 配置与环境变量     | 0.5 天         |
| 阶段 6         | 测试与验证         | 2 天           |
| **总计** |                    | **7 天** |

## 参考资料

- [Fun-ASR 流式提交](https://github.com/FunAudioLLM/Fun-ASR/commit/c38d22f)
- [Fun-ASR-Nano 模型](https://modelscope.cn/models/FunAudioLLM/Fun-ASR-Nano-2512)
- [FunASR 文档](https://github.com/alibaba-damo-academy/FunASR)
