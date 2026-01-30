# Qwen3-ASR 简化集成方案

## 核心发现

Qwen3-ASR vLLM 后端接口极其简洁：

```python
from qwen_asr import Qwen3ASRModel

# 初始化
asr = Qwen3ASRModel.LLM(
    model="Qwen/Qwen3-ASR-1.7B",
    gpu_memory_utilization=0.8,
    forced_aligner="Qwen/Qwen3-ForcedAligner-0.6B",  # 可选，时间戳
    max_inference_batch_size=32,
    max_new_tokens=1024,
)

# 推理（支持URL/base64/数组/批量）
results = asr.transcribe(
    audio="https://example.com/audio.wav",
    language=None,           # None=自动检测
    return_time_stamps=True, # True=返回字级时间戳
)

# 结果
print(results[0].language)  # "Chinese"
print(results[0].text)      # "识别文本"
print(results[0].time_stamps[0].text)       # "字"
print(results[0].time_stamps[0].start_time) # 0.0
```

## 简化方案

### 1. 最小实现：Qwen3ASREngine

```python
# app/services/asr/qwen3_engine.py
import torch
from typing import Optional, List
from qwen_asr import Qwen3ASRModel

from .engine import BaseASREngine, ASRRawResult, ASRSegmentResult, ASRFullResult

class Qwen3ASREngine(BaseASREngine):
    """Qwen3-ASR引擎 - vLLM后端"""

    def __init__(
        self,
        model_path: str = "Qwen/Qwen3-ASR-1.7B",
        device: str = "cuda:0",
        gpu_memory_utilization: float = 0.8,
        forced_aligner_path: Optional[str] = None,
        max_inference_batch_size: int = 32,
    ):
        self._device = device
        self.model = Qwen3ASRModel.LLM(
            model=model_path,
            gpu_memory_utilization=gpu_memory_utilization,
            forced_aligner=forced_aligner_path,
            forced_aligner_kwargs=dict(
                dtype=torch.bfloat16,
                device_map=device,
            ) if forced_aligner_path else None,
            max_inference_batch_size=max_inference_batch_size,
            max_new_tokens=1024,
        )

    def transcribe_file(
        self,
        audio_path: str,
        hotwords: str = "",
        enable_punctuation: bool = True,  # Qwen3内置，忽略参数
        enable_itn: bool = True,
        enable_vad: bool = False,         # Qwen3内置VAD
        sample_rate: int = 16000,
        language: Optional[str] = None,
    ) -> str:
        """单文件转写"""
        results = self.model.transcribe(
            audio=audio_path,
            context=hotwords if hotwords else "",
            language=language,
            return_time_stamps=False,
        )
        return results[0].text if results else ""

    def transcribe_file_with_vad(
        self,
        audio_path: str,
        hotwords: str = "",
        enable_punctuation: bool = True,
        enable_itn: bool = True,
        sample_rate: int = 16000,
    ) -> ASRRawResult:
        """带时间戳转写（使用ForcedAligner）"""
        results = self.model.transcribe(
            audio=audio_path,
            context=hotwords if hotwords else "",
            return_time_stamps=True,  # 启用时间戳
        )

        if not results:
            return ASRRawResult(text="", segments=[])

        r = results[0]
        segments = []

        # 转换时间戳格式
        if r.time_stamps and r.time_stamps.items:
            # 按句子分组（简单策略：按标点分句）
            current_text = ""
            current_start = r.time_stamps.items[0].start_time
            current_end = r.time_stamps.items[0].end_time

            for item in r.time_stamps.items:
                current_text += item.text
                current_end = item.end_time

                # 遇到标点分句
                if item.text in "。！？；\n":
                    segments.append(ASRSegmentResult(
                        text=current_text,
                        start_time=current_start,
                        end_time=current_end,
                    ))
                    current_text = ""
                    if item != r.time_stamps.items[-1]:
                        current_start = r.time_stamps.items[r.time_stamps.items.index(item) + 1].start_time

            # 添加最后一句
            if current_text:
                segments.append(ASRSegmentResult(
                    text=current_text,
                    start_time=current_start,
                    end_time=current_end,
                ))
        else:
            # 无时间戳时返回整段
            segments.append(ASRSegmentResult(
                text=r.text,
                start_time=0.0,
                end_time=0.0,
            ))

        return ASRRawResult(text=r.text, segments=segments)

    def _transcribe_batch(
        self,
        segments: List[Any],
        **kwargs
    ) -> List[str]:
        """批量转写 - 利用Qwen3原生批处理能力"""
        paths = [seg.temp_file for seg in segments if seg.temp_file]
        if not paths:
            return [""] * len(segments)

        results = self.model.transcribe(audio=paths, return_time_stamps=False)
        return [r.text if r.text else "" for r in results]

    def is_model_loaded(self) -> bool:
        return self.model is not None

    @property
    def device(self) -> str:
        return self._device

    @property
    def supports_realtime(self) -> bool:
        return False  # vLLM后端暂不支持streaming（可用vllm serve单独部署）
```

### 2. 配置简化

```json
// app/services/asr/models.json
{
    "models": {
        "qwen3-asr-1.7b": {
            "name": "Qwen3-ASR-1.7B",
            "engine": "qwen3",
            "description": "Qwen3-ASR 1.7B，52种语言，字级时间戳",
            "languages": ["zh", "en", "yue", "ja", "ko"],
            "default": false,
            "supports_realtime": false,
            "models": {
                "offline": "Qwen/Qwen3-ASR-1.7B"
            },
            "extra_kwargs": {
                "forced_aligner_path": "Qwen/Qwen3-ForcedAligner-0.6B",
                "gpu_memory_utilization": 0.8
            }
        },
        "qwen3-asr-0.6b": {
            "name": "Qwen3-ASR-0.6B",
            "engine": "qwen3",
            "description": "Qwen3-ASR 0.6B轻量版",
            "languages": ["zh", "en", "yue", "ja", "ko"],
            "supports_realtime": false,
            "models": {
                "offline": "Qwen/Qwen3-ASR-0.6B"
            },
            "extra_kwargs": {
                "forced_aligner_path": "Qwen/Qwen3-ForcedAligner-0.6B",
                "gpu_memory_utilization": 0.6
            }
        }
    }
}
```

### 3. Manager扩展（3行代码）

```python
# app/services/asr/manager.py

def _create_engine(self, config: ModelConfig) -> BaseASREngine:
    if config.engine == "funasr":
        return FunASREngine(...)
    elif config.engine == "qwen3":
        from .qwen3_engine import Qwen3ASREngine
        return Qwen3ASREngine(
            model_path=config.offline_model_path,
            device=settings.DEVICE,
            **config.extra_kwargs
        )
```

### 4. 依赖

```txt
# requirements.txt
qwen-asr[vllm]>=0.1.0
```

## 与原方案对比

| 方面 | 原方案 | 简化方案 |
|------|--------|----------|
| 代码量 | ~500行 | ~150行 |
| 后端支持 | vLLM + Transformers | 仅vLLM |
| Loader架构 | 复用现有Loader工厂 | 不需要，直接初始化 |
| 批处理 | 需单独实现 | 原生支持 |
| 时间戳 | 复杂后处理 | ForcedAligner内置 |
| 实时流式 | 计划实现 | 暂不支持（建议用vllm serve） |

## 关键设计决策

### 1. 只支持vLLM后端
**理由**：
- Transformers后端性能差（无PagedAttention）
- vLLM初始化简单（一行代码）
- 用户要求"直接用vLLM"

### 2. 不复用Loader工厂
**理由**：
- `Qwen3ASRModel.LLM()` 直接完成所有初始化
- 不需要像FunASR那样组合VAD/PUNC模型
- Loader模式反而增加复杂度

### 3. 时间戳处理策略
**Qwen3输出**：字/词级时间戳 `[{text, start_time, end_time}, ...]`

**转换策略**：
- 按标点符号分句（。！？；）
- 每句一个 segment
- 保留原始字级精度在 start_time/end_time

### 4. 流式识别
**当前**：暂不支持（vLLM后端streaming需要特殊处理）

**建议**：高并发流式场景直接用官方命令
```bash
qwen-asr-serve Qwen/Qwen3-ASR-1.7B --gpu-memory-utilization 0.8
```

或 vLLM serve：
```bash
vllm serve Qwen/Qwen3-ASR-1.7B
```

## 使用示例

### API调用
```bash
# 1.7B模型
curl -X POST "http://localhost:8000/stream/v1/asr?model_id=qwen3-asr-1.7b" \
    --data-binary @audio.wav

# 0.6B轻量版
curl -X POST "http://localhost:8000/stream/v1/asr?model_id=qwen3-asr-0.6b" \
    --data-binary @audio.wav
```

### Python
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="empty")

result = client.audio.transcriptions.create(
    model="qwen3-asr-1.7b",
    file=open("audio.wav", "rb"),
    response_format="verbose_json"
)
```

## 实现任务

- [ ] `app/services/asr/qwen3_engine.py` - 主引擎（150行）
- [ ] `app/services/asr/manager.py` - 添加qwen3分支（3行）
- [ ] `app/services/asr/models.json` - 添加模型配置
- [ ] `requirements.txt` - 添加 qwen-asr[vllm]
- [ ] 测试验证

**预估工作量**：2-3小时（主要在工作量测试）
