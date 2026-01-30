# Qwen3-ASR 使用说明

## 已实现功能

✅ Qwen3-ASR-1.7B 和 Qwen3-ASR-0.6B 模型支持
✅ vLLM 内嵌后端（高性能）
✅ 字级时间戳（通过 ForcedAligner）
✅ 52 种语言和方言自动检测
✅ 批量推理优化
✅ 阿里 ASR API 兼容
✅ OpenAI API 兼容

## 安装依赖

```bash
pip install -r requirements.txt
```

注意：`qwen-asr[vllm]` 需要 vLLM 作为依赖，建议在有 CUDA 的环境中安装。

## 模型配置

`app/services/asr/models.json` 已添加以下模型：

```json
{
    "qwen3-asr-1.7b": {
        "name": "Qwen3-ASR-1.7B",
        "engine": "qwen3",
        "models": {"offline": "Qwen/Qwen3-ASR-1.7B"},
        "extra_kwargs": {
            "gpu_memory_utilization": 0.8,
            "forced_aligner_path": "Qwen/Qwen3-ForcedAligner-0.6B"
        }
    },
    "qwen3-asr-0.6b": {
        "name": "Qwen3-ASR-0.6B",
        "engine": "qwen3",
        "models": {"offline": "Qwen/Qwen3-ASR-0.6B"},
        "extra_kwargs": {
            "gpu_memory_utilization": 0.6,
            "forced_aligner_path": "Qwen/Qwen3-ForcedAligner-0.6B"
        }
    }
}
```

## API 使用示例

### 1. 阿里 ASR API

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

### 2. OpenAI 兼容 API

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="empty"
)

# 转写
with open("audio.wav", "rb") as f:
    result = client.audio.transcriptions.create(
        model="qwen3-asr-1.7b",  # 或 qwen3-asr-0.6b
        file=f,
        response_format="verbose_json"  # 包含时间戳和语言信息
    )
    print(result.text)
```

### 3. Python 直接使用

```python
from app.services.asr.manager import get_model_manager

manager = get_model_manager()
engine = manager.get_asr_engine("qwen3-asr-1.7b")

# 简单转写
text = engine.transcribe_file("audio.wav")
print(text)

# 带时间戳转写
result = engine.transcribe_file_with_vad("audio.wav")
for seg in result.segments:
    print(f"[{seg.start_time:.2f}-{seg.end_time:.2f}] {seg.text}")
```

## 性能对比

| 模型 | 参数量 | GPU显存 | 适合场景 |
|------|--------|---------|----------|
| qwen3-asr-1.7b ⭐ | 1.7B | ~8GB | **默认模型**，最佳准确率，52种语言 |
| paraformer-large | 220M | ~2GB | 高精度中文 |
| fun-asr-nano | - | ~4GB | 多语言+方言 |

## 注意事项

1. **首次启动**：模型会自动从 HuggingFace/ModelScope 下载，可能需要几分钟
2. **GPU 显存**：1.7B 模型建议至少 8GB 显存，0.6B 建议 4GB
3. **长音频**：自动分段处理，与现有 FunASR 逻辑一致
4. **实时识别**：暂不支持 WebSocket 流式（vLLM 后端限制）

## 故障排查

### 模型加载失败
```bash
# 检查 vLLM 是否安装
python -c "import vllm; print(vllm.__version__)"

# 检查 qwen-asr 是否安装
python -c "from qwen_asr import Qwen3ASRModel; print('OK')"
```

### 显存不足
修改 `models.json` 中的 `gpu_memory_utilization`：
- 1.7B 模型：0.8 → 0.7 或更低
- 0.6B 模型：0.6 → 0.5 或更低

### 模型下载慢
```bash
# 使用 ModelScope 镜像（国内）
export MODELSCOPE_CACHE=~/.cache/modelscope
modelscope download --model Qwen/Qwen3-ASR-1.7B
```

## 文件清单

- `app/services/asr/qwen3_engine.py` - Qwen3-ASR 引擎实现（~200行）
- `app/services/asr/manager.py` - 添加 qwen3 引擎支持
- `app/services/asr/models.json` - 添加模型配置
- `requirements.txt` - 添加 qwen-asr[vllm] 依赖
- `app/api/v1/asr.py` - 更新 API 文档
- `app/api/v1/openai_compatible.py` - 更新 OpenAI API 文档
