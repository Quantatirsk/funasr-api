<div align="center">

<h3>Ready-to-use Local Speech Recognition API Service</h3>

Speech recognition API service powered by [FunASR](https://github.com/alibaba-damo-academy/FunASR) and [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR), supporting 52 languages, compatible with OpenAI API and Alibaba Cloud Speech API.

[简体中文](./docs/README_zh.md)

---

![Static Badge](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Static Badge](https://img.shields.io/badge/Torch-2.3.1-%23EE4C2C?logo=pytorch&logoColor=white)
![Static Badge](https://img.shields.io/badge/CUDA-12.1+-%2376B900?logo=nvidia&logoColor=white)

</div>

## Demo

<video src="https://raw.githubusercontent.com/Quantatirsk/funasr-api/main/demo/demo.mp4" controls width="100%"></video>

## Features

- **Multi-Model Support** - Integrates [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) 1.7B/0.6B and Paraformer Large ASR models
- **Speaker Diarization** - Automatic multi-speaker identification using CAM++ model
- **OpenAI API Compatible** - Supports `/v1/audio/transcriptions` endpoint, works with OpenAI SDK
- **Alibaba Cloud API Compatible** - Supports Alibaba Cloud Speech RESTful API and WebSocket streaming protocol
- **WebSocket Streaming** - Real-time streaming speech recognition with low latency
- **Smart Far-Field Filtering** - Automatically filters far-field sounds and ambient noise in streaming ASR
- **Intelligent Audio Segmentation** - VAD-based greedy merge algorithm for automatic long audio splitting
- **GPU Batch Processing** - Batch inference support, 2-3x faster than sequential processing
- **Flexible Configuration** - Environment variable based configuration

## Quick Deployment

### 1. Download Models (Required for First Deployment)

Models will be automatically downloaded on first startup. For pre-download or offline deployment:

```bash
# Create model directories
mkdir -p models/modelscope models/huggingface

# Start service, models will be downloaded to models/ directory
docker-compose up -d

# Or use symlinks (if models are already in other locations)
ln -s ~/.cache/modelscope ./models/modelscope
ln -s ~/.cache/huggingface ./models/huggingface
```

> For detailed instructions, see [Model Setup Guide](./docs/MODEL_SETUP.md)

### 2. Docker Deployment (Recommended)

```bash
# Start service (GPU version) - using docker-compose (recommended)
docker-compose up -d

# Or using docker run (GPU version)
docker run -d --name funasr-api \
  --gpus all \
  -p 17003:8000 \
  -v ./models/modelscope:/root/.cache/modelscope \
  -v ./models/huggingface:/root/.cache/huggingface \
  -v ./logs:/app/logs \
  -v ./temp:/app/temp \
  quantatrisk/funasr-api:gpu-latest
```

Service URLs:
- **docker-compose**: `http://localhost:17003` (via Nginx proxy)
- **docker run**: `http://localhost:17003` (direct mapping)
- **API Docs**: `http://localhost:17003/docs`

> **Note**: docker-compose uses port 17003 as Nginx entrypoint, internal service runs on port 8000

**Model Mount Paths** (docker-compose configuration):
- **GPU Mode**: `./models/modelscope:/root/.cache/modelscope` and `./models/huggingface:/root/.cache/huggingface`
- **CPU Mode**: Only `./models/modelscope:/root/.cache/modelscope` (Qwen3 models require GPU)

**CPU Version**: Use image `quantatrisk/funasr-api:cpu-latest`

**Offline Deployment**: Pack and copy the `models/` directory to the offline machine. See [MODEL_SETUP.md](./docs/MODEL_SETUP.md) for details.

> Detailed deployment instructions: [Deployment Guide](./docs/deployment.md)

### Local Development

**System Requirements:**

- Python 3.10+
- CUDA 12.1+ (optional, for GPU acceleration)
- FFmpeg (audio format conversion)

**Installation:**

```bash
# Clone project
cd FunASR-API

# Install dependencies
pip install -r requirements.txt

# Start service
python start.py
```

## API Endpoints

### OpenAI Compatible API

| Endpoint | Method | Function |
|----------|--------|----------|
| `/v1/audio/transcriptions` | POST | Audio transcription (OpenAI compatible) |
| `/v1/models` | GET | Model list |

**Request Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file` | file | Required | Audio file |
| `model` | string | `qwen3-asr-1.7b` | Model selection |
| `language` | string | Auto-detect | Language code (zh/en/ja) |
| `enable_speaker_diarization` | bool | `true` | Enable speaker diarization |
| `word_timestamps` | bool | `true` | Return word-level timestamps (Qwen3-ASR only) |
| `response_format` | string | `verbose_json` | Output format |
| `prompt` | string | - | Prompt text (reserved) |
| `temperature` | float | `0` | Sampling temperature (reserved) |

**Usage Examples:**

```python
# Using OpenAI SDK
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="any")

with open("audio.wav", "rb") as f:
    transcript = client.audio.transcriptions.create(
        model="whisper-1",  # Maps to default model
        file=f,
        response_format="verbose_json"  # Get segments and speaker info
    )
print(transcript.text)
```

```bash
# Using curl
curl -X POST "http://localhost:8000/v1/audio/transcriptions" \
  -H "Authorization: Bearer any" \
  -F "file=@audio.wav" \
  -F "model=paraformer-large" \
  -F "response_format=verbose_json" \
  -F "enable_speaker_diarization=true"
```

**Supported Response Formats:** `json`, `text`, `srt`, `vtt`, `verbose_json`

### Alibaba Cloud Compatible API

| Endpoint | Method | Function |
|----------|--------|----------|
| `/stream/v1/asr` | POST | Speech recognition (long audio support) |
| `/stream/v1/asr/models` | GET | Model list |
| `/stream/v1/asr/health` | GET | Health check |
| `/ws/v1/asr` | WebSocket | Streaming ASR (Alibaba Cloud protocol compatible) |
| `/ws/v1/asr/funasr` | WebSocket | FunASR streaming (backward compatible) |
| `/ws/v1/asr/qwen` | WebSocket | Qwen3-ASR streaming |
| `/ws/v1/asr/test` | GET | WebSocket test page |

**Request Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_id` | string | `qwen3-asr-1.7b` | Model ID |
| `audio_address` | string | - | Audio URL (optional) |
| `sample_rate` | int | `16000` | Sample rate |
| `enable_speaker_diarization` | bool | `true` | Enable speaker diarization |
| `word_timestamps` | bool | `false` | Return word-level timestamps (Qwen3-ASR only) |
| `vocabulary_id` | string | - | Hotwords (format: `word1 weight1 word2 weight2`) |

**Usage Examples:**

```bash
# Basic usage
curl -X POST "http://localhost:8000/stream/v1/asr" \
  -H "Content-Type: application/octet-stream" \
  --data-binary @audio.wav

# With parameters
curl -X POST "http://localhost:8000/stream/v1/asr?enable_speaker_diarization=true" \
  -H "Content-Type: application/octet-stream" \
  --data-binary @audio.wav
```

**Response Example:**

```json
{
  "task_id": "xxx",
  "status": 200,
  "message": "SUCCESS",
  "result": "Speaker1 content...\nSpeaker2 content...",
  "duration": 60.5,
  "processing_time": 1.234,
  "segments": [
    {
      "text": "Today is a nice day.",
      "start_time": 0.0,
      "end_time": 2.5,
      "speaker_id": "Speaker1",
      "word_tokens": [
        {"text": "Today", "start_time": 0.0, "end_time": 0.5},
        {"text": "is", "start_time": 0.5, "end_time": 0.7},
        {"text": "a nice day", "start_time": 0.7, "end_time": 1.5}
      ]
    }
  ]
}
```

**WebSocket Streaming Test:** Visit `http://localhost:8000/ws/v1/asr/test`

## Speaker Diarization

Multi-speaker automatic identification based on CAM++ model:

- **Enabled by Default** - `enable_speaker_diarization=true`
- **Automatic Detection** - No preset speaker count needed, model auto-detects
- **Speaker Labels** - Response includes `speaker_id` field (e.g., "Speaker1", "Speaker2")
- **Smart Merging** - Two-layer merge strategy to avoid isolated short segments:
  - Layer 1: Accumulate merge same-speaker segments < 10 seconds
  - Layer 2: Accumulate merge continuous segments up to 60 seconds
- **Subtitle Support** - SRT/VTT output includes speaker labels `[Speaker1] text content`

Disable speaker diarization:

```bash
# OpenAI API
-F "enable_speaker_diarization=false"

# Alibaba Cloud API
?enable_speaker_diarization=false
```

## Audio Processing

### Intelligent Segmentation Strategy

Automatic long audio segmentation:

1. **VAD Voice Detection** - Detect voice boundaries, filter silence
2. **Greedy Merge** - Accumulate voice segments, ensure each segment does not exceed `MAX_SEGMENT_SEC` (default 90s)
3. **Silence Split** - Force split when silence between voice segments exceeds 3 seconds
4. **Batch Inference** - Multi-segment parallel processing, 2-3x performance improvement in GPU mode

### WebSocket Streaming Limitations

**FunASR Model Limitations** (using `/ws/v1/asr` or `/ws/v1/asr/funasr`):
- ✅ Real-time speech recognition, low latency
- ✅ Sentence-level timestamps
- ❌ **Word-level timestamps** (not implemented)
- ❌ **Confidence scores** (not implemented)

**Qwen3-ASR Streaming** (using `/ws/v1/asr/qwen`):
- ✅ Word-level timestamps
- ✅ Multi-language real-time recognition

## Supported Models

| Model ID | Name | Description | Features |
|----------|------|-------------|----------|
| `qwen3-asr-1.7b` | Qwen3-ASR 1.7B | High-performance multilingual ASR, 52 languages + dialects, vLLM backend | Offline/Realtime |
| `qwen3-asr-0.6b` | Qwen3-ASR 0.6B | Lightweight multilingual ASR, suitable for low VRAM environments | Offline/Realtime |
| `paraformer-large` | Paraformer Large | High-precision Chinese speech recognition | Offline/Realtime |

**Dynamic Model Loading:**

System automatically selects appropriate Qwen3-ASR model based on VRAM:
- **VRAM >= 32GB**: Auto-load `qwen3-asr-1.7b`
- **VRAM < 32GB**: Auto-load `qwen3-asr-0.6b`
- **No CUDA**: Only load `paraformer-large` (Qwen3 requires vLLM/GPU)

Use `QWEN_ASR_MODEL` environment variable to force specific model version.

**Preload Custom Models:**

```bash
# Preload paraformer-large on startup
export AUTO_LOAD_CUSTOM_ASR_MODELS="paraformer-large"
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | `0.0.0.0` | Service bind address |
| `PORT` | `8000` | Service port |
| `DEBUG` | `false` | Debug mode |
| `DEVICE` | `auto` | Device selection: `auto`, `cpu`, `cuda:0` |
| `AUTO_LOAD_CUSTOM_ASR_MODELS` | - | Preload custom models |
| `APPTOKEN` | - | API access token |
| `APPKEY` | - | App key |
| `LOG_LEVEL` | `INFO` | Log level (DEBUG/INFO/WARNING/ERROR) |
| `WORKERS` | `1` | Worker processes |
| `MAX_AUDIO_SIZE` | `2048` | Max audio file size (MB, supports units like 2GB) |

### Performance Optimization

| Variable | Default | Description |
|----------|---------|-------------|
| `ASR_BATCH_SIZE` | `4` | ASR batch size (GPU: 4, CPU: 2) |
| `INFERENCE_THREAD_POOL_SIZE` | `4` | Inference thread pool size (CPU: 1) |
| `MAX_SEGMENT_SEC` | `90` | Max audio segment duration (seconds) |
| `WS_MAX_BUFFER_SIZE` | `160000` | WebSocket audio buffer size (samples) |
| `ENABLE_STREAMING_VLLM` | `false` | Load streaming VLLM instance (saves VRAM) |

### Model Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `QWEN_ASR_MODEL` | `auto` | Qwen3-ASR model selection: auto/1.7B/0.6B |
| `MODELSCOPE_PATH` | `~/.cache/modelscope/hub/models` | ModelScope cache path |
| `HF_HOME` | `~/.cache/huggingface` | HuggingFace cache path (GPU mode) |
| `ASR_ENABLE_LM` | `true` | Enable language model (Paraformer) |
| `LM_WEIGHT` | `0.15` | Language model weight (0.1-0.3) |
| `LM_BEAM_SIZE` | `10` | Language model decoding beam size |

### Near-Field Filtering

| Variable | Default | Description |
|----------|---------|-------------|
| `ASR_ENABLE_NEARFIELD_FILTER` | `true` | Enable far-field sound filtering |
| `ASR_NEARFIELD_RMS_THRESHOLD` | `0.01` | RMS energy threshold |
| `ASR_NEARFIELD_FILTER_LOG_ENABLED` | `true` | Enable filtering logs |

> Detailed configuration: [Near-Field Filter Docs](./docs/nearfield_filter.md)

## Resource Requirements

**Minimum (CPU):**

- CPU: 4 cores
- Memory: 16GB
- Disk: 20GB

**Recommended (GPU):**

- CPU: 4 cores
- Memory: 16GB
- GPU: NVIDIA GPU (16GB+ VRAM)
- Disk: 20GB

## API Documentation

After starting the service:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Links

- **Deployment Guide**: [Detailed Docs](./docs/deployment.md)
- **Near-Field Filter Config**: [Config Guide](./docs/nearfield_filter.md)
- **FunASR**: [FunASR GitHub](https://github.com/alibaba-damo-academy/FunASR)
- **Chinese README**: [中文文档](./docs/README_zh.md)

## License

This project uses the MIT License - see [LICENSE](LICENSE) file for details.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Quantatirsk/funasr-api&type=Date)](https://star-history.com/#Quantatirsk/funasr-api&Date)

## Contributing

Issues and Pull Requests are welcome to improve the project!
