# FunASR-API

Production-ready local speech recognition API service powered by [FunASR](https://github.com/alibaba-damo-academy/FunASR) and [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR).

## Quick Start

### GPU Version (Recommended)

```bash
docker run -d --name funasr-api \
  --gpus all \
  -p 17003:8000 \
  -e ENABLED_MODELS=auto \
  -e API_KEY=your_api_key \
  -v ./models/modelscope:/root/.cache/modelscope \
  -v ./models/huggingface:/root/.cache/huggingface \
  -v ./logs:/app/logs \
  -v ./temp:/app/temp \
  quantatrisk/funasr-api:gpu-latest
```

### CPU Version

```bash
docker run -d --name funasr-api \
  -p 17003:8000 \
  -e ENABLED_MODELS=paraformer-large \
  -e API_KEY=your_api_key \
  -v ./models/modelscope:/root/.cache/modelscope \
  -v ./logs:/app/logs \
  -v ./temp:/app/temp \
  quantatrisk/funasr-api:cpu-latest
```

## Supported Tags

| Tag | Description |
|-----|-------------|
| `gpu-latest` | GPU version with CUDA 12.6, auto model selection |
| `cpu-latest` | CPU-only version, Paraformer model only |

## Features

- **Multi-Model Support**: Qwen3-ASR (1.7B/0.6B) + Paraformer Large
- **OpenAI API Compatible**: `/v1/audio/transcriptions` endpoint
- **Alibaba Cloud Compatible**: RESTful and WebSocket streaming API
- **Speaker Diarization**: Automatic multi-speaker identification
- **Word-Level Timestamps**: Qwen3-ASR supports precise timestamps
- **Smart Far-Field Filtering**: Reduces ambient noise in streaming
- **GPU Batch Processing**: 2-3x faster with batch inference

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLED_MODELS` | `auto` | Models to load: `auto`, `all`, or comma-separated list |
| `API_KEY` | - | API authentication key (optional) |
| `LOG_LEVEL` | `INFO` | Log level: DEBUG, INFO, WARNING, ERROR |
| `MAX_AUDIO_SIZE` | `2048` | Max audio file size in MB |
| `ASR_BATCH_SIZE` | `4` | Batch size for inference (GPU: 4, CPU: 2) |
| `MAX_SEGMENT_SEC` | `90` | Max audio segment duration in seconds |
| `DEVICE` | `auto` | Device: `auto`, `cpu`, `cuda:0` |

## Auto Mode Behavior

- **VRAM >= 32GB**: Auto-load `qwen3-asr-1.7b` + `paraformer-large`
- **VRAM < 32GB**: Auto-load `qwen3-asr-0.6b` + `paraformer-large`
- **No CUDA**: Only `paraformer-large` (Qwen3 requires GPU)

## API Endpoints

- **OpenAI Compatible**: `POST /v1/audio/transcriptions`
- **Alibaba Cloud**: `POST /stream/v1/asr`
- **WebSocket**: `/ws/v1/asr`, `/ws/v1/asr/qwen`
- **Health Check**: `GET /stream/v1/asr/health`
- **API Docs**: `http://localhost:17003/docs`

## Quick Test

```bash
# Health check
curl http://localhost:17003/stream/v1/asr/health

# Transcription with OpenAI API
curl -X POST "http://localhost:17003/v1/audio/transcriptions" \
  -H "Authorization: Bearer your_api_key" \
  -F "file=@audio.wav" \
  -F "model=qwen3-asr-1.7b" \
  -F "response_format=verbose_json"

# Transcription with Alibaba Cloud API
curl -X POST "http://localhost:17003/stream/v1/asr" \
  -H "Content-Type: application/octet-stream" \
  --data-binary @audio.wav
```

## Model Storage

Models are cached in Docker volumes for offline use:

```bash
# ModelScope models (Paraformer, VAD, CAM++)
./models/modelscope:/root/.cache/modelscope

# HuggingFace models (Qwen3-ASR, GPU only)
./models/huggingface:/root/.cache/huggingface
```

First run downloads models automatically. Pre-download for offline deployment:

```bash
# Use the helper script (recommended)
./scripts/prepare-models.sh

# Or manually with Docker
docker run --rm \
  -v ./models/modelscope:/root/.cache/modelscope \
  -v ./models/huggingface:/root/.cache/huggingface \
  quantatrisk/funasr-api:gpu-latest \
  python -c "from app.utils.download_models import download_models; download_models()"
```

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

## Links

- **GitHub**: https://github.com/Quantatirsk/funasr-api
- **Documentation**: https://github.com/Quantatirsk/funasr-api/tree/main/docs
- **Issue Tracker**: https://github.com/Quantatirsk/funasr-api/issues

## License

MIT License
