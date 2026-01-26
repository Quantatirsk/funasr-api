<div align="center">

  <h3>开箱即用的本地私有化部署语音识别服务</h3>

基于 FunASR 的语音识别 API 服务，提供高精度中文语音识别(ASR)功能，兼容阿里云语音 API 和 OpenAI Audio API。

---

![Static Badge](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Static Badge](https://img.shields.io/badge/Torch-2.3.1-%23EE4C2C?logo=pytorch&logoColor=white)
![Static Badge](https://img.shields.io/badge/CUDA-12.1+-%2376B900?logo=nvidia&logoColor=white)

</div>

## 主要特性

- **多模型支持** - 集成 Paraformer Large 和 Fun-ASR-Nano 2 个高质量 ASR 模型
- **说话人分离** - 基于 CAM++ 模型自动识别多说话人，返回说话人标记
- **OpenAI API 兼容** - 支持 `/v1/audio/transcriptions` 端点，可直接使用 OpenAI SDK
- **阿里云 API 兼容** - 支持阿里云语音识别 RESTful API 和 WebSocket 流式协议
- **WebSocket 流式识别** - 支持实时流式语音识别，低延迟
- **智能远场过滤** - 流式 ASR 自动过滤远场声音和环境音，减少误触发
- **灵活配置** - 支持环境变量配置，可按需加载模型

## 快速部署

### Docker 部署(推荐)

```bash
# 启动服务（GPU 版本）
docker run -d --name funasr-api \
  --gpus all \
  -p 8000:8000 \
  -v ./logs:/app/logs \
  -v ./temp:/app/temp \
  quantatrisk/funasr-api:gpu-latest

# 或使用 docker-compose
docker-compose up -d
```

服务将在 `http://localhost:8000` 启动

**CPU 版本**请使用镜像 `quantatrisk/funasr-api:latest`

> 详细部署说明请查看 [部署指南](./docs/deployment.md)

### 本地开发

**系统要求:**

- Python 3.10+
- CUDA 12.1+(可选，用于 GPU 加速)
- FFmpeg(音频格式转换)

**安装步骤:**

```bash
# 克隆项目
cd FunASR-API

# 安装依赖
pip install -r requirements.txt

# 启动服务
python start.py
```

## API 接口

### OpenAI 兼容接口

| 端点 | 方法 | 功能 |
|------|------|------|
| `/v1/audio/transcriptions` | POST | 音频转写（OpenAI 兼容） |
| `/v1/models` | GET | 模型列表 |

**请求参数:**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `file` | file | 必填 | 音频文件 |
| `model` | string | `paraformer-large` | 模型选择 |
| `language` | string | 自动检测 | 语言代码 (zh/en/ja) |
| `enable_speaker_diarization` | bool | `true` | 启用说话人分离 |
| `response_format` | string | `json` | 输出格式 |

**使用示例:**

```python
# 使用 OpenAI SDK
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="any")

with open("audio.wav", "rb") as f:
    transcript = client.audio.transcriptions.create(
        model="whisper-1",  # 会映射到默认模型
        file=f,
        response_format="verbose_json"  # 获取分段和说话人信息
    )
print(transcript.text)
```

```bash
# 使用 curl
curl -X POST "http://localhost:8000/v1/audio/transcriptions" \
  -H "Authorization: Bearer any" \
  -F "file=@audio.wav" \
  -F "model=paraformer-large" \
  -F "response_format=verbose_json" \
  -F "enable_speaker_diarization=true"
```

**支持的响应格式:** `json`, `text`, `srt`, `vtt`, `verbose_json`

### 阿里云兼容接口

| 端点 | 方法 | 功能 |
|------|------|------|
| `/stream/v1/asr` | POST | 语音识别（支持长音频） |
| `/stream/v1/asr/models` | GET | 模型列表 |
| `/stream/v1/asr/health` | GET | 健康检查 |
| `/ws/v1/asr` | WebSocket | 流式语音识别 |
| `/ws/v1/asr/test` | GET | WebSocket 测试页面 |

**请求参数:**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model_id` | string | `paraformer-large` | 模型 ID |
| `audio_address` | string | - | 音频 URL（可选） |
| `sample_rate` | int | `16000` | 采样率 |
| `enable_speaker_diarization` | bool | `true` | 启用说话人分离 |
| `vocabulary_id` | string | - | 热词（格式：`词1 权重1 词2 权重2`） |

**使用示例:**

```bash
# 基本用法
curl -X POST "http://localhost:8000/stream/v1/asr" \
  -H "Content-Type: application/octet-stream" \
  --data-binary @audio.wav

# 带参数
curl -X POST "http://localhost:8000/stream/v1/asr?enable_speaker_diarization=true" \
  -H "Content-Type: application/octet-stream" \
  --data-binary @audio.wav
```

**响应示例:**

```json
{
  "task_id": "xxx",
  "status": 200,
  "message": "SUCCESS",
  "result": "说话人1的内容...\n说话人2的内容...",
  "duration": 60.5,
  "segments": [
    {
      "text": "今天天气不错。",
      "start_time": 0.0,
      "end_time": 2.5,
      "speaker_id": "说话人1"
    },
    {
      "text": "是的，适合出去走走。",
      "start_time": 2.8,
      "end_time": 5.2,
      "speaker_id": "说话人2"
    }
  ]
}
```

**WebSocket 流式识别测试:** 访问 `http://localhost:8000/ws/v1/asr/test`

## 说话人分离

基于 CAM++ 模型实现多说话人自动识别：

- **默认开启** - `enable_speaker_diarization=true`
- **自动识别** - 无需预设说话人数量，模型自动检测
- **说话人标记** - 响应中包含 `speaker_id` 字段（如 "说话人1"、"说话人2"）
- **字幕支持** - SRT/VTT 格式输出包含说话人标记 `[说话人1] 文本内容`

关闭说话人分离：
```bash
# OpenAI API
-F "enable_speaker_diarization=false"

# 阿里云 API
?enable_speaker_diarization=false
```

## 支持的模型

| 模型 ID | 名称 | 说明 | 特性 |
|---------|------|------|------|
| `paraformer-large` | Paraformer Large | 高精度中文语音识别（默认） | 支持离线/实时 |
| `fun-asr-nano` | Fun-ASR-Nano | 轻量级多语言ASR，支持31种语言和方言 | 仅离线 |

**模型加载模式 (`ASR_MODEL_MODE`):**

- `offline` - 仅加载离线模型
- `realtime` - 仅加载实时流式模型
- `all` - 加载所有模型（默认）

**预加载自定义模型:**

```bash
# 启动时预加载 Fun-ASR-Nano
export AUTO_LOAD_CUSTOM_ASR_MODELS="fun-asr-nano"
```

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `HOST` | `0.0.0.0` | 服务绑定地址 |
| `PORT` | `8000` | 服务端口 |
| `DEBUG` | `false` | 调试模式 |
| `DEVICE` | `auto` | 设备选择: `auto`, `cpu`, `cuda:0` |
| `ASR_MODEL_MODE` | `all` | 模型加载模式 |
| `AUTO_LOAD_CUSTOM_ASR_MODELS` | - | 预加载的自定义模型 |
| `APPTOKEN` | - | API 访问令牌 |
| `APPKEY` | - | 应用密钥 |

### 远场过滤配置

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `ASR_ENABLE_NEARFIELD_FILTER` | `true` | 启用远场声音过滤 |
| `ASR_NEARFIELD_RMS_THRESHOLD` | `0.01` | RMS 能量阈值 |
| `ASR_NEARFIELD_FILTER_LOG_ENABLED` | `true` | 启用过滤日志 |

> 详细配置说明请查看 [远场过滤文档](./docs/nearfield_filter.md)

## 资源需求

**最小配置（CPU）:**
- CPU: 4 核
- 内存: 8GB
- 磁盘: 10GB

**推荐配置（GPU）:**
- CPU: 8 核
- 内存: 16GB
- GPU: NVIDIA GPU (4GB+ 显存)
- 磁盘: 20GB

## API 文档

启动服务后访问：
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## 相关链接

- **部署指南**: [详细文档](./docs/deployment.md)
- **远场过滤配置**: [配置指南](./docs/nearfield_filter.md)
- **FunASR**: [FunASR GitHub](https://github.com/alibaba-damo-academy/FunASR)

## 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 贡献

欢迎提交 Issue 和 Pull Request 来改进项目!
