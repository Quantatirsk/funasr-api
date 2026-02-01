# 模型下载与部署指南

本文档说明如何预下载模型以及在内网环境中部署 FunASR-API。

## 模型存储结构

FunASR-API 使用以下模型仓库：

| 平台 | 路径 | 用途 |
|------|------|------|
| ModelScope | `~/.cache/modelscope/hub/models` | FunASR 模型（Paraformer、VAD、CAM++ 等） |
| HuggingFace | `~/.cache/huggingface` | Qwen3-ASR 模型（需要 GPU） |

## 自动下载（推荐）

模型会在服务首次启动时自动下载。启动服务后，模型将自动缓存到上述路径。

## 手动预下载

如需预下载模型（例如内网部署场景），可以使用以下方式：

### 1. 使用 ModelScope 下载 FunASR 模型

```python
# 使用 ModelScope 下载模型
from modelscope import snapshot_download

# 下载 Paraformer-large 模型
snapshot_download("iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch")

# 下载 VAD 模型
snapshot_download("damo/speech_fsmn_vad_zh-cn-16k-common-pytorch")

# 下载标点模型
snapshot_download("iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch")

# 下载实时标点模型
snapshot_download("iic/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727")

# 下载 CAM++ 说话人分离模型
snapshot_download("iic/speech_campplus_speaker-diarization_common")

# 下载 N-gram 语言模型
snapshot_download("iic/speech_ngram_lm_zh-cn-ai-wesp-fst")
```

### 2. 使用 HuggingFace 下载 Qwen3-ASR 模型

```python
# 使用 HuggingFace 下载 Qwen3-ASR 模型
from huggingface_hub import snapshot_download

# 下载 Qwen3-ASR 1.7B 模型
snapshot_download("Qwen/Qwen3-ASR-1.7B")

# 下载 Qwen3-ASR 0.6B 模型
snapshot_download("Qwen/Qwen3-ASR-0.6B")
```

## 内网部署

### 步骤 1: 在外网机器下载模型

```bash
# 创建模型目录
mkdir -p models/modelscope models/huggingface

# 使用 Docker 下载模型
docker run --rm \
  -v $(pwd)/models/modelscope:/root/.cache/modelscope \
  -v $(pwd)/models/huggingface:/root/.cache/huggingface \
  quantatrisk/funasr-api:gpu-latest

# 服务启动后会自动下载模型，等待下载完成后停止容器
```

### 步骤 2: 打包模型

```bash
# 打包模型目录
tar -czvf funasr-models.tar.gz models/

# 传输到内网机器
scp funasr-models.tar.gz user@internal-server:/path/to/deploy/
```

### 步骤 3: 内网机器部署

```bash
# 在内网机器解压模型
cd /path/to/deploy/
tar -xzvf funasr-models.tar.gz

# 启动服务（使用本地模型）
docker run -d --name funasr-api \
  --gpus all \
  -p 17003:8000 \
  -v ./models/modelscope:/root/.cache/modelscope \
  -v ./models/huggingface:/root/.cache/huggingface \
  -v ./logs:/app/logs \
  -v ./temp:/app/temp \
  -e DEVICE=auto \
  -e MODELSCOPE_PATH=/root/.cache/modelscope/hub/models \
  -e HF_HOME=/root/.cache/huggingface \
  quantatrisk/funasr-api:gpu-latest
```

## Docker Compose 内网部署

```yaml
services:
  funasr-api:
    image: quantatrisk/funasr-api:gpu-latest
    container_name: funasr-api
    ports:
      - "17003:8000"
    volumes:
      # 挂载预下载的模型
      - ./models/modelscope:/root/.cache/modelscope
      - ./models/huggingface:/root/.cache/huggingface
      - ./temp:/app/temp
      - ./logs:/app/logs
    environment:
      - DEBUG=false
      - LOG_LEVEL=INFO
      - DEVICE=auto
      # 显式设置模型缓存路径
      - MODELSCOPE_PATH=/root/.cache/modelscope/hub/models
      - HF_HOME=/root/.cache/huggingface
    restart: unless-stopped
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

## 模型路径映射说明

### 挂载方案 1: 项目本地目录（推荐）

将模型与项目放在一起，便于备份和迁移：

```yaml
volumes:
  - ./models/modelscope:/root/.cache/modelscope
  - ./models/huggingface:/root/.cache/huggingface
```

### 挂载方案 2: 用户级缓存目录

多项目共享模型，节省磁盘空间：

```yaml
volumes:
  - ~/.cache/modelscope:/root/.cache/modelscope
  - ~/.cache/huggingface:/root/.cache/huggingface
```

## 验证模型加载

启动服务后，可以通过以下方式验证模型是否正确加载：

```bash
# 查看模型列表
curl http://localhost:17003/stream/v1/asr/models

# 查看服务日志
docker logs funasr-api | grep -E "模型|加载|model"
```

## 故障排除

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| 模型下载失败 | 网络问题 | 检查网络连接，或手动下载后挂载 |
| 模型加载失败 | 路径错误 | 检查挂载路径和 MODELSCOPE_PATH/HF_HOME 环境变量 |
| 显存不足 | GPU 显存不够 | 切换到 CPU 版本或更小的模型 |
| 权限错误 | 文件权限 | 检查模型目录的读写权限 |
