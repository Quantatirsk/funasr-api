# FunASR-API 部署指南

快速部署 FunASR-API 语音识别服务，支持 CPU 和 GPU 两种模式。

## 快速部署

### GPU 版本部署（推荐）

适用于生产环境，提供更快的推理速度：

**前置要求：**
- NVIDIA GPU (CUDA 12.1+)
- 已安装 NVIDIA Container Toolkit

```bash
# 使用 docker run
docker run -d --name funasr-api \
  --gpus all \
  -p 8000:8000 \
  -v ./logs:/app/logs \
  -v ./temp:/app/temp \
  -e DEVICE=auto \
  quantatrisk/funasr-api:gpu-latest

# 或使用 docker-compose
docker-compose up -d
```

### CPU 版本部署

适用于开发测试或无 GPU 环境：

```bash
docker run -d --name funasr-api \
  -p 8000:8000 \
  -v ./logs:/app/logs \
  -v ./temp:/app/temp \
  -e DEVICE=cpu \
  quantatrisk/funasr-api:latest
```

### 验证部署

```bash
# 健康检查
curl http://localhost:8000/stream/v1/asr/health

# 查看可用模型
curl http://localhost:8000/stream/v1/asr/models

# 测试语音识别
curl -X POST "http://localhost:8000/stream/v1/asr" \
  -H "Content-Type: application/octet-stream" \
  --data-binary @test.wav
```

## 从源码构建镜像

### 使用构建脚本

项目提供了 `build.sh` 脚本简化构建流程：

```bash
# 构建所有版本（CPU + GPU）
./build.sh

# 仅构建 GPU 版本
./build.sh -t gpu

# 构建指定版本并推送
./build.sh -t all -v 1.0.0 -p

# 查看帮助
./build.sh -h
```

**构建脚本参数：**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-t, --type` | 构建类型: `cpu`, `gpu`, `all` | `all` |
| `-v, --version` | 版本标签 | `latest` |
| `-p, --push` | 构建后推送到 Docker Hub | 否 |
| `-r, --registry` | 镜像仓库 | `quantatrisk` |

### 手动构建

```bash
# 构建 CPU 版本
docker build -t funasr-api:latest -f Dockerfile .

# 构建 GPU 版本
docker build -t funasr-api:gpu-latest -f Dockerfile.gpu .
```

### 预下载的模型

构建时会自动下载以下模型：

| 模型 | 说明 |
|------|------|
| Qwen3-ASR-1.7B | 多语言 ASR（默认，52种语言+方言，字级时间戳） |
| Paraformer Large (VAD+PUNC) | 高精度中文 ASR |
| Paraformer Large Online | 实时流式 ASR |
| Fun-ASR-Nano | 多语言+方言 ASR |
| VAD Model | 语音活动检测 |
| CAM++ | 说话人分离 |
| Punctuation Model | 标点预测（离线） |
| Punctuation Model (Realtime) | 标点预测（实时） |
| Language Model | N-gram 语言模型 |

## 环境变量配置

### 服务器配置

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `HOST` | `0.0.0.0` | 服务绑定地址 |
| `PORT` | `8000` | 服务端口 |
| `DEBUG` | `false` | 调试模式（启用后可访问 /docs） |
| `WORKERS` | `1` | 工作进程数（多进程会复制模型，显存成倍增加） |
| `INFERENCE_THREAD_POOL_SIZE` | `4` | 推理线程池大小 |

### 设备配置

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `DEVICE` | `auto` | 设备选择：`auto`, `cpu`, `cuda:0` |
| `NVIDIA_VISIBLE_DEVICES` | - | 可见的 GPU 设备 |

### ASR 模型配置

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `ASR_MODEL_MODE` | `all` | 模型加载模式：`offline`, `realtime`, `all` |
| `AUTO_LOAD_CUSTOM_ASR_MODELS` | - | 预加载的自定义模型（如 `fun-asr-nano`） |
| `ASR_ENABLE_REALTIME_PUNC` | `true` | 是否启用实时标点模型 |

**模式说明：**

| 模式 | 说明 | 适用场景 |
|------|------|----------|
| `offline` | 仅加载离线模型 | REST API 调用 |
| `realtime` | 仅加载实时流式模型 | WebSocket 流式识别 |
| `all` | 加载所有模型（默认） | 完整功能 |

### 远场过滤配置

流式 ASR 远场声音过滤功能，自动过滤远场声音和环境音：

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `ASR_ENABLE_NEARFIELD_FILTER` | `true` | 启用远场声音过滤 |
| `ASR_NEARFIELD_RMS_THRESHOLD` | `0.01` | RMS 能量阈值 |
| `ASR_NEARFIELD_FILTER_LOG_ENABLED` | `true` | 启用过滤日志 |

详细配置请参考 [远场过滤文档](./nearfield_filter.md)

### 鉴权配置

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `APPTOKEN` | - | API 访问令牌（X-NLS-Token header） |
| `APPKEY` | - | 应用密钥（appkey 参数） |

**使用示例：**

```bash
# 使用 Token
curl -H "X-NLS-Token: your_token" http://localhost:8000/stream/v1/asr/health

# 使用 Bearer Token（OpenAI 兼容）
curl -H "Authorization: Bearer your_token" http://localhost:8000/v1/models
```

### 日志配置

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `LOG_LEVEL` | `INFO` | 日志级别：`DEBUG`, `INFO`, `WARNING` |
| `LOG_FILE` | `logs/funasr-api.log` | 日志文件路径 |
| `LOG_MAX_BYTES` | `20971520` | 单个日志文件最大大小（20MB） |
| `LOG_BACKUP_COUNT` | `50` | 日志备份文件数量 |

## Docker Compose 配置

### 基础配置

```yaml
services:
  funasr-api:
    image: quantatrisk/funasr-api:gpu-latest
    container_name: funasr-api
    ports:
      - "8000:8000"
    volumes:
      - ./temp:/app/temp
      - ./logs:/app/logs
    environment:
      - DEBUG=false
      - LOG_LEVEL=INFO
      - DEVICE=auto
      - ASR_MODEL_MODE=all
    restart: unless-stopped
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

### 生产环境配置（带 Nginx）

```yaml
services:
  nginx:
    image: nginx:1.27-alpine
    container_name: funasr-nginx
    ports:
      - "80:80"
    volumes:
      - ./funasr.conf:/etc/nginx/conf.d/default.conf:ro
    depends_on:
      - funasr-api
    restart: unless-stopped
    networks:
      - funasr-net

  funasr-api:
    image: quantatrisk/funasr-api:gpu-latest
    container_name: funasr-api
    expose:
      - "8000"
    volumes:
      - ./temp:/app/temp
      - ./data:/app/data
      - ./logs:/app/logs
    environment:
      - DEBUG=false
      - LOG_LEVEL=INFO
      - DEVICE=cuda:0
      - NVIDIA_VISIBLE_DEVICES=0
      - WORKERS=1
      - INFERENCE_THREAD_POOL_SIZE=4
    restart: unless-stopped
    networks:
      - funasr-net
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ["0"]
              capabilities: [gpu]

networks:
  funasr-net:
    name: funasr-net
```

## 服务监控

### 健康检查

```bash
curl http://localhost:8000/stream/v1/asr/health
```

### 日志监控

```bash
# 实时查看日志
docker logs -f funasr-api

# 查看错误日志
docker logs funasr-api 2>&1 | grep -i error
```

### 资源监控

```bash
# 容器资源使用
docker stats funasr-api

# GPU 使用情况
docker exec -it funasr-api nvidia-smi
```

## 资源需求

### 最小配置（CPU 版本）

- CPU: 4 核
- 内存: 8GB
- 磁盘: 10GB

### 推荐配置（GPU 版本）

- CPU: 8 核
- 内存: 16GB
- GPU: NVIDIA GPU (6GB+ 显存，含说话人分离模型)
- 磁盘: 25GB

## 故障排除

### 常见问题

| 问题 | 症状 | 解决方案 |
|------|------|----------|
| GPU 内存不足 | CUDA OOM 错误 | 设置 `DEVICE=cpu` 或使用更大显存的 GPU |
| 模型加载慢 | 首次启动超时 | 模型会自动下载，首次需要等待 |
| 端口被占用 | 端口冲突错误 | 修改端口映射：`"8080:8000"` |
| 说话人分离失败 | CAM++ 模型错误 | 检查模型是否完整下载，显存是否充足 |

### 调试模式

```bash
# 启用调试模式
docker run -e DEBUG=true -e LOG_LEVEL=DEBUG ...

# 进入容器调试
docker exec -it funasr-api /bin/bash
```

## 更新服务

```bash
# 拉取最新镜像
docker pull quantatrisk/funasr-api:gpu-latest

# 重启服务
docker-compose down && docker-compose up -d
```
