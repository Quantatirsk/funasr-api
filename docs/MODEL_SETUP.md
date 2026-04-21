# 模型下载与部署指南

本文档说明如何预下载模型以及在内网环境中部署 FunASR-API。

## 模型存储结构

FunASR-API 使用以下模型仓库：

| 平台 | 路径 | 用途 |
|------|------|------|
| ModelScope | `~/.cache/modelscope/hub/models` | FunASR 模型（Paraformer、VAD、CAM++ 等） |
| HuggingFace | `~/.cache/huggingface` | Qwen3-ASR 模型（CUDA 和 CPU/macOS 共用） |

## 预下载（推荐）

当前默认启用 `HF_HUB_LOCAL_FILES_ONLY=1`，推荐在启动前先准备模型缓存，而不是依赖首次启动时联网拉取。

如需预下载模型（例如内网部署场景），推荐使用辅助脚本：

### 方式一：使用辅助脚本（推荐）

```bash
# 运行模型准备脚本
./scripts/prepare-models.sh
```

脚本会直接按当前机器的运行计划准备模型包：
- 自动选择当前应使用的 Qwen3-ASR 版本
- 始终包含 WebSocket realtime 所需的 Paraformer 栈
- 始终包含 VAD 与 CAM++ 依赖

下载完成后，脚本会自动打包为 `funasr-models-<timestamp>.tar.gz`，可直接复制到内网服务器使用。

### 方式二：直接使用项目 CLI

如果不需要交互式包装层，可以直接使用项目内置 CLI：

```bash
# 仅下载到默认缓存目录
uv run python -m app.utils.download_models

# 同时导出到指定目录（适合离线打包）
uv run python -m app.utils.download_models --export-dir ./models
```

这样会直接复用项目当前的能力分组和运行计划，不需要在文档里手工维护模型 ID 列表。

## 内网部署

### 步骤 1: 在外网机器准备模型

```bash
# 运行辅助脚本，按当前运行计划导出模型包
./scripts/prepare-models.sh

# 脚本会生成类似 funasr-models-20250206-143022.tar.gz 的文件
```

### 步骤 2: 传输到内网服务器

```bash
# 将打包的模型传输到内网服务器
scp funasr-models-*.tar.gz user@internal-server:/opt/funasr-api/
```

### 步骤 3: 内网机器部署

```bash
# 进入部署目录
cd /opt/funasr-api/

# 解压模型（会自动解压到 models/ 目录）
tar -xzvf funasr-models-*.tar.gz

# 启动服务（docker-compose.yml 已配置好模型挂载）
docker-compose up -d
```

解压后的目录结构：

```
/opt/funasr-api/
├── docker-compose.yml
├── models/
│   ├── modelscope/     # FunASR 模型（Paraformer、VAD、CAM++ 等）
│   └── huggingface/    # Qwen3-ASR 模型
└── funasr-models-*.tar.gz
```

此结构与 `docker-compose.yml` 中的挂载配置一致，无需额外配置即可直接使用。

## Docker Compose 内网部署

使用 `prepare-models.sh` 准备的模型目录与 `docker-compose.yml` 完全兼容：

```yaml
services:
  funasr-api:
    image: quantatrisk/funasr-api:gpu-latest
    container_name: funasr-api
    ports:
      - "17003:8000"
    volumes:
      # 挂载预下载的模型（与 prepare-models.sh 输出结构一致）
      - ./models/modelscope:/root/.cache/modelscope
      - ./models/huggingface:/root/.cache/huggingface
      - ./temp:/app/temp
      - ./logs:/app/logs
    environment:
      - LOG_LEVEL=INFO
      - DEVICE=auto
    restart: unless-stopped
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

只需确保 `models/` 目录与 `docker-compose.yml` 在同一目录下即可。

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
