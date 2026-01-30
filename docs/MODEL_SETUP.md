# 模型下载与部署指南

## 架构说明

从本版本开始，模型文件与 Docker 镜像分离：

- **镜像**：只包含代码和依赖（体积小，约 2-3GB）
- **模型**：通过 Volume 挂载到容器（灵活管理，约 10-20GB）

### 优势

1. **镜像体积小**：无需在每次构建时下载模型
2. **内网部署友好**：直接复制 `models/` 目录即可
3. **模型更新方便**：无需重建镜像
4. **多环境共享**：多个容器可共享同一模型目录

---

## 快速开始

### 1. 下载模型到本地

```bash
# 在项目根目录执行
python scripts/download_models.py
```

脚本会将模型下载到 `~/.cache/modelscope/`，大约需要 10-20GB 空间。

### 2. 创建模型目录软链接（推荐）

```bash
# 在项目根目录创建 models 目录，链接到 ModelScope 缓存
ln -s ~/.cache/modelscope ./models
```

或者直接复制模型文件：

```bash
# 复制模型到项目目录（适合内网部署）
cp -r ~/.cache/modelscope ./models
```

### 3. 启动服务

```bash
docker-compose up -d
```

容器会自动挂载 `./models` 目录到 `/root/.cache/modelscope`。

---

## 目录结构

```
funasr-api/
├── models/                    # 模型文件目录（挂载到容器）
│   ├── hub/                   # ModelScope 模型缓存
│   │   ├── iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch/
│   │   ├── FunAudioLLM/Fun-ASR-Nano-2512/
│   │   └── ...
│   └── models/                # 备用缓存路径
├── temp/                      # 临时文件
├── data/                      # 数据目录
├── logs/                      # 日志目录
└── docker-compose.yml
```

---

## 内网部署

### 方案 1：打包模型目录

在有网络的机器上：

```bash
# 1. 下载模型
python scripts/download_models.py

# 2. 打包模型目录
tar -czf models.tar.gz -C ~/.cache modelscope
```

在内网机器上：

```bash
# 1. 解压模型
mkdir -p models
tar -xzf models.tar.gz -C ./models

# 2. 拉取镜像（通过 Docker 镜像仓库或离线导入）
docker pull quantatrisk/funasr-api:gpu-latest

# 3. 启动服务
docker-compose up -d
```

### 方案 2：使用 NFS/共享存储

多台机器共享同一个模型目录：

```yaml
# docker-compose.yml
volumes:
  - /mnt/shared/modelscope:/root/.cache/modelscope  # NFS 挂载点
```

---

## 模型列表

当前支持的模型（约 15GB）：

| 模型 | 用途 | 大小 |
|------|------|------|
| Qwen3-ASR-1.7B | 离线 ASR（默认，52种语言+字级时间戳） | ~4GB |
| Qwen3-ASR-0.6B | 轻量多语言 ASR | ~1.5GB |
| Paraformer Large (VAD+PUNC) | 高精度中文 ASR | ~2GB |
| Paraformer Large Online | 实时 ASR | ~2GB |
| Fun-ASR-Nano | 多语言+方言 ASR | ~1GB |
| FSMN VAD | 语音活动检测 | ~50MB |
| CAM++ | 说话人分离 | ~500MB |
| CT-Transformer | 标点符号 | ~500MB |
| N-gram LM | 语言模型 | ~8GB |

---

## 常见问题

### Q: 如何验证模型是否正确挂载？

```bash
# 进入容器检查
docker exec -it funasr-api ls -lh /root/.cache/modelscope/hub
```

应该能看到模型目录列表。

### Q: 模型下载失败怎么办？

```bash
# 设置 ModelScope 镜像（如果官方源慢）
export MODELSCOPE_CACHE=~/.cache/modelscope
python scripts/download_models.py
```

### Q: 如何只下载部分模型？

编辑 `scripts/download_models.py`，注释掉不需要的模型：

```python
models = [
    # ("iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch", "Paraformer Large"),
    ("FunAudioLLM/Fun-ASR-Nano-2512", "Fun-ASR-Nano"),  # 只下载这个
]
```

### Q: 容器启动报错找不到模型？

检查挂载路径是否正确：

```bash
# 确保 models 目录存在且有内容
ls -lh ./models/hub

# 检查 docker-compose.yml 中的挂载配置
grep -A 5 "volumes:" docker-compose.yml
```

---

## 升级说明

### 从旧版本（模型内置镜像）迁移

1. **导出现有模型**（可选）：

```bash
# 从旧容器中导出模型
docker cp funasr-api:/root/.cache/modelscope ./models
```

2. **更新镜像**：

```bash
docker-compose pull
docker-compose up -d
```

3. **验证**：

```bash
# 检查容器日志
docker-compose logs -f funasr-api
```

---

## 性能优化

### 使用 SSD 存储模型

模型加载速度取决于磁盘 I/O，建议：

- 将 `models/` 目录放在 SSD 上
- 避免使用网络存储（NFS）作为主存储

### 预热模型

首次启动时，模型会被加载到内存/显存，可能需要 30-60 秒。

---

## 技术细节

### 环境变量

容器内的 `MODELSCOPE_CACHE` 环境变量指向 `/root/.cache/modelscope`，与挂载路径一致。

### 模型加载流程

1. 应用启动时，`ASRManager` 读取 `models.json`
2. 通过 `LoaderFactory` 创建对应的加载器
3. 加载器从 `MODELSCOPE_CACHE` 路径加载模型
4. 模型被加载到内存/显存，准备推理

### 自定义模型路径

如果需要使用其他路径，修改 `docker-compose.yml`：

```yaml
environment:
  - MODELSCOPE_CACHE=/custom/path
volumes:
  - /host/custom/path:/custom/path
```
