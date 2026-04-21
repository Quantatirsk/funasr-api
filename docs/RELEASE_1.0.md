# Release 1.0

## 说明

`v1.0.0` 是一次相对于当前 `main` 分支的**大规模 breaking refactor**。

如果你之前使用的是 `main` 分支语义，请不要假设可以无感升级；建议把这次版本视为：

- 新的运行时基线
- 新的部署基线
- 新的 API 兼容语义基线

当前对比范围：

- `main`: `778e82d`
- `dev`: `e65c4c5` 及之前连续的运行时重构提交

## 主要 breaking changes

### 1. 依赖与部署方式切换到 `uv`

当前依赖真源已经变成：

- [pyproject.toml](/Users/quant/Documents/funasr-api/pyproject.toml)
- [uv.lock](/Users/quant/Documents/funasr-api/uv.lock)

旧的这些文件已经移除：

- `requirements.txt`
- `requirements-cpu.txt`
- `requirements-apple-silicon.txt`

当前安装方式：

- CPU: `uv sync --group cpu`
- GPU: `uv sync --group gpu`

### 2. 运行时栈重构

当前主线运行时已经改成：

- `CUDA -> official vLLM`
- `CPU / macOS -> vendored QwenASR Rust`

同时：

- 旧的 `MLX` / Apple Silicon 独立 GPU 路径已移除
- `mps` 现在会归一化到 `cpu`

这意味着：

- Apple Silicon 现在默认走 Rust CPU backend
- 不再存在旧版 `MLX` 行为语义

### 3. 默认模型选择语义改变

当前默认规则：

- `macOS / Apple Silicon`
  - 默认始终选择 `qwen3-asr-0.6b`
  - 不再根据内存大小自动切到 `qwen3-asr-1.7b`
  - 只有调用方显式指定时才使用 `1.7b`
- `Linux / CPU`
  - 默认 `qwen3-asr-0.6b`
- `Linux / CUDA`
  - 按显存自动在 `0.6b / 1.7b` 间选择

### 4. 离线路径不再让客户端真正选择模型

离线路径现在是**单激活模型语义**：

- 客户端传入的 `model` / `model_id`
  - 仅作为兼容参数保留
  - 对离线路径会被忽略

因此，如果旧客户端依赖：

- 通过请求参数切换离线模型

那么这条行为现在已经不再成立。

### 5. `ENABLED_MODELS` 已删除

当前部署不再通过 `ENABLED_MODELS` 配置“启用哪组模型”。

现在的模型计划由运行时自动决定：

- 离线 Qwen 主模型
- Paraformer realtime capability

这也意味着：

- 旧的 `auto / all / 手工逗号列表` 语义已经退出主线

### 6. API 与元数据语义收敛

这轮重构里，模型与 capability 的表达被重新收紧：

- `/v1/models`
  - 只表达**离线模型列表**
- `/stream/v1/asr/models`
  - 表达**声明条目 + runtime 状态**
- `paraformer-large`
  - 不再被当成“普通离线模型”
  - 现在明确是 websocket realtime capability

### 7. 内部实现层发生大幅变化

包括但不限于：

- vendored `qwenasr` Rust backend 引入
- Rust runtime 共享只读模型权重
- x86_64 Rust fallback bring-up
- CUDA 官方 vLLM 适配层替换旧包装
- Rust CPU benchmark / sensitivity 工具链加入
- CAM++ batched SV 默认值从 `16` 调到 `32`

这些变化本身不一定都直接暴露为公共 API，但会显著影响：

- 性能表现
- 默认路径
- 部署方式
- 文档和排障方法

## 升级时最需要注意的事

如果你是从 `main` 迁移到 `v1.0.0`，优先检查这几件事：

1. 不要再使用 `requirements*.txt` / `pip install -r ...`
2. 不要再依赖 `MLX` / `mps` 路径
3. 不要再依赖客户端通过 `model` / `model_id` 切换离线模型
4. 不要再依赖 `ENABLED_MODELS`
5. macOS / Apple Silicon 默认模型已经固定为 `qwen3-asr-0.6b`
6. 部署和运行时真值请优先参考 [runtime_instruction.md](/Users/quant/Documents/funasr-api/docs/runtime_instruction.md)

## 推荐阅读顺序

如果你要基于 `v1.0.0` 部署或继续开发，建议按这个顺序看：

1. [README.md](/Users/quant/Documents/funasr-api/README.md)
2. [docs/runtime_instruction.md](/Users/quant/Documents/funasr-api/docs/runtime_instruction.md)
3. [docs/deployment.md](/Users/quant/Documents/funasr-api/docs/deployment.md)
4. [docs/MODEL_SETUP.md](/Users/quant/Documents/funasr-api/docs/MODEL_SETUP.md)
