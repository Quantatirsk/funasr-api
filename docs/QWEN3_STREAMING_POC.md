# Qwen3-ASR vLLM 流式识别 POC

## 概述

本 POC 实现了基于 Qwen3-ASR vLLM 后端的 WebSocket 流式识别端点（方案B：独立端点）。

## 实现特性

### 1. 服务端实现

- **文件**: `app/api/v1/websocket_qwen3_asr.py`
- **端点**: `ws://host/ws/v1/qwen3/asr`
- **协议**: 简化自定义协议（非阿里云兼容）

### 2. Qwen3-ASR 引擎扩展

- **文件**: `app/services/asr/qwen3_engine.py`
- 新增 `Qwen3StreamingState` 状态包装器
- 新增流式方法：
  - `init_streaming_state()` - 初始化流式状态
  - `streaming_transcribe()` - 处理音频块
  - `finish_streaming_transcribe()` - 结束识别
- 修改 `supports_realtime = True`

### 3. 通信协议

#### 连接流程

```
1. 客户端发送控制消息: {"type": "start", "payload": {...}}
2. 服务端返回: {"type": "started", "task_id": "...", "params": {...}}
3. 客户端持续发送二进制音频数据（PCM 16kHz 16bit）
4. 服务端返回中间结果: {"type": "result", "results": [...]}
5. 客户端发送: {"type": "stop"}
6. 服务端返回最终结果: {"type": "final", "result": {...}}
```

#### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `format` | string | "pcm" | 音频格式: "pcm" 或 "wav" |
| `sample_rate` | int | 16000 | 采样率（会自动重采样到 16kHz） |
| `language` | string | null | 强制语言，null 表示自动检测 |
| `context` | string | "" | 热词/上下文提示 |
| `chunk_size_sec` | float | 2.0 | 每块音频长度（秒） |
| `unfixed_chunk_num` | int | 2 | 前 N 个 chunk 不使用 prefix |
| `unfixed_token_num` | int | 5 | 回滚 token 数，减少边界抖动 |

#### 响应格式

**中间结果**:
```json
{
  "type": "result",
  "task_id": "abc123",
  "results": [
    {
      "text": "识别文本",
      "language": "Chinese",
      "chunk_id": 1,
      "is_partial": true
    }
  ]
}
```

**最终结果**:
```json
{
  "type": "final",
  "task_id": "abc123",
  "result": {
    "text": "完整识别文本",
    "language": "Chinese",
    "total_chunks": 5
  }
}
```

## 技术实现细节

### Qwen3-ASR 流式机制

Qwen3-ASR 的流式实现采用**累积重推理**策略：

1. **音频累积**: 每接收一个 chunk，将其追加到 `audio_accum`
2. **完整重推理**: 每次将**所有已接收音频**重新送入 vLLM 模型
3. **Prefix 回滚**: 通过 `unfixed_token_num` 回滚最后 K 个 token，减少边界抖动
4. **状态管理**: `ASRStreamingState` 维护累积音频、解码文本等状态

```
音频 chunk → buffer 累积 → 达 chunk_size → 拼接全部音频 → vLLM 推理 → 解析结果
                ↑___________________________________________↓
                          (下一轮，重新推理全部音频)
```

### 性能特点

| 特性 | 说明 |
|------|------|
| 延迟 | 随音频长度增加（非恒定延迟） |
| 显存 | 需存储全部累积音频 |
| 准确度 | 高（利用全部上下文） |
| 时间戳 | 流式模式不支持 |

## 使用方法

### 1. 配置模型

确保 `ASR_MODEL_TYPE=qwen3-asr-1.7b` 或 `qwen3-asr-0.6b`

```bash
export ASR_MODEL_TYPE=qwen3-asr-1.7b
```

### 2. 启动服务

```bash
python -m app.main
```

### 3. 测试连接

使用提供的测试脚本:

```bash
python scripts/test_qwen3_streaming.py --audio /path/to/test.wav
```

或使用 websocat:

```bash
# 连接
websocat ws://localhost:8000/ws/v1/qwen3/asr

# 发送开始消息
{"type": "start", "payload": {"format": "pcm", "sample_rate": 16000}}

# 发送音频数据（二进制）
# ...

# 发送停止消息
{"type": "stop"}
```

## 已知限制

1. **延迟累积**: 音频越长，每次推理延迟越高
2. **无时间戳**: 流式模式不支持字词级时间戳
3. **仅 vLLM**: 仅支持 Qwen3-ASR vLLM 后端
4. **无断句**: 当前实现不包含自动句子边界检测

## 与 FunASR 流式的对比

| 特性 | FunASR WebSocket | Qwen3-ASR WebSocket |
|------|------------------|---------------------|
| 协议 | 阿里云兼容 | 自定义简化 |
| 流式机制 | 真正增量解码 | 累积重推理 |
| 延迟 | 低且恒定 | 随长度增加 |
| 模型支持 | Paraformer, Fun-ASR-Nano | 仅 Qwen3-ASR |
| 时间戳 | 支持 | 不支持 |
| 语言检测 | 有限 | 52种语言自动检测 |

## 后续优化方向

1. 添加句子边界检测（VAD 或基于语义）
2. 实现音频长度限制（防止显存溢出）
3. 添加阿里云协议兼容层
4. 优化重采样（使用更高效的算法）
