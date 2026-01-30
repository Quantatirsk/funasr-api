# Qwen3-ASR ForcedAligner 字词级时间戳研究报告

## 概述

Qwen3-ASR（0.6B/1.7B）通过 **Qwen3-ForcedAligner** 模型支持字词级时间戳功能。这是一个独立的对齐模型，可以将语音识别结果与音频进行精确对齐，输出每个字符或词语的开始和结束时间。

## ForcedAligner 工作原理

### 1. 架构组成

```
Qwen3-ForcedAligner
├── Qwen3ForceAlignProcessor    # 文本预处理器
│   ├── 中文分词（逐字）
│   ├── 英文分词（空格分隔）
│   ├── 日文分词（Nagisa）
│   └── 韩文分词（SoyNLP）
├── Qwen3ASRForConditionalGeneration  # 对齐模型
└── 时间戳后处理器
    ├── LIS（最长递增子序列）修正
    └── 异常时间戳插值修复
```

### 2. 处理流程

```python
# 1. ASR 识别（主模型）
text = "你好世界"

# 2. 文本分词（根据语言）
word_list = ["你", "好", "世", "界"]

# 3. 构造对齐输入
aligner_input = "<|audio_start|><|audio_pad|><|audio_end|>你<timestamp><timestamp>好<timestamp><timestamp>世<timestamp><timestamp>界<timestamp><timestamp>"

# 4. ForcedAligner 推理
logits = model.thinker(**inputs).logits
timestamp_token_ids = logits.argmax(dim=-1)

# 5. 解码时间戳
timestamp_ms = token_ids * timestamp_segment_time
start_time = timestamp_ms[i * 2]
end_time = timestamp_ms[i * 2 + 1]
```

### 3. 输出格式

```python
@dataclass
class ForcedAlignItem:
    text: str        # 字符或词语
    start_time: int  # 开始时间（毫秒）
    end_time: int    # 结束时间（毫秒）

@dataclass
class ForcedAlignResult:
    items: List[ForcedAlignItem]
```

示例输出：
```json
{
  "items": [
    {"text": "你", "start_time": 0.0, "end_time": 0.35},
    {"text": "好", "start_time": 0.35, "end_time": 0.68},
    {"text": "世", "start_time": 0.68, "end_time": 0.92},
    {"text": "界", "start_time": 0.92, "end_time": 1.15}
  ]
}
```

## 多语言分词策略

### 中文（Chinese）
- **策略**：逐字分割（CJK Unified Ideographs）
- **处理逻辑**：每个汉字独立成词
- **示例**：`"你好世界"` → `["你", "好", "世", "界"]`

### 英文（English）
- **策略**：空格分隔 + 标点清理
- **处理逻辑**：按空格分词，保留字母和数字
- **示例**：`"Hello world"` → `["Hello", "world"]`

### 日文（Japanese）
- **策略**：Nagisa 分词器
- **处理逻辑**：使用 Nagisa 进行形态学分析
- **依赖**：`nagisa` 库
- **示例**：`"こんにちは"` → `["こんにちは"]`

### 韩文（Korean）
- **策略**：SoyNLP LTokenizer
- **处理逻辑**：使用自定义词典进行分词
- **依赖**：`soynlp` 库
- **词典**：`korean_dict_jieba.dict`

## 时间戳修正算法

### 1. LIS（最长递增子序列）检测

```python
def fix_timestamp(data):
    # 找出最长递增子序列（正常的时间戳应该是递增的）
    dp = [1] * n
    parent = [-1] * n

    for i in range(1, n):
        for j in range(i):
            if data[j] <= data[i] and dp[j] + 1 > dp[i]:
                dp[i] = dp[j] + 1
                parent[i] = j

    # 标记异常点
    is_normal = [False] * n
    for idx in lis_indices:
        is_normal[idx] = True
```

### 2. 异常时间戳修复

对于被标记为异常的时间戳，使用线性插值修复：

```python
# 小段异常（<=2个）：就近对齐
if anomaly_count <= 2:
    result[k] = left_val if (k - left) <= (right - k) else right_val

# 大段异常（>2个）：线性插值
else:
    step = (right_val - left_val) / (anomaly_count + 1)
    result[k] = left_val + step * (k - i + 1)
```

## 本项目实现现状

### 已实现功能

1. **模型配置**（`models.json`）
   ```json
   {
     "qwen3-asr-1.7b": {
       "forced_aligner_path": "Qwen/Qwen3-ForcedAligner-0.6B"
     }
   }
   ```

2. **引擎初始化**（`qwen3_engine.py`）
   ```python
   self.model = Qwen3ASRModel.LLM(
       model=model_path,
       forced_aligner=forced_aligner_path,      # 启用时间戳模型
       forced_aligner_kwargs={...},              # 配置参数
       ...
   )
   ```

3. **时间戳获取**（`transcribe_file_with_vad`）
   ```python
   results = self.model.transcribe(
       audio=audio_path,
       return_time_stamps=True,  # 启用时间戳
   )
   ```

### 当前限制

1. **聚合为句子级别**：当前实现将字级时间戳聚合成句子级别
   ```python
   # 当前：聚合为句子
   for item in time_stamps.items:
       current_text += item.text
       if char in "。！？；":
           # 保存整句，丢失字级信息
           segments.append(ASRSegmentResult(...))
   ```

2. **API 响应不含字词级时间戳**：仅返回句子级 `segments`

## 建议改进方案

### 方案 1：扩展数据结构（推荐）

在 `ASRSegmentResult` 中添加 `word_tokens` 字段：

```python
@dataclass
class WordToken:
    text: str
    start_time: float
    end_time: float

@dataclass
class ASRSegmentResult:
    text: str
    start_time: float
    end_time: float
    speaker_id: Optional[str] = None
    word_tokens: Optional[List[WordToken]] = None  # 新增
```

### 方案 2：API 参数控制

添加 `word_timestamps` 参数：

```python
# 请求参数
{
  "word_timestamps": true  # 新增参数
}

# 响应格式
{
  "result": "你好世界",
  "segments": [{
    "text": "你好世界",
    "start_time": 0.0,
    "end_time": 1.15,
    "word_tokens": [  # 新增字段
      {"text": "你", "start_time": 0.0, "end_time": 0.35},
      {"text": "好", "start_time": 0.35, "end_time": 0.68},
      {"text": "世", "start_time": 0.68, "end_time": 0.92},
      {"text": "界", "start_time": 0.92, "end_time": 1.15}
    ]
  }]
}
```

### 方案 3：ForcedAligner 独立 API

提供独立的强制对齐接口：

```python
POST /v1/audio/align
{
  "audio": "...",
  "text": "你好世界"
}

Response:
{
  "items": [
    {"text": "你", "start_time": 0.0, "end_time": 0.35},
    ...
  ]
}
```

## 性能考量

### 时间戳生成开销

| 阶段 | 耗时 | 说明 |
|------|------|------|
| ASR 识别 | ~0.1×RT | 主要耗时 |
| ForcedAligner | ~0.05×RT | 轻量级模型 |
| 总耗时 | ~0.15×RT | 比纯 ASR 慢约 50% |

### 内存占用

- **Qwen3-ForcedAligner-0.6B**：约 1.2GB GPU 显存
- **建议**：与主模型共用同一张 GPU 卡

### 长音频处理

```python
# 自动分段处理
max_chunk_sec = 60  # 每段最大 60 秒

# 分段对齐后合并
aligned_results = []
for chunk in audio_chunks:
    result = forced_aligner.align(chunk.audio, chunk.text, chunk.language)
    aligned_results.append(result)

# 合并时间戳（加上时间偏移）
final_result = merge_aligned_results(aligned_results, chunk_offsets)
```

## 使用建议

### 1. 何时启用字词级时间戳

**建议使用场景**：
- 字幕生成（需要精确到字的显示时间）
- 语音编辑（需要定位到特定词语）
- 语音评测（需要评估每个字的发音）

**不建议使用场景**：
- 纯文本转写（增加不必要的计算）
- 实时流式识别（ForcedAligner 只支持离线）

### 2. 配置参数优化

```python
# 批处理大小
max_inference_batch_size=32  # 根据 GPU 显存调整

# 数据类型
dtype=torch.bfloat16  # 显存紧张时使用 fp16

# 设备映射
device_map="cuda:0"  # 指定 GPU 设备
```

## 参考

- Qwen3-ASR 源码：`ref/Qwen3-ASR/qwen_asr/inference/qwen3_forced_aligner.py`
- 本项目实现：`app/services/asr/qwen3_engine.py`
- HuggingFace 模型：[Qwen/Qwen3-ForcedAligner-0.6B](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B)
