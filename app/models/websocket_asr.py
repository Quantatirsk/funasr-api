# -*- coding: utf-8 -*-
"""
WebSocket ASR 数据模型 - 阿里云协议
"""

from typing import Optional, Dict, Any, Union, List
from pydantic import BaseModel, Field, field_validator
import uuid


class AliyunASRWSHeader(BaseModel):
    """阿里云WebSocket ASR消息头部"""

    message_id: str = Field(..., description="消息ID，32位唯一ID")
    task_id: str = Field(..., description="任务ID，32位唯一ID")
    namespace: str = Field(..., description="命名空间，固定为SpeechTranscriber")
    name: str = Field(..., description="消息名称")
    appkey: Optional[str] = Field(None, description="应用密钥")
    status: Optional[int] = Field(None, description="状态码")
    status_text: Optional[str] = Field(None, description="状态文本")
    status_message: Optional[str] = Field(None, description="状态消息")

    @staticmethod
    def generate_message_id() -> str:
        """生成32位消息ID"""
        return str(uuid.uuid4()).replace("-", "")[:32]


class AliyunStartTranscriptionPayload(BaseModel):
    """StartTranscription 消息负载"""

    format: str = Field(default="pcm", description="音频格式: pcm, wav, opus, speex, amr, mp3, aac")
    sample_rate: int = Field(default=16000, description="音频采样率: 8000/16000")
    enable_intermediate_result: bool = Field(default=True, description="是否返回中间识别结果")
    enable_punctuation_prediction: bool = Field(default=True, description="是否在后处理中添加标点")
    enable_inverse_text_normalization: bool = Field(default=True, description="是否将中文数字转为阿拉伯数字")
    customization_id: Optional[str] = Field(None, description="自学习模型ID")
    vocabulary_id: Optional[str] = Field(None, description="定制泛热词ID")
    max_sentence_silence: int = Field(default=800, ge=200, le=2000, description="语音断句检测阈值(ms)")
    enable_words: bool = Field(default=False, description="是否开启返回词信息")
    disfluency: bool = Field(default=False, description="过滤语气词")
    speech_noise_threshold: Optional[float] = Field(None, ge=-1.0, le=1.0, description="噪音参数阈值")
    enable_semantic_sentence_detection: bool = Field(default=False, description="是否开启语义断句")

    @field_validator("format")
    @classmethod
    def validate_format(cls, v):
        supported_formats = ["pcm", "wav", "opus", "speex", "amr", "mp3", "aac"]
        if v.lower() not in supported_formats:
            raise ValueError(f"不支持的音频格式: {v}")
        return v.lower()

    @field_validator("sample_rate")
    @classmethod
    def validate_sample_rate(cls, v):
        supported_rates = [8000, 16000]
        if v not in supported_rates:
            raise ValueError(f"不支持的采样率: {v}")
        return v


class AliyunWordInfo(BaseModel):
    """词信息"""

    text: str = Field("", description="文本")
    startTime: int = Field(0, description="词开始时间(ms)")
    endTime: int = Field(0, description="词结束时间(ms)")


class AliyunTranscriptionResultPayload(BaseModel):
    """识别结果负载"""

    session_id: Optional[str] = Field(None, description="会话ID")
    index: Optional[int] = Field(None, description="句子编号，从1开始递增")
    time: Optional[int] = Field(None, description="已处理的音频时长(ms)")
    begin_time: Optional[int] = Field(None, description="句子开始时间(ms)")
    result: Optional[str] = Field(None, description="识别结果文本")
    confidence: Optional[float] = Field(None, description="置信度[0.0,1.0]")
    words: Optional[List[AliyunWordInfo]] = Field(None, description="词信息列表")
    status: Optional[int] = Field(None, description="状态码")


class AliyunStashResult(BaseModel):
    """暂存结果（语义断句）"""

    sentenceId: int = Field(0, description="句子编号")
    beginTime: int = Field(0, description="句子开始时间(ms)")
    text: str = Field("", description="转写内容")
    currentTime: int = Field(0, description="当前处理时间(ms)")


class AliyunASRWSMessage(BaseModel):
    """阿里云WebSocket ASR消息"""

    header: AliyunASRWSHeader = Field(..., description="消息头部")
    payload: Optional[
        Union[
            AliyunStartTranscriptionPayload,
            AliyunTranscriptionResultPayload,
            Dict[str, Any],
        ]
    ] = Field(None, description="消息负载")


class AliyunASRNamespace:
    """阿里云ASR命名空间"""

    SPEECH_TRANSCRIBER = "SpeechTranscriber"


class AliyunASRMessageName:
    """阿里云ASR消息名称"""

    START_TRANSCRIPTION = "StartTranscription"
    STOP_TRANSCRIPTION = "StopTranscription"

    TRANSCRIPTION_STARTED = "TranscriptionStarted"
    SENTENCE_BEGIN = "SentenceBegin"
    TRANSCRIPTION_RESULT_CHANGED = "TranscriptionResultChanged"
    SENTENCE_END = "SentenceEnd"
    TRANSCRIPTION_COMPLETED = "TranscriptionCompleted"
    TASK_FAILED = "TaskFailed"


class AliyunASRStatus:
    """阿里云ASR状态码"""

    SUCCESS = 20000000
    TASK_FAILED = 40000000
    INVALID_PARAMETER = 40000001
    MESSAGE_INVALID = 40000002
    AUTHENTICATION_FAILED = 40100005
    QUOTA_EXCEEDED = 40300016
    INTERNAL_ERROR = 50000000
    SERVICE_UNAVAILABLE = 50300018