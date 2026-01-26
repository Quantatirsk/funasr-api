# -*- coding: utf-8 -*-
"""
说话人分离模块
基于 CAM++ 的说话人分离，用于多说话人音频分割
"""

import logging
import numpy as np
import librosa
import soundfile as sf
import tempfile
import os
import threading
from typing import List, Optional
from dataclasses import dataclass

from ..core.config import settings
from ..core.exceptions import DefaultServerErrorException

logger = logging.getLogger(__name__)

# 全局 CAM++ pipeline 缓存
_global_diarization_pipeline = None
_diarization_pipeline_lock = threading.Lock()


@dataclass
class SpeakerSegment:
    """说话人分段信息"""

    start_ms: int
    end_ms: int
    speaker_id: str
    audio_data: Optional[np.ndarray] = None
    temp_file: Optional[str] = None

    @property
    def start_sec(self) -> float:
        return self.start_ms / 1000.0

    @property
    def end_sec(self) -> float:
        return self.end_ms / 1000.0

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms

    @property
    def duration_sec(self) -> float:
        return self.duration_ms / 1000.0


def get_global_diarization_pipeline():
    """获取全局说话人分离 pipeline（懒加载单例）"""
    global _global_diarization_pipeline

    with _diarization_pipeline_lock:
        if _global_diarization_pipeline is None:
            try:
                from modelscope.pipelines import pipeline
                from modelscope.utils.constant import Tasks

                logger.info("正在加载 CAM++ 说话人分离模型...")
                _global_diarization_pipeline = pipeline(
                    task=Tasks.speaker_diarization,
                    model='iic/speech_campplus_speaker-diarization_common',
                )
                logger.info("CAM++ 模型加载成功")
            except Exception as e:
                logger.error(f"CAM++ 模型加载失败: {e}")
                raise DefaultServerErrorException(f"说话人分离模型加载失败: {str(e)}")

    return _global_diarization_pipeline


class SpeakerDiarizer:
    """基于 CAM++ 的说话人分离器"""

    DEFAULT_MAX_SEGMENT_SEC = 55.0
    DEFAULT_MIN_SEGMENT_SEC = 1.0
    DEFAULT_SAMPLE_RATE = 16000

    def __init__(
        self,
        max_segment_sec: float = DEFAULT_MAX_SEGMENT_SEC,
        min_segment_sec: float = DEFAULT_MIN_SEGMENT_SEC,
    ):
        self.max_segment_sec = max_segment_sec
        self.min_segment_sec = min_segment_sec
        self.max_segment_ms = int(max_segment_sec * 1000)
        self.min_segment_ms = int(min_segment_sec * 1000)

    def diarize(
        self, audio_path: str
    ) -> List[SpeakerSegment]:
        """执行说话人分离

        Args:
            audio_path: 音频文件路径

        Returns:
            原始分段列表（未合并）
        """
        try:
            pipeline = get_global_diarization_pipeline()

            logger.info(f"开始说话人分离: {audio_path}")
            result = pipeline(audio_path)

            # 解析结果: {'text': [[start, end, speaker_id], ...]}
            # pipeline 返回类型不确定，需要安全地获取 'text' 字段
            if isinstance(result, dict):
                raw_output = result.get('text', [])
            else:
                raw_output = getattr(result, 'text', []) or []

            segments = []
            for seg in raw_output:
                if isinstance(seg, list) and len(seg) == 3:
                    try:
                        start_ms = int(float(seg[0]) * 1000)
                        end_ms = int(float(seg[1]) * 1000)
                        speaker_id = f"说话人{int(seg[2]) + 1}"
                        segments.append(SpeakerSegment(
                            start_ms=start_ms,
                            end_ms=end_ms,
                            speaker_id=speaker_id,
                        ))
                    except (ValueError, TypeError) as e:
                        logger.warning(f"跳过格式错误的片段: {seg}, 错误: {e}")

            logger.info(f"说话人分离完成，原始片段数: {len(segments)}")
            return segments

        except Exception as e:
            logger.error(f"说话人分离失败: {e}")
            raise DefaultServerErrorException(f"说话人分离失败: {str(e)}")

    def merge_consecutive_segments(
        self, segments: List[SpeakerSegment]
    ) -> List[SpeakerSegment]:
        """合并同一说话人的连续片段"""
        if not segments:
            return []

        # 按开始时间排序
        sorted_segments = sorted(segments, key=lambda x: x.start_ms)

        merged = []
        current = SpeakerSegment(
            start_ms=sorted_segments[0].start_ms,
            end_ms=sorted_segments[0].end_ms,
            speaker_id=sorted_segments[0].speaker_id,
        )

        for seg in sorted_segments[1:]:
            if seg.speaker_id == current.speaker_id:
                # 同一说话人，扩展结束时间
                current.end_ms = max(current.end_ms, seg.end_ms)
            else:
                # 不同说话人，保存当前段，开始新段
                merged.append(current)
                current = SpeakerSegment(
                    start_ms=seg.start_ms,
                    end_ms=seg.end_ms,
                    speaker_id=seg.speaker_id,
                )

        # 保存最后一段
        merged.append(current)

        logger.info(f"合并同一说话人连续片段: {len(segments)} → {len(merged)}")
        return merged

    def split_long_segments(
        self, segments: List[SpeakerSegment]
    ) -> List[SpeakerSegment]:
        """将超过最大时长的片段切分"""
        result = []

        for seg in segments:
            if seg.duration_ms <= self.max_segment_ms:
                result.append(seg)
            else:
                # 需要切分
                current_start = seg.start_ms
                while current_start < seg.end_ms:
                    current_end = min(current_start + self.max_segment_ms, seg.end_ms)
                    # 确保最后一段不会太短
                    remaining = seg.end_ms - current_end
                    if 0 < remaining < self.min_segment_ms:
                        current_end = seg.end_ms

                    result.append(SpeakerSegment(
                        start_ms=current_start,
                        end_ms=current_end,
                        speaker_id=seg.speaker_id,
                    ))
                    current_start = current_end

        split_count = len(result) - len(segments)
        if split_count > 0:
            logger.info(f"切分超长片段: {len(segments)} → {len(result)} (+{split_count})")

        return result

    def split_audio_by_speakers(
        self,
        audio_path: str,
        output_dir: Optional[str] = None,
    ) -> List[SpeakerSegment]:
        """完整的说话人分离流程

        流程：
        1. 执行说话人分离
        2. 合并同一说话人连续片段
        3. 切分超长片段
        4. 提取音频数据，保存临时文件

        Args:
            audio_path: 音频文件路径
            output_dir: 输出目录

        Returns:
            SpeakerSegment 列表
        """
        try:
            # 1. 执行说话人分离
            raw_segments = self.diarize(audio_path)

            if not raw_segments:
                logger.warning("说话人分离未检测到任何片段")
                return []

            # 2. 合并同一说话人连续片段
            merged_segments = self.merge_consecutive_segments(raw_segments)

            # 3. 切分超长片段
            final_segments = self.split_long_segments(merged_segments)

            # 4. 加载音频并提取片段
            logger.info("加载音频并提取片段...")
            audio_data, sr = librosa.load(audio_path, sr=self.DEFAULT_SAMPLE_RATE)

            output_dir = output_dir or settings.TEMP_DIR
            os.makedirs(output_dir, exist_ok=True)

            for idx, seg in enumerate(final_segments):
                start_sample = int(seg.start_ms / 1000 * sr)
                end_sample = int(seg.end_ms / 1000 * sr)

                seg.audio_data = audio_data[start_sample:end_sample]

                # 保存临时文件
                temp_file = tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".wav",
                    dir=output_dir,
                    prefix=f"{seg.speaker_id}_{idx:03d}_",
                )
                temp_path = temp_file.name
                temp_file.close()

                sf.write(temp_path, seg.audio_data, sr)
                seg.temp_file = temp_path

            # 统计
            unique_speakers = sorted(set(seg.speaker_id for seg in final_segments))
            logger.info(
                f"音频分割完成: {len(final_segments)} 个片段, "
                f"{len(unique_speakers)} 个说话人"
            )
            for spk in unique_speakers:
                spk_segs = [s for s in final_segments if s.speaker_id == spk]
                total_time = sum(s.duration_sec for s in spk_segs)
                logger.info(f"  {spk}: {len(spk_segs)} 片段, {total_time:.2f}s")

            return final_segments

        except Exception as e:
            logger.error(f"说话人分离流程失败: {e}")
            raise DefaultServerErrorException(f"说话人分离失败: {str(e)}")

    @staticmethod
    def cleanup_segments(segments: List[SpeakerSegment]) -> None:
        """清理临时文件"""
        for seg in segments:
            if seg.temp_file and os.path.exists(seg.temp_file):
                try:
                    os.remove(seg.temp_file)
                except Exception as e:
                    logger.warning(f"清理临时文件失败: {seg.temp_file}, {e}")
