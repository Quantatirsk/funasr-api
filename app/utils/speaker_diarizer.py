# -*- coding: utf-8 -*-
"""
说话人分离模块
基于 CAM++ 的说话人分离，用于多说话人音频分割
"""

from loguru import logger
import numpy as np
import librosa
import soundfile as sf
import tempfile
import os
import threading
from typing import List, Optional, Tuple
from dataclasses import dataclass

from ..core.config import settings
from ..core.exceptions import DefaultServerErrorException

# 全局 CAM++ pipeline 缓存（单例）
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
                from ..services.asr.engine import resolve_model_path

                model_id = 'iic/speech_campplus_speaker-diarization_common'
                model_path = resolve_model_path(model_id)

                logger.info(f"正在加载 CAM++ 说话人分离模型: {model_path}")
                _global_diarization_pipeline = pipeline(
                    task=Tasks.speaker_diarization,
                    model=model_path,
                )
                logger.info("CAM++ 模型加载成功")
            except Exception as e:
                logger.error(f"CAM++ 模型加载失败: {e}")
                raise DefaultServerErrorException(f"说话人分离模型加载失败: {str(e)}")

    return _global_diarization_pipeline


class SpeakerDiarizer:
    """基于 CAM++ 的说话人分离器"""

    DEFAULT_MAX_SEGMENT_SEC = 80.0
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
            # 诊断日志：打印前20个原始片段
            for i, seg in enumerate(segments[:20]):
                logger.debug(
                    f"[CAM++原始] #{i}: {seg.start_sec:.2f}-{seg.end_sec:.2f}s "
                    f"({seg.duration_sec:.2f}s) {seg.speaker_id}"
                )
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
                logger.debug(
                    f"[合并中断] 说话人切换: {current.speaker_id} → {seg.speaker_id} "
                    f"在 {seg.start_sec:.2f}s，保存片段 {current.start_sec:.2f}-{current.end_sec:.2f}s"
                )
                merged.append(current)
                current = SpeakerSegment(
                    start_ms=seg.start_ms,
                    end_ms=seg.end_ms,
                    speaker_id=seg.speaker_id,
                )

        # 保存最后一段
        merged.append(current)

        logger.info(f"合并同一说话人连续片段: {len(segments)} → {len(merged)}")
        # 诊断日志：打印合并后的前20个片段
        for i, seg in enumerate(merged[:20]):
            logger.debug(
                f"[合并后] #{i}: {seg.start_sec:.2f}-{seg.end_sec:.2f}s "
                f"({seg.duration_sec:.2f}s) {seg.speaker_id}"
            )
        return merged

    def split_long_segments(
        self,
        audio_path: str,
        segments: List[SpeakerSegment]
    ) -> List[SpeakerSegment]:
        """将超过最大时长的片段使用VAD智能切分，并贪婪合并可合并的连续片段

        利用 AudioSplitter 的 VAD 能力，按语音边界切分，
        避免在说话中间切断。同时在处理过程中贪婪合并同一说话人的连续片段。
        """
        from .audio_splitter import AudioSplitter

        result = []
        splitter = AudioSplitter(max_segment_sec=self.max_segment_sec)

        # 只获取一次VAD（优化：避免重复调用）
        all_vad_segments = None

        for seg in segments:
            # 核心优化：先尝试和result中最后一个片段合并
            if self._try_merge_with_last(result, seg):
                logger.debug(
                    f"[贪婪合并成功] {seg.start_sec:.2f}-{seg.end_sec:.2f}s "
                    f"合并到前一个片段，新范围: {result[-1].start_sec:.2f}-{result[-1].end_sec:.2f}s"
                )
                continue  # 合并成功，处理下一个片段

            # 不能合并，检查是否需要切分
            if seg.duration_ms <= self.max_segment_ms:
                logger.debug(
                    f"[直接添加] {seg.start_sec:.2f}-{seg.end_sec:.2f}s "
                    f"({seg.duration_sec:.2f}s) {seg.speaker_id}"
                )
                result.append(seg)
            else:
                # 使用 VAD 智能切分
                logger.info(f"说话人 {seg.speaker_id} 片段 {seg.start_ms}ms-{seg.end_ms}ms 超长，使用VAD切分")

                try:
                    # 延迟获取VAD（只在需要时获取）
                    if all_vad_segments is None:
                        all_vad_segments = splitter.get_vad_segments(audio_path)

                    # 过滤出完全在该段范围内的 VAD 段
                    # 关键修改：只使用完全包含在说话人片段内的 VAD 边界
                    seg_vad_segments = [
                        (max(start, seg.start_ms), min(end, seg.end_ms))
                        for start, end in all_vad_segments
                        if start < seg.end_ms and end > seg.start_ms  # 有交集
                    ]

                    if seg_vad_segments:
                        logger.debug(f"[VAD过滤] 找到 {len(seg_vad_segments)} 个VAD段")
                        # 直接使用 VAD 边界，贪婪合并确保不超过最大时长
                        sub_segments = self._merge_vad_for_speaker(
                            seg_vad_segments, seg.start_ms, seg.end_ms, seg.speaker_id
                        )
                        logger.info(
                            f"[VAD切分] {seg.speaker_id} "
                            f"{seg.start_sec:.2f}-{seg.end_sec:.2f}s → {len(sub_segments)} 个子片段"
                        )
                        # 逐个添加子片段，尝试与前一个片段合并
                        for i, sub_seg in enumerate(sub_segments):
                            logger.debug(
                                f"  子片段{i}: {sub_seg.start_sec:.2f}-{sub_seg.end_sec:.2f}s "
                                f"({sub_seg.duration_sec:.2f}s)"
                            )
                            if not self._try_merge_with_last(result, sub_seg):
                                result.append(sub_seg)
                    else:
                        # VAD 未检测到语音，fallback 到硬切
                        logger.warning(f"VAD 未检测到语音边界，fallback 到硬切")
                        hard_split_segments = self._hard_split_segment(seg)
                        for sub_seg in hard_split_segments:
                            if not self._try_merge_with_last(result, sub_seg):
                                result.append(sub_seg)

                except Exception as e:
                    logger.error(f"VAD 切分失败，fallback 到硬切: {e}")
                    hard_split_segments = self._hard_split_segment(seg)
                    for sub_seg in hard_split_segments:
                        if not self._try_merge_with_last(result, sub_seg):
                            result.append(sub_seg)

        logger.info(f"切分并合并完成: {len(segments)} → {len(result)}")
        # 诊断日志：打印最终的前30个片段
        for i, seg in enumerate(result[:30]):
            logger.info(
                f"[最终片段] #{i}: {seg.start_sec:.2f}-{seg.end_sec:.2f}s "
                f"({seg.duration_sec:.2f}s) {seg.speaker_id}"
            )
        return result

    def _try_merge_with_last(
        self,
        result: List[SpeakerSegment],
        seg: SpeakerSegment
    ) -> bool:
        """尝试将seg合并到result的最后一个片段

        Args:
            result: 已处理的片段列表
            seg: 待合并的片段

        Returns:
            True if merged, False otherwise
        """
        if not result:
            return False

        last = result[-1]

        # 必须是同一说话人
        if last.speaker_id != seg.speaker_id:
            logger.debug(
                f"[合并失败-说话人] {last.speaker_id} != {seg.speaker_id} "
                f"at {seg.start_sec:.2f}s"
            )
            return False

        # 检查合并后是否超过最大时长
        merged_duration_ms = seg.end_ms - last.start_ms
        if merged_duration_ms > self.max_segment_ms:
            logger.debug(
                f"[合并失败-超时] {last.start_sec:.2f}-{seg.end_sec:.2f}s = "
                f"{merged_duration_ms/1000:.2f}s > {self.max_segment_sec}s"
            )
            return False

        # 可以合并，直接扩展最后一个片段的结束时间
        last.end_ms = seg.end_ms
        logger.debug(
            f"[合并成功] {last.speaker_id} "
            f"{last.start_ms}-{last.end_ms}ms ({merged_duration_ms/1000:.2f}s)"
        )
        return True

    def _merge_vad_for_speaker(
        self,
        vad_segments: List[Tuple[int, int]],
        speaker_start: int,
        speaker_end: int,
        speaker_id: str
    ) -> List[SpeakerSegment]:
        """为单个说话人合并VAD段，确保每段不超过最大时长

        与 AudioSplitter.merge_segments_greedy 不同，这里：
        1. 不假设从0开始
        2. 只处理已过滤的 VAD 段
        3. 确保结果连续无间隙
        """
        if not vad_segments:
            return []

        # 按开始时间排序
        sorted_vad = sorted(vad_segments, key=lambda x: x[0])

        merged = []
        current_start = sorted_vad[0][0]
        current_end = sorted_vad[0][1]

        for vad_start, vad_end in sorted_vad[1:]:
            # 检查合并后是否超过最大时长
            if vad_end - current_start <= self.max_segment_ms:
                # 可以合并，扩展当前段
                current_end = vad_end
            else:
                # 超过限制，保存当前段，开始新段
                if current_end - current_start >= self.min_segment_ms:
                    merged.append(SpeakerSegment(
                        start_ms=current_start,
                        end_ms=current_end,
                        speaker_id=speaker_id,
                    ))
                current_start = vad_start
                current_end = vad_end

        # 保存最后一段
        if current_end - current_start >= self.min_segment_ms:
            merged.append(SpeakerSegment(
                start_ms=current_start,
                end_ms=current_end,
                speaker_id=speaker_id,
            ))

        # 使用 speaker_start 和 speaker_end 限制结果边界
        # 确保所有段都在说话人片段范围内
        final_merged = []
        for seg in merged:
            actual_start = max(seg.start_ms, speaker_start)
            actual_end = min(seg.end_ms, speaker_end)
            if actual_end - actual_start >= self.min_segment_ms:
                final_merged.append(SpeakerSegment(
                    start_ms=actual_start,
                    end_ms=actual_end,
                    speaker_id=speaker_id,
                ))

        return final_merged

    def _hard_split_segment(self, seg: SpeakerSegment) -> List[SpeakerSegment]:
        """硬性切分（fallback）

        Args:
            seg: 待切分的片段

        Returns:
            切分后的片段列表
        """
        segments = []
        current_start = seg.start_ms
        while current_start < seg.end_ms:
            current_end = min(current_start + self.max_segment_ms, seg.end_ms)
            remaining = seg.end_ms - current_end
            if 0 < remaining < self.min_segment_ms:
                current_end = seg.end_ms

            segments.append(SpeakerSegment(
                start_ms=current_start,
                end_ms=current_end,
                speaker_id=seg.speaker_id,
            ))
            current_start = current_end

        return segments

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

            # 3. 切分超长片段（使用VAD智能切分）
            final_segments = self.split_long_segments(audio_path, merged_segments)

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
