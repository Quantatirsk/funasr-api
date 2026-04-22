# -*- coding: utf-8 -*-
"""Qwen Rust CPU end-to-end benchmark for the current runtime configuration."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from app.core.config import settings
from app.services.asr.qwen3_engine import Qwen3ASREngine
from app.utils.audio import get_audio_duration
from app.utils.audio_splitter import AudioSplitter


@dataclass
class WorkerBenchRow:
    cpu_count: int
    rust_workers: int
    asr_concurrency: int
    align_concurrency: int
    audio_file: str
    audio_duration_sec: float
    batch_size: int
    engine_init_sec: float
    vad_sec: float
    vad_segments: int
    asr_sec: float
    asr_calls: int
    align_sec: float
    align_calls: int
    total_sec: float
    rtf: float
    segments: int
    word_tokens: int
    text_len: int


def _persist_rows(rows: list[WorkerBenchRow], json_out: Path | None) -> None:
    if json_out is None:
        return
    payload = [asdict(row) for row in rows]
    tmp_path = json_out.with_suffix(f"{json_out.suffix}.tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(json_out)


def _render_markdown(rows: list[WorkerBenchRow]) -> str:
    if not rows:
        return "# Qwen Rust CPU 对比结果\n\n暂无结果。\n"

    audio_file = rows[0].audio_file
    audio_duration_sec = rows[0].audio_duration_sec
    batch_size = rows[0].batch_size
    cpu_count = rows[0].cpu_count

    lines = [
        "# Qwen Rust CPU 对比结果",
        "",
        f"- 音频文件：`{audio_file}`",
        f"- 音频时长：`{audio_duration_sec:.2f}s`",
        f"- batch size：`{batch_size}`",
        f"- CPU 数量：`{cpu_count}`",
        "",
        f"- Rust workers：`{rows[0].rust_workers}`",
        f"- ASR concurrency：`{rows[0].asr_concurrency}`",
        f"- Align concurrency：`{rows[0].align_concurrency}`",
        "",
        "| total_sec | RTF | engine_init_sec | vad_sec | asr_sec | align_sec | segments | word_tokens | text_len |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for row in rows:
        lines.append(
            "| "
            f"{row.total_sec:.2f} | "
            f"{row.rtf:.4f} | "
            f"{row.engine_init_sec:.3f} | "
            f"{row.vad_sec:.3f} | "
            f"{row.asr_sec:.2f} | "
            f"{row.align_sec:.2f} | "
            f"{row.segments} | "
            f"{row.word_tokens} | "
            f"{row.text_len} |"
        )

    row = rows[0]

    lines.extend(
        [
            "",
            "## 结论",
            "",
            f"- 当前配置：workers=`{row.rust_workers}` / asr=`{row.asr_concurrency}` / align=`{row.align_concurrency}`",
            f"- 总耗时：`{row.total_sec:.2f}s`",
            f"- RTF：`{row.rtf:.4f}`",
            "",
        ]
    )
    return "\n".join(lines)


def _persist_markdown(rows: list[WorkerBenchRow], markdown_out: Path | None) -> None:
    if markdown_out is None:
        return
    tmp_path = markdown_out.with_suffix(f"{markdown_out.suffix}.tmp")
    tmp_path.write_text(_render_markdown(rows), encoding="utf-8")
    tmp_path.replace(markdown_out)


def _log_progress(row: WorkerBenchRow) -> None:
    print(
        (
            f"[bench] workers={row.rust_workers} "
            f"asr={row.asr_concurrency} "
            f"align={row.align_concurrency} "
            f"total={row.total_sec:.2f}s "
            f"rtf={row.rtf:.4f} "
            f"asr_sec={row.asr_sec:.2f}s "
            f"align_sec={row.align_sec:.2f}s "
            f"segments={row.segments} "
            f"words={row.word_tokens}"
        ),
        file=sys.stderr,
        flush=True,
)


def _clean_segments(segments: list) -> None:
    AudioSplitter.cleanup_segments(segments)


def _prepare_segments(audio_file: Path) -> tuple[float, float, list]:
    duration = get_audio_duration(str(audio_file))
    splitter = AudioSplitter(device="cpu")
    t0 = time.perf_counter()
    segments = splitter.split_audio_file(str(audio_file))
    vad_sec = time.perf_counter() - t0
    if not segments:
        raise RuntimeError("VAD returned no segments")
    return duration, vad_sec, segments


def _build_engine(
    model_path: str,
    forced_aligner_path: str,
    *,
    batch_size: int,
) -> tuple[Qwen3ASREngine, float]:
    settings.DEVICE = "cpu"
    settings.ASR_BATCH_SIZE = batch_size

    t0 = time.perf_counter()
    engine = Qwen3ASREngine(
        model_path=model_path,
        forced_aligner_path=forced_aligner_path,
        device="cpu",
    )
    return engine, time.perf_counter() - t0


def _run_asr_stage(
    engine: Qwen3ASREngine,
    segments: list,
) -> tuple[dict[int, str], float]:
    valid_segments = [
        (idx, seg) for idx, seg in enumerate(segments) if getattr(seg, "temp_file", None)
    ]
    t0 = time.perf_counter()
    texts = engine._run_rust_asr_stage(
        valid_segments=valid_segments,
        hotwords="",
        enable_punctuation=True,
        enable_itn=True,
        sample_rate=16000,
    )
    return texts, time.perf_counter() - t0


def _run_align_stage(
    engine: Qwen3ASREngine,
    segments: list,
    texts: dict[int, str],
) -> tuple[dict[int, list], float]:
    valid_segments = [
        (idx, seg) for idx, seg in enumerate(segments) if getattr(seg, "temp_file", None)
    ]
    t0 = time.perf_counter()
    aligned = engine._run_rust_align_stage(
        valid_segments=valid_segments,
        texts=texts,
    )
    return aligned, time.perf_counter() - t0


def _summarize(
    *,
    cpu_count: int,
    audio_file: Path,
    audio_duration_sec: float,
    batch_size: int,
    engine_init_sec: float,
    vad_sec: float,
    segments: list,
    texts: dict[int, str],
    aligned: dict[int, list],
    asr_sec: float,
    align_sec: float,
    total_sec: float,
) -> WorkerBenchRow:
    text_len = sum(len(text) for text in texts.values())
    word_tokens = sum(len(items) for items in aligned.values())
    return WorkerBenchRow(
        cpu_count=cpu_count,
        rust_workers=settings.QWEN_RUST_CPU_WORKERS,
        asr_concurrency=settings.QWEN_RUST_ASR_CONCURRENCY or settings.QWEN_RUST_CPU_WORKERS,
        align_concurrency=settings.QWEN_RUST_ALIGN_CONCURRENCY or settings.QWEN_RUST_CPU_WORKERS,
        audio_file=str(audio_file),
        audio_duration_sec=audio_duration_sec,
        batch_size=batch_size,
        engine_init_sec=engine_init_sec,
        vad_sec=vad_sec,
        vad_segments=len(segments),
        asr_sec=asr_sec,
        asr_calls=len(texts),
        align_sec=align_sec,
        align_calls=len(aligned),
        total_sec=total_sec,
        rtf=total_sec / audio_duration_sec if audio_duration_sec else 0.0,
        segments=len(texts),
        word_tokens=word_tokens,
        text_len=text_len,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Qwen Rust CPU end-to-end benchmark")
    parser.add_argument("--audio-file", required=True, help="Input audio file path")
    parser.add_argument("--model-path", default="Qwen/Qwen3-ASR-0.6B")
    parser.add_argument("--forced-aligner-path", default="Qwen/Qwen3-ForcedAligner-0.6B")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--json-out", help="Optional JSON output path")
    parser.add_argument(
        "--markdown-out",
        help="Optional Markdown report output path. Defaults to a sibling .md next to --json-out.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    audio_file = Path(args.audio_file).expanduser().resolve()
    cpu_count = os.cpu_count() or 1
    out_path = Path(args.json_out).expanduser().resolve() if args.json_out else None
    markdown_out = Path(args.markdown_out).expanduser().resolve() if args.markdown_out else None
    if markdown_out is None and out_path is not None:
        markdown_out = out_path.with_suffix(".md")

    duration, vad_sec, segments = _prepare_segments(audio_file)
    rows: list[WorkerBenchRow] = []

    try:
        engine, init_sec = _build_engine(
            args.model_path,
            args.forced_aligner_path,
            batch_size=args.batch_size,
        )

        t0 = time.perf_counter()
        texts, asr_sec = _run_asr_stage(engine, segments)
        aligned, align_sec = _run_align_stage(engine, segments, texts)
        total_sec = time.perf_counter() - t0

        row = _summarize(
            cpu_count=cpu_count,
            audio_file=audio_file,
            audio_duration_sec=duration,
            batch_size=args.batch_size,
            engine_init_sec=init_sec,
            vad_sec=vad_sec,
            segments=segments,
            texts=texts,
            aligned=aligned,
            asr_sec=asr_sec,
            align_sec=align_sec,
            total_sec=total_sec,
        )
        rows.append(row)
        _persist_rows(rows, out_path)
        _persist_markdown(rows, markdown_out)
        _log_progress(row)
    finally:
        _clean_segments(segments)

    payload = [asdict(row) for row in rows]
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if markdown_out is not None:
        markdown_out.parent.mkdir(parents=True, exist_ok=True)
        markdown_out.write_text(_render_markdown(rows), encoding="utf-8")

    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
