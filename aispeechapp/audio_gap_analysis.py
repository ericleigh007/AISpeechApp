from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from aispeechapp.candidates import PROJECT_ROOT


@dataclass(frozen=True)
class AudioGap:
    start_s: float
    duration_s: float
    peak_db: float


@dataclass(frozen=True)
class WavGapReport:
    path: str
    sample_rate: int
    duration_s: float
    peak: float
    silence_threshold_db: float
    max_gap_s: float
    gap_count: int
    gaps: list[AudioGap]


@dataclass(frozen=True)
class TimingGap:
    previous_chunk: int
    next_chunk: int
    wall_gap_s: float
    previous_audio_s: float
    underrun_risk_s: float


@dataclass(frozen=True)
class TimingGapReport:
    path: str
    output_path: str | None
    playback_mode: str | None
    realtime_factor: float | None
    max_chunk_gap_over_audio_s: float | None
    timing_gap_count: int
    timing_gaps: list[TimingGap]


def _to_mono(audio: np.ndarray) -> np.ndarray:
    array = np.asarray(audio, dtype=np.float32)
    if array.ndim == 2:
        array = array.mean(axis=1)
    return array.reshape(-1)


def analyze_wav_gaps(
    path: Path,
    *,
    frame_ms: float = 20.0,
    silence_threshold_db: float = -55.0,
    min_gap_ms: float = 120.0,
) -> WavGapReport:
    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    samples = _to_mono(audio)
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    frame_size = max(1, int(sample_rate * frame_ms / 1000.0))
    min_gap_frames = max(1, int(min_gap_ms / frame_ms))
    gap_frames: list[AudioGap] = []
    if samples.size and peak > 0.0:
        frame_count = int(np.ceil(samples.size / frame_size))
        silent_runs: list[tuple[int, int, float]] = []
        run_start: int | None = None
        run_peak_db = -120.0
        for frame_index in range(frame_count):
            frame = samples[frame_index * frame_size : (frame_index + 1) * frame_size]
            frame_peak = float(np.max(np.abs(frame))) if frame.size else 0.0
            frame_db = 20.0 * np.log10(max(frame_peak / peak, 1e-6))
            if frame_db <= silence_threshold_db:
                if run_start is None:
                    run_start = frame_index
                    run_peak_db = frame_db
                else:
                    run_peak_db = max(run_peak_db, frame_db)
            elif run_start is not None:
                if frame_index - run_start >= min_gap_frames and run_start > 0:
                    silent_runs.append((run_start, frame_index, run_peak_db))
                run_start = None
        gap_frames = [
            AudioGap(
                start_s=round(start * frame_size / sample_rate, 3),
                duration_s=round((end - start) * frame_size / sample_rate, 3),
                peak_db=round(run_peak_db, 1),
            )
            for start, end, run_peak_db in silent_runs
        ]
    return WavGapReport(
        path=str(path),
        sample_rate=sample_rate,
        duration_s=round(samples.size / sample_rate, 3) if sample_rate else 0.0,
        peak=round(peak, 6),
        silence_threshold_db=silence_threshold_db,
        max_gap_s=max((gap.duration_s for gap in gap_frames), default=0.0),
        gap_count=len(gap_frames),
        gaps=gap_frames,
    )


def _timing_report_from_payload(
    path_label: str,
    payload: dict,
    *,
    tolerance_s: float,
) -> TimingGapReport:
    chunks = payload.get("chunks", [])
    timing_gaps: list[TimingGap] = []
    for previous, next_chunk in zip(chunks, chunks[1:], strict=False):
        wall_gap_s = round(float(next_chunk["elapsed_s"]) - float(previous["elapsed_s"]), 3)
        previous_audio_s = float(previous["audio_duration_s"])
        underrun_risk_s = round(wall_gap_s - previous_audio_s, 3)
        if underrun_risk_s > tolerance_s:
            timing_gaps.append(
                TimingGap(
                    previous_chunk=int(previous["index"]),
                    next_chunk=int(next_chunk["index"]),
                    wall_gap_s=wall_gap_s,
                    previous_audio_s=previous_audio_s,
                    underrun_risk_s=underrun_risk_s,
                )
            )
    return TimingGapReport(
        path=path_label,
        output_path=payload.get("output_path"),
        playback_mode=payload.get("playback_mode"),
        realtime_factor=payload.get("realtime_factor"),
        max_chunk_gap_over_audio_s=payload.get("max_chunk_gap_over_audio_s"),
        timing_gap_count=len(timing_gaps),
        timing_gaps=timing_gaps,
    )


def analyze_timing_gaps(path: Path, *, tolerance_s: float = 0.05) -> TimingGapReport:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return _timing_report_from_payload(str(path), payload, tolerance_s=tolerance_s)


def analyze_history_timing_gaps(path: Path, *, tolerance_s: float = 0.05) -> list[TimingGapReport]:
    reports: list[TimingGapReport] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if "chunks" not in payload:
            continue
        reports.append(
            _timing_report_from_payload(
                f"{path}#{line_number}",
                payload,
                tolerance_s=tolerance_s,
            )
        )
    return reports


def scan_paths(
    wav_paths: list[Path],
    timing_paths: list[Path],
    history_paths: list[Path] | None = None,
) -> dict:
    wav_reports = [asdict(analyze_wav_gaps(path)) for path in wav_paths if path.exists()]
    timing_reports = [asdict(analyze_timing_gaps(path)) for path in timing_paths if path.exists()]
    for history_path in history_paths or []:
        if history_path.exists():
            timing_reports.extend(asdict(report) for report in analyze_history_timing_gaps(history_path))
    live_timing_gap_failures = [
        item
        for item in timing_reports
        if item["timing_gap_count"] > 0 and item.get("playback_mode") in {None, "live"}
    ]
    return {
        "wav_reports": wav_reports,
        "timing_reports": timing_reports,
        "wav_gap_failures": [item for item in wav_reports if item["gap_count"] > 0],
        "live_timing_gap_failures": live_timing_gap_failures,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Scan generated audio for silence and timing gaps.")
    parser.add_argument("--wav", type=Path, action="append", default=[])
    parser.add_argument("--timing-report", type=Path, action="append", default=[])
    parser.add_argument("--history-jsonl", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "reports" / "audio_gap_scan.json")
    args = parser.parse_args(argv)

    report = scan_paths(args.wav, args.timing_report, args.history_jsonl)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 1 if report["wav_gap_failures"] or report["live_timing_gap_failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
