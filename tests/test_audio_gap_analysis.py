from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from aispeechapp.audio_gap_analysis import (
    analyze_history_timing_gaps,
    analyze_timing_gaps,
    analyze_wav_gaps,
    scan_paths,
)


def test_analyze_wav_gaps_passes_continuous_tone(tmp_path: Path):
    path = tmp_path / "tone.wav"
    sample_rate = 16000
    t = np.arange(sample_rate, dtype=np.float32) / sample_rate
    audio = np.sin(2.0 * np.pi * 440.0 * t).astype(np.float32) * 0.2
    sf.write(path, audio, sample_rate)

    report = analyze_wav_gaps(path)

    assert report.gap_count == 0
    assert report.max_gap_s == 0.0


def test_analyze_wav_gaps_detects_inserted_silence(tmp_path: Path):
    path = tmp_path / "gap.wav"
    sample_rate = 16000
    tone = np.ones(int(sample_rate * 0.3), dtype=np.float32) * 0.2
    silence = np.zeros(int(sample_rate * 0.2), dtype=np.float32)
    sf.write(path, np.concatenate([tone, silence, tone]), sample_rate)

    report = analyze_wav_gaps(path)

    assert report.gap_count == 1
    assert report.gaps[0].start_s == 0.3
    assert report.gaps[0].duration_s == 0.2


def test_analyze_wav_gaps_ignores_leading_and_trailing_silence(tmp_path: Path):
    path = tmp_path / "edge_silence.wav"
    sample_rate = 16000
    silence = np.zeros(int(sample_rate * 0.2), dtype=np.float32)
    tone = np.ones(int(sample_rate * 0.3), dtype=np.float32) * 0.2
    sf.write(path, np.concatenate([silence, tone, silence]), sample_rate)

    report = analyze_wav_gaps(path)

    assert report.gap_count == 0


def test_analyze_timing_gaps_detects_streaming_underrun_risk(tmp_path: Path):
    path = tmp_path / "streaming.json"
    path.write_text(
        json.dumps(
            {
                "output_path": "out.wav",
                "realtime_factor": 2.0,
                "max_chunk_gap_over_audio_s": 0.39,
                "chunks": [
                    {"index": 0, "elapsed_s": 1.0, "audio_duration_s": 0.16},
                    {"index": 1, "elapsed_s": 1.55, "audio_duration_s": 0.16},
                ],
            }
        ),
        encoding="utf-8",
    )

    report = analyze_timing_gaps(path)

    assert report.timing_gap_count == 1
    assert report.timing_gaps[0].underrun_risk_s == 0.39
    assert report.playback_mode is None


def test_scan_paths_does_not_fail_timing_gaps_for_after_generation_playback(tmp_path: Path):
    path = tmp_path / "streaming.json"
    path.write_text(
        json.dumps(
            {
                "output_path": "out.wav",
                "playback_mode": "after_generation",
                "chunks": [
                    {"index": 0, "elapsed_s": 1.0, "audio_duration_s": 0.16},
                    {"index": 1, "elapsed_s": 1.55, "audio_duration_s": 0.16},
                ],
            }
        ),
        encoding="utf-8",
    )

    report = scan_paths([], [path])

    assert report["timing_reports"][0]["timing_gap_count"] == 1
    assert report["live_timing_gap_failures"] == []


def test_analyze_history_timing_gaps_skips_demo_lines_without_chunks(tmp_path: Path):
    path = tmp_path / "history.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"demo": True}),
                json.dumps(
                    {
                        "output_path": "out.wav",
                        "chunks": [
                            {"index": 0, "elapsed_s": 1.0, "audio_duration_s": 0.16},
                            {"index": 1, "elapsed_s": 1.18, "audio_duration_s": 0.16},
                        ],
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    reports = analyze_history_timing_gaps(path)

    assert len(reports) == 1
    assert reports[0].path.endswith("#2")
    assert reports[0].timing_gap_count == 0
