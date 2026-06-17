from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from aispeechapp.voxcpm2_streaming import (
    append_streaming_history,
    synthesize_voxcpm2_streaming,
    write_streaming_report,
)


class FakeStreamingModel:
    def __init__(self, chunks: list[np.ndarray]) -> None:
        self.chunks = chunks
        self.calls: list[dict[str, object]] = []

    def generate_streaming(self, *args: object, **kwargs: object):
        self.calls.append(dict(kwargs))
        yield from self.chunks


class FakeClock:
    def __init__(self, values: list[float]) -> None:
        self.values = values
        self.index = 0

    def __call__(self) -> float:
        value = self.values[self.index]
        self.index += 1
        return value


class FakeAudioSink:
    def __init__(self) -> None:
        self.writes: list[np.ndarray] = []
        self.closed = False

    def write(self, audio: np.ndarray) -> None:
        self.writes.append(audio.copy())

    def close(self) -> None:
        self.closed = True


def test_streaming_synthesis_writes_chunks_and_latency(tmp_path: Path):
    model = FakeStreamingModel(
        [
            np.ones(480, dtype=np.float32) * 0.1,
            np.ones(960, dtype=np.float32) * 0.2,
        ]
    )
    reference = tmp_path / "reference.wav"
    sf.write(reference, np.zeros(1600, dtype=np.float32), 16000)
    output = tmp_path / "out.wav"

    result = synthesize_voxcpm2_streaming(
        text="Hello from VoxCPM2.",
        output_path=output,
        reference_wav_path=reference,
        model_factory=lambda: model,
        clock=FakeClock([10.0, 10.25, 10.75, 10.8]),
    )

    assert output.exists()
    info = sf.info(str(output))
    assert info.samplerate == 48000
    assert info.frames == 1440
    assert result.chunk_count == 2
    assert result.first_chunk_latency_s == 0.25
    assert result.audio_duration_s == 0.03
    assert result.max_chunk_gap_s == 0.5
    assert result.max_chunk_gap_over_audio_s == 0.48
    assert result.played_to_device is False
    assert result.audio_device is None
    assert model.calls[0]["reference_wav_path"] == str(reference)
    assert model.calls[0]["cfg_value"] == 2.0
    assert model.calls[0]["inference_timesteps"] == 10
    assert model.calls[0]["min_len"] == 2
    assert model.calls[0]["max_len"] == 4096
    assert model.calls[0]["normalize"] is True
    assert model.calls[0]["denoise"] is False
    assert model.calls[0]["retry_badcase"] is False


def test_streaming_synthesis_passes_generation_knobs(tmp_path: Path):
    model = FakeStreamingModel([np.ones(240, dtype=np.float32)])

    result = synthesize_voxcpm2_streaming(
        text="Hello from VoxCPM2.",
        output_path=tmp_path / "out.wav",
        cfg_value=2.8,
        inference_timesteps=14,
        min_len=4,
        max_len=2048,
        normalize=False,
        denoise=True,
        retry_badcase=True,
        retry_badcase_max_times=5,
        retry_badcase_ratio_threshold=4.5,
        model_factory=lambda: model,
        clock=FakeClock([0.0, 0.1, 0.2]),
    )

    assert result.cfg_value == 2.8
    assert result.inference_timesteps == 14
    assert result.min_len == 4
    assert result.max_len == 2048
    assert result.normalize is False
    assert result.denoise is True
    assert result.retry_badcase is True
    assert result.retry_badcase_max_times == 5
    assert result.retry_badcase_ratio_threshold == 4.5
    assert model.calls[0]["cfg_value"] == 2.8
    assert model.calls[0]["inference_timesteps"] == 14
    assert model.calls[0]["min_len"] == 4
    assert model.calls[0]["max_len"] == 2048
    assert model.calls[0]["normalize"] is False
    assert model.calls[0]["denoise"] is True
    assert model.calls[0]["retry_badcase"] is True
    assert model.calls[0]["retry_badcase_max_times"] == 5
    assert model.calls[0]["retry_badcase_ratio_threshold"] == 4.5


def test_streaming_synthesis_can_play_chunks_to_audio_sink(tmp_path: Path):
    model = FakeStreamingModel([np.ones(240, dtype=np.float32)])
    sink = FakeAudioSink()

    result = synthesize_voxcpm2_streaming(
        text="Hello from VoxCPM2.",
        output_path=tmp_path / "out.wav",
        play_audio=True,
        audio_device="Test Device",
        model_factory=lambda: model,
        audio_sink_factory=lambda sample_rate, device, prebuffer_s, latency: sink,
        clock=FakeClock([0.0, 0.1, 0.2]),
    )

    assert result.played_to_device is True
    assert result.audio_device == "Test Device"
    assert result.playback_prebuffer_s == 0.45
    assert result.audio_latency == "high"
    assert len(sink.writes) == 1
    assert sink.writes[0].shape == (240,)
    assert sink.closed is True


def test_streaming_report_and_history_are_written(tmp_path: Path):
    result = synthesize_voxcpm2_streaming(
        text="Hello from VoxCPM2.",
        output_path=tmp_path / "out.wav",
        model_factory=lambda: FakeStreamingModel([np.ones(240, dtype=np.float32)]),
        clock=FakeClock([0.0, 0.1, 0.2]),
    )
    report = tmp_path / "report.json"
    history = tmp_path / "history.jsonl"

    write_streaming_report(result, report)
    append_streaming_history(result, history)
    append_streaming_history(result, history)

    assert '"chunk_count": 1' in report.read_text(encoding="utf-8")
    assert len(history.read_text(encoding="utf-8").splitlines()) == 2


def test_streaming_synthesis_rejects_missing_reference(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        synthesize_voxcpm2_streaming(
            text="Hello.",
            output_path=tmp_path / "out.wav",
            reference_wav_path=tmp_path / "missing.wav",
            model_factory=lambda: FakeStreamingModel([]),
        )
