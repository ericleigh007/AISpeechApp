from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from aispeechapp.voxcpm2_streaming import (
    SoundDeviceSink,
    StreamingPeakNormalizer,
    _prepared_default_model_factory,
    _maybe_compile_model,
    _maybe_quantize_model,
    _set_torch_seed,
    append_streaming_history,
    clear_default_model_caches,
    _default_model_factory,
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


def test_default_voxcpm2_factory_disables_internal_compile(monkeypatch):
    calls: list[dict[str, object]] = []

    class FakeVoxCPM:
        @classmethod
        def from_pretrained(cls, *args: object, **kwargs: object):
            calls.append({"args": args, "kwargs": kwargs})
            return "model"

    monkeypatch.setitem(sys.modules, "voxcpm", types.SimpleNamespace(VoxCPM=FakeVoxCPM))
    monkeypatch.setattr("aispeechapp.voxcpm2_streaming.configure_model_caches", lambda: None)
    clear_default_model_caches()
    try:
        assert _default_model_factory() == "model"
    finally:
        clear_default_model_caches()

    assert calls[0]["kwargs"]["optimize"] is False


def test_prepared_default_model_cache_reuses_same_candidate(monkeypatch):
    calls: list[str] = []

    def fake_loader():
        calls.append("load")
        return FakeStreamingModel([np.ones(240, dtype=np.float32)])

    def fake_quantize(model, *, method: str, targets: str):
        calls.append(f"quantize:{method}:{targets}")
        return model, f"prepared:{method}:{targets}"

    monkeypatch.setattr("aispeechapp.voxcpm2_streaming._load_default_voxcpm2_model", fake_loader)
    monkeypatch.setattr("aispeechapp.voxcpm2_streaming._maybe_quantize_model", fake_quantize)
    clear_default_model_caches()
    try:
        first_model, first_status = _prepared_default_model_factory(
            "voxcpm2_bnb_int8",
            "bnb_int8_linear",
            "lm_only",
        )
        second_model, second_status = _prepared_default_model_factory(
            "voxcpm2_bnb_int8",
            "bnb_int8_linear",
            "lm_only",
        )
    finally:
        clear_default_model_caches()

    assert first_model is second_model
    assert first_status == second_status == "prepared:bnb_int8_linear:lm_only"
    assert calls == ["load", "quantize:bnb_int8_linear:lm_only"]


def test_prepared_default_model_cache_reloads_when_candidate_changes(monkeypatch):
    calls: list[str] = []

    def fake_loader():
        calls.append("load")
        return FakeStreamingModel([np.ones(240, dtype=np.float32)])

    def fake_quantize(model, *, method: str, targets: str):
        calls.append(f"quantize:{method}:{targets}")
        return model, f"prepared:{method}:{targets}"

    monkeypatch.setattr("aispeechapp.voxcpm2_streaming._load_default_voxcpm2_model", fake_loader)
    monkeypatch.setattr("aispeechapp.voxcpm2_streaming._maybe_quantize_model", fake_quantize)
    clear_default_model_caches()
    try:
        first_model, _ = _prepared_default_model_factory(
            "voxcpm2",
            "disabled",
            "lm_only",
        )
        second_model, _ = _prepared_default_model_factory(
            "voxcpm2_bnb_int8",
            "bnb_int8_linear",
            "lm_only",
        )
    finally:
        clear_default_model_caches()

    assert first_model is not second_model
    assert calls == [
        "load",
        "quantize:disabled:lm_only",
        "load",
        "quantize:bnb_int8_linear:lm_only",
    ]


def test_default_synthesis_reuses_prepared_model_for_same_candidate(monkeypatch, tmp_path: Path):
    calls: list[str] = []

    def fake_loader():
        calls.append("load")
        return FakeStreamingModel([np.ones(240, dtype=np.float32)])

    def fake_quantize(model, *, method: str, targets: str):
        calls.append(f"quantize:{method}:{targets}")
        return model, f"prepared:{method}:{targets}"

    monkeypatch.setattr("aispeechapp.voxcpm2_streaming._load_default_voxcpm2_model", fake_loader)
    monkeypatch.setattr("aispeechapp.voxcpm2_streaming._maybe_quantize_model", fake_quantize)
    clear_default_model_caches()
    try:
        first = synthesize_voxcpm2_streaming(
            text="Hello from VoxCPM2.",
            output_path=tmp_path / "first.wav",
            model_cache_key="voxcpm2_bnb_int8",
            quantization_method="bnb_int8_linear",
            clock=FakeClock([0.0, 0.1, 0.2]),
        )
        second = synthesize_voxcpm2_streaming(
            text="Hello again from VoxCPM2.",
            output_path=tmp_path / "second.wav",
            model_cache_key="voxcpm2_bnb_int8",
            quantization_method="bnb_int8_linear",
            clock=FakeClock([0.0, 0.1, 0.2]),
        )
    finally:
        clear_default_model_caches()

    assert first.quantization_status == second.quantization_status == "prepared:bnb_int8_linear:lm_only"
    assert calls == ["load", "quantize:bnb_int8_linear:lm_only"]


def test_default_synthesis_loads_new_prepared_model_when_candidate_changes(monkeypatch, tmp_path: Path):
    calls: list[str] = []

    def fake_loader():
        calls.append("load")
        return FakeStreamingModel([np.ones(240, dtype=np.float32)])

    def fake_quantize(model, *, method: str, targets: str):
        calls.append(f"quantize:{method}:{targets}")
        return model, f"prepared:{method}:{targets}"

    monkeypatch.setattr("aispeechapp.voxcpm2_streaming._load_default_voxcpm2_model", fake_loader)
    monkeypatch.setattr("aispeechapp.voxcpm2_streaming._maybe_quantize_model", fake_quantize)
    clear_default_model_caches()
    try:
        synthesize_voxcpm2_streaming(
            text="Hello from VoxCPM2.",
            output_path=tmp_path / "first.wav",
            model_cache_key="voxcpm2",
            clock=FakeClock([0.0, 0.1, 0.2]),
        )
        synthesize_voxcpm2_streaming(
            text="Hello from quantized VoxCPM2.",
            output_path=tmp_path / "second.wav",
            model_cache_key="voxcpm2_bnb_int8",
            quantization_method="bnb_int8_linear",
            clock=FakeClock([0.0, 0.1, 0.2]),
        )
    finally:
        clear_default_model_caches()

    assert calls == [
        "load",
        "quantize:disabled:lm_only",
        "load",
        "quantize:bnb_int8_linear:lm_only",
    ]


def test_compile_default_synthesis_uses_uncached_fresh_model(monkeypatch, tmp_path: Path):
    calls: list[str] = []

    def fake_loader():
        calls.append("load")
        return FakeStreamingModel([np.ones(240, dtype=np.float32)])

    def fake_quantize(model, *, method: str, targets: str):
        calls.append(f"quantize:{method}:{targets}")
        return model, f"prepared:{method}:{targets}"

    def fake_compile(model, *, enabled: bool, mode: str):
        calls.append(f"compile:{enabled}:{mode}")
        return model, "compiled:fresh"

    monkeypatch.setattr("aispeechapp.voxcpm2_streaming._load_default_voxcpm2_model", fake_loader)
    monkeypatch.setattr("aispeechapp.voxcpm2_streaming._maybe_quantize_model", fake_quantize)
    monkeypatch.setattr("aispeechapp.voxcpm2_streaming._maybe_compile_model", fake_compile)
    clear_default_model_caches()
    try:
        first = synthesize_voxcpm2_streaming(
            text="Hello from compiled VoxCPM2.",
            output_path=tmp_path / "first.wav",
            torch_compile=True,
            clock=FakeClock([0.0, 0.1, 0.2]),
        )
        second = synthesize_voxcpm2_streaming(
            text="Hello again from compiled VoxCPM2.",
            output_path=tmp_path / "second.wav",
            torch_compile=True,
            clock=FakeClock([0.0, 0.1, 0.2]),
        )
    finally:
        clear_default_model_caches()

    assert first.torch_compile_status == second.torch_compile_status == "compiled:fresh"
    assert calls == [
        "load",
        "quantize:disabled:lm_only",
        "compile:True:reduce-overhead",
        "load",
        "quantize:disabled:lm_only",
        "compile:True:reduce-overhead",
    ]


def test_sound_device_sink_writes_audio_directly_after_prebuffer(monkeypatch):
    events: list[tuple[str, object]] = []

    class FakeOutputStream:
        def __init__(self, samplerate, channels, dtype, device, latency):
            events.append(("init", (samplerate, channels, dtype, device, latency)))

        def start(self):
            events.append(("start", None))

        def write(self, audio):
            events.append(("write", tuple(audio.shape)))

        def stop(self):
            events.append(("stop", None))

        def close(self):
            events.append(("close", None))

    class FakeSoundDevice:
        OutputStream = FakeOutputStream

    monkeypatch.setitem(sys.modules, "sounddevice", FakeSoundDevice())

    sink = SoundDeviceSink(48000, device=None, prebuffer_s=0.02, latency="high")
    sink.write(np.ones(480, dtype=np.float32))
    assert not any(event[0] == "write" for event in events)
    sink.write(np.ones(480, dtype=np.float32) * 0.5)
    sink.write(np.ones(480, dtype=np.float32) * 0.25)
    sink.close()

    assert [event[0] for event in events] == ["init", "start", "write", "write", "write", "stop", "close"]


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
    assert result.seed is None
    assert result.quantization_method == "disabled"
    assert result.quantization_status == "disabled"
    assert result.audio_normalization is False
    assert result.audio_target_peak == 0.85
    assert result.playback_mode == "after_generation"
    assert result.torch_compile is False
    assert result.torch_compile_status == "disabled"


def test_streaming_peak_normalizer_lifts_quiet_audio():
    normalizer = StreamingPeakNormalizer(target_peak=0.8, smoothing=1.0)
    audio = np.ones(1000, dtype=np.float32) * 0.1

    normalized = normalizer.process(audio)

    assert np.max(np.abs(normalized)) == pytest.approx(0.4)


def test_streaming_peak_normalizer_limits_loud_audio():
    normalizer = StreamingPeakNormalizer(target_peak=0.8, smoothing=1.0)
    audio = np.ones(1000, dtype=np.float32) * 2.0

    normalized = normalizer.process(audio)

    assert np.max(np.abs(normalized)) == pytest.approx(0.8)


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
        seed=123,
        quantization_method="dynamic_int8_cpu",
        quantization_targets="base_lm_only",
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
    assert result.seed == 123
    assert result.quantization_method == "dynamic_int8_cpu"
    assert result.quantization_targets == "base_lm_only"
    assert result.quantization_status.startswith("not_applicable")
    assert result.torch_compile is False
    assert result.torch_compile_mode == "reduce-overhead"
    assert model.calls[0]["cfg_value"] == 2.8
    assert model.calls[0]["inference_timesteps"] == 14
    assert model.calls[0]["min_len"] == 4
    assert model.calls[0]["max_len"] == 2048
    assert model.calls[0]["normalize"] is False
    assert model.calls[0]["denoise"] is True
    assert model.calls[0]["retry_badcase"] is True
    assert model.calls[0]["retry_badcase_max_times"] == 5
    assert model.calls[0]["retry_badcase_ratio_threshold"] == 4.5


def test_set_torch_seed_sets_cpu_and_cuda_seed(monkeypatch):
    calls: list[tuple[str, int]] = []

    fake_torch = SimpleNamespace(
        manual_seed=lambda seed: calls.append(("manual_seed", seed)),
        cuda=SimpleNamespace(
            is_available=lambda: True,
            manual_seed_all=lambda seed: calls.append(("manual_seed_all", seed)),
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    assert _set_torch_seed(77) == "set:torch+cuda"
    assert calls == [("manual_seed", 77), ("manual_seed_all", 77)]


def test_set_torch_seed_can_be_disabled():
    assert _set_torch_seed(None) == "disabled"


def test_quantization_skips_cuda_voxcpm_model():
    model = SimpleNamespace(tts_model=SimpleNamespace(device="cuda:0"))

    quantized, status = _maybe_quantize_model(
        model,
        method="dynamic_int8_cpu",
        targets="lm_only",
    )

    assert quantized is model
    assert status == "skipped:dynamic_int8_cpu requires CPU model, found cuda:0"


def test_quantization_dynamic_int8_quantizes_cpu_lm_targets():
    import torch
    from torch.ao.nn.quantized.dynamic.modules.linear import Linear as DynamicQuantizedLinear

    model = SimpleNamespace(
        tts_model=SimpleNamespace(
            device="cpu",
            base_lm=torch.nn.Sequential(torch.nn.Linear(4, 4)),
            residual_lm=torch.nn.Sequential(torch.nn.Linear(4, 4)),
        )
    )

    quantized, status = _maybe_quantize_model(
        model,
        method="dynamic_int8_cpu",
        targets="lm_only",
    )

    assert quantized is model
    assert status == "dynamic_int8_cpu:tts_model.base_lm,tts_model.residual_lm"
    assert isinstance(model.tts_model.base_lm[0], DynamicQuantizedLinear)
    assert isinstance(model.tts_model.residual_lm[0], DynamicQuantizedLinear)


def test_quantization_torchao_int8_quantizes_lm_targets():
    import torch

    model = SimpleNamespace(
        tts_model=SimpleNamespace(
            device="cuda" if torch.cuda.is_available() else "cpu",
            base_lm=torch.nn.Sequential(torch.nn.Linear(4, 4)),
            residual_lm=torch.nn.Sequential(torch.nn.Linear(4, 4)),
        )
    )
    if torch.cuda.is_available():
        model.tts_model.base_lm = model.tts_model.base_lm.cuda().bfloat16()
        model.tts_model.residual_lm = model.tts_model.residual_lm.cuda().bfloat16()

    quantized, status = _maybe_quantize_model(
        model,
        method="torchao_int8_weight_only",
        targets="lm_only",
    )

    assert quantized is model
    assert status == "torchao_int8_weight_only:tts_model.base_lm,tts_model.residual_lm"
    assert "torchao" in type(model.tts_model.base_lm[0].weight).__module__
    assert "torchao" in type(model.tts_model.residual_lm[0].weight).__module__


def test_quantization_bnb_int8_replaces_lm_linear_layers():
    import bitsandbytes as bnb
    import torch

    model = SimpleNamespace(
        tts_model=SimpleNamespace(
            device="cuda" if torch.cuda.is_available() else "cpu",
            base_lm=torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU(), torch.nn.Linear(4, 4)),
            residual_lm=torch.nn.Sequential(torch.nn.Linear(4, 4)),
        )
    )
    if torch.cuda.is_available():
        model.tts_model.base_lm = model.tts_model.base_lm.cuda().bfloat16()
        model.tts_model.residual_lm = model.tts_model.residual_lm.cuda().bfloat16()

    quantized, status = _maybe_quantize_model(
        model,
        method="bnb_int8_linear",
        targets="lm_only",
    )

    assert quantized is model
    assert status == "bnb_int8_linear:tts_model.base_lm:2,tts_model.residual_lm:1"
    assert isinstance(model.tts_model.base_lm[0], bnb.nn.Linear8bitLt)
    assert isinstance(model.tts_model.base_lm[2], bnb.nn.Linear8bitLt)
    assert isinstance(model.tts_model.residual_lm[0], bnb.nn.Linear8bitLt)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    output = model.tts_model.base_lm(torch.randn(1, 4, device=device, dtype=dtype))
    assert output.shape == (1, 4)


def test_streaming_synthesis_can_opt_into_torch_compile(monkeypatch, tmp_path: Path):
    compiled: list[tuple[object, str]] = []

    class FakeModule:
        pass

    class ModelWithModule(FakeStreamingModel):
        def __init__(self) -> None:
            super().__init__([np.ones(240, dtype=np.float32)])
            self.decoder = FakeModule()

    def fake_compile(module, *, mode):
        compiled.append((module, mode))
        return ("compiled", module, mode)

    fake_torch = SimpleNamespace(
        compile=fake_compile,
        nn=SimpleNamespace(Module=FakeModule),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    model = ModelWithModule()
    original_decoder = model.decoder

    result = synthesize_voxcpm2_streaming(
        text="Hello from VoxCPM2.",
        output_path=tmp_path / "out.wav",
        torch_compile=True,
        torch_compile_mode="default",
        model_factory=lambda: model,
        clock=FakeClock([0.0, 0.1, 0.2]),
    )

    assert compiled == [(original_decoder, "default")]
    assert model.decoder == ("compiled", original_decoder, "default")
    assert result.torch_compile is True
    assert result.torch_compile_mode == "default"
    assert result.torch_compile_status == "compiled:modules:decoder:default"


def test_torch_compile_does_not_wrap_voxcpm_tts_model(monkeypatch):
    compiled: list[tuple[object, str]] = []

    class FakeModule:
        pass

    class FakeVoxCPMWrapper:
        def __init__(self) -> None:
            self.tts_model = FakeModule()

    def fake_compile(module, *, mode):
        compiled.append((module, mode))
        return ("compiled", module, mode)

    fake_torch = SimpleNamespace(
        compile=fake_compile,
        nn=SimpleNamespace(Module=FakeModule),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    model = FakeVoxCPMWrapper()
    original_tts_model = model.tts_model

    compiled_model, status = _maybe_compile_model(
        model,
        enabled=True,
        mode="max-autotune",
    )

    assert compiled_model is model
    assert model.tts_model is original_tts_model
    assert compiled == []
    assert status == "not_applicable: no torch.nn.Module target found"


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
    assert result.playback_mode == "after_generation"
    assert result.audio_device == "Test Device"
    assert result.playback_prebuffer_s == 0.45
    assert result.audio_latency == "high"
    assert len(sink.writes) == 1
    assert sink.writes[0].shape == (240,)
    assert np.max(np.abs(sink.writes[0])) > 0.1
    assert sink.closed is True


def test_after_generation_playback_defers_audio_device_until_model_finishes(tmp_path: Path):
    sink = FakeAudioSink()
    writes_during_generation: list[int] = []

    class MonitoringModel(FakeStreamingModel):
        def generate_streaming(self, *args: object, **kwargs: object):
            for chunk in self.chunks:
                writes_during_generation.append(len(sink.writes))
                yield chunk

    model = MonitoringModel([np.ones(240, dtype=np.float32), np.ones(240, dtype=np.float32)])

    result = synthesize_voxcpm2_streaming(
        text="Hello from VoxCPM2.",
        output_path=tmp_path / "out.wav",
        play_audio=True,
        playback_mode="after_generation",
        model_factory=lambda: model,
        audio_sink_factory=lambda sample_rate, device, prebuffer_s, latency: sink,
        clock=FakeClock([0.0, 0.1, 0.2, 0.3]),
    )

    assert writes_during_generation == [0, 0]
    assert len(sink.writes) == 2
    assert result.playback_mode == "after_generation"


def test_live_playback_writes_audio_device_during_generation(tmp_path: Path):
    sink = FakeAudioSink()
    writes_during_generation: list[int] = []
    observer_calls_before_write: list[int] = []

    class MonitoringModel(FakeStreamingModel):
        def generate_streaming(self, *args: object, **kwargs: object):
            for chunk in self.chunks:
                writes_during_generation.append(len(sink.writes))
                yield chunk

    model = MonitoringModel([np.ones(240, dtype=np.float32), np.ones(240, dtype=np.float32)])

    result = synthesize_voxcpm2_streaming(
        text="Hello from VoxCPM2.",
        output_path=tmp_path / "out.wav",
        play_audio=True,
        playback_mode="live",
        model_factory=lambda: model,
        audio_sink_factory=lambda sample_rate, device, prebuffer_s, latency: sink,
        audio_observer=lambda _audio, _sample_rate: observer_calls_before_write.append(len(sink.writes)),
        clock=FakeClock([0.0, 0.1, 0.2, 0.3]),
    )

    assert writes_during_generation == [0, 1]
    assert observer_calls_before_write == [0, 1]
    assert len(sink.writes) == 2
    assert result.playback_mode == "live"


def test_streaming_synthesis_can_disable_audio_normalization(tmp_path: Path):
    model = FakeStreamingModel([np.ones(240, dtype=np.float32) * 0.1])
    sink = FakeAudioSink()

    synthesize_voxcpm2_streaming(
        text="Hello from VoxCPM2.",
        output_path=tmp_path / "out.wav",
        play_audio=True,
        audio_normalization=False,
        model_factory=lambda: model,
        audio_sink_factory=lambda sample_rate, device, prebuffer_s, latency: sink,
        clock=FakeClock([0.0, 0.1, 0.2]),
    )

    assert np.max(np.abs(sink.writes[0])) == pytest.approx(0.1)


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
