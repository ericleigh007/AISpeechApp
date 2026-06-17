from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import soundfile as sf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "synthesize_backend.py"


def _load_backend_module():
    spec = importlib.util.spec_from_file_location("synthesize_backend_under_test", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_synthesize_backend_routes_omnivoice(monkeypatch, tmp_path: Path):
    module = _load_backend_module()
    calls = []
    monkeypatch.setattr(module, "_write_omnivoice", lambda text, output: calls.append((text, output)))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "synthesize_backend.py",
            "--candidate",
            "omnivoice",
            "--text",
            "hello",
            "--language-code",
            "en",
            "--language-hint",
            "English",
            "--output",
            str(tmp_path / "out.wav"),
        ],
    )

    assert module.main() == 0
    assert calls == [("hello", tmp_path / "out.wav")]


def test_synthesize_backend_routes_vibevoice(monkeypatch, tmp_path: Path):
    module = _load_backend_module()
    calls = []
    monkeypatch.setattr(
        module,
        "_write_vibevoice_15b",
        lambda text, output: calls.append((text, output)),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "synthesize_backend.py",
            "--candidate",
            "microsoft_vibevoice_15b",
            "--text",
            "hello",
            "--language-code",
            "en",
            "--language-hint",
            "English",
            "--output",
            str(tmp_path / "out.wav"),
        ],
    )

    assert module.main() == 0
    assert calls == [("hello", tmp_path / "out.wav")]


def test_synthesize_backend_routes_voxcpm2_options(monkeypatch, tmp_path: Path):
    module = _load_backend_module()
    calls = []
    monkeypatch.setattr(module, "_write_voxcpm2", lambda text, output, options: calls.append((text, output, options)))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "synthesize_backend.py",
            "--candidate",
            "voxcpm2",
            "--text",
            "hello",
            "--language-code",
            "en",
            "--language-hint",
            "English",
            "--output",
            str(tmp_path / "out.wav"),
            "--options-json",
            '{"cfg_value": 2.4, "inference_timesteps": 12}',
        ],
    )

    assert module.main() == 0
    assert calls == [("hello", tmp_path / "out.wav", {"cfg_value": 2.4, "inference_timesteps": 12})]


def test_write_audio_file_wav(tmp_path: Path):
    module = _load_backend_module()
    output = tmp_path / "tone.wav"
    audio = np.sin(np.linspace(0, np.pi * 2, 2400, dtype=np.float32))

    module._write_audio_file(output, audio, 24000)

    data, sample_rate = sf.read(output)
    assert output.exists()
    assert sample_rate == 24000
    assert len(data) == len(audio)


def test_write_audio_file_mp3(tmp_path: Path):
    module = _load_backend_module()
    if shutil.which("ffmpeg") is None:
        raise AssertionError("ffmpeg is required for MP3 output tests on this workstation.")

    output = tmp_path / "tone.mp3"
    audio = np.sin(np.linspace(0, np.pi * 2, 2400, dtype=np.float32))

    module._write_audio_file(output, audio, 24000)

    assert output.exists()
    assert output.stat().st_size > 0
    assert not (tmp_path / "tone.tmp.wav").exists()


def test_write_audio_file_rejects_unsupported_suffix(tmp_path: Path):
    module = _load_backend_module()
    audio = np.zeros(120, dtype=np.float32)

    try:
        module._write_audio_file(tmp_path / "tone.flac", audio, 24000)
    except RuntimeError as exc:
        assert "Unsupported output extension" in str(exc)
    else:
        raise AssertionError("Expected unsupported extension to raise RuntimeError.")


def test_omnivoice_backend_writes_mp3_with_reference_voice(monkeypatch, tmp_path: Path):
    module = _load_backend_module()
    if shutil.which("ffmpeg") is None:
        raise AssertionError("ffmpeg is required for MP3 output tests on this workstation.")

    voice_refs = tmp_path / "voice_refs"
    voice_refs.mkdir()
    (voice_refs / "first_impression.wav").write_bytes(b"fake wav placeholder")
    (voice_refs / "first_impression.txt").write_text("reference transcript", encoding="utf-8")

    class FakeOmniVoice:
        @classmethod
        def from_pretrained(cls, model_id, device_map, dtype):
            assert model_id == "k2-fsa/OmniVoice"
            assert device_map == "cpu"
            assert dtype == "float32"
            return cls()

        def generate(self, text, ref_audio, ref_text):
            assert text == "hello from omnivoice"
            assert ref_audio.endswith("first_impression.wav")
            assert ref_text == "reference transcript"
            return [np.sin(np.linspace(0, np.pi * 2, 2400, dtype=np.float32))]

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        float16="float16",
        float32="float32",
    )
    fake_omnivoice = SimpleNamespace(OmniVoice=FakeOmniVoice)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "omnivoice", fake_omnivoice)
    monkeypatch.chdir(tmp_path)

    output = tmp_path / "omnivoice.mp3"
    module._write_omnivoice("hello from omnivoice", output)

    assert output.exists()
    assert output.stat().st_size > 0


def test_vibevoice_backend_writes_mp3_from_pipeline(monkeypatch, tmp_path: Path):
    module = _load_backend_module()
    if shutil.which("ffmpeg") is None:
        raise AssertionError("ffmpeg is required for MP3 output tests on this workstation.")

    def fake_pipeline(task, model, device, torch_dtype):
        assert task == "text-to-speech"
        assert model == "microsoft/VibeVoice-1.5B"
        assert device == -1
        assert torch_dtype == "float32"

        def run(text):
            assert text == "hello from vibevoice"
            return {
                "audio": np.sin(np.linspace(0, np.pi * 2, 2400, dtype=np.float32)),
                "sampling_rate": 24000,
            }

        return run

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        bfloat16="bfloat16",
        float32="float32",
    )
    fake_transformers = SimpleNamespace(pipeline=fake_pipeline)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    output = tmp_path / "vibevoice.mp3"
    module._write_vibevoice_15b("hello from vibevoice", output)

    assert output.exists()
    assert output.stat().st_size > 0
