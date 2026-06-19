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
    reference = tmp_path / "voice.wav"
    monkeypatch.setattr(
        module,
        "_write_omnivoice",
        lambda text, output, reference_wav_path: calls.append((text, output, reference_wav_path)),
    )
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
            "--reference-wav-path",
            str(reference),
        ],
    )

    assert module.main() == 0
    assert calls == [("hello", tmp_path / "out.wav", reference)]


def test_synthesize_backend_routes_vibevoice(monkeypatch, tmp_path: Path):
    module = _load_backend_module()
    calls = []
    monkeypatch.setattr(
        module,
        "_write_vibevoice_15b",
        lambda text, output, options, reference_wav_path: calls.append(
            (text, output, options, reference_wav_path)
        ),
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
    assert calls == [("hello", tmp_path / "out.wav", {}, None)]


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


def test_qwen_backend_uses_style_instruction_option(monkeypatch, tmp_path: Path):
    module = _load_backend_module()

    class FakeModel:
        @classmethod
        def from_pretrained(cls, model_id, device_map, dtype, attn_implementation):
            assert model_id == "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
            return cls()

        def generate_custom_voice(
            self,
            *,
            text,
            language,
            speaker,
            instruct,
            max_new_tokens,
        ):
            assert text == "hello from qwen"
            assert language == "English"
            assert speaker == "Ryan"
            assert instruct == "warm calm delivery"
            assert max_new_tokens == 256
            return [np.zeros(2400, dtype=np.float32)], 24000

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        bfloat16="bfloat16",
        float32="float32",
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "qwen_tts", SimpleNamespace(Qwen3TTSModel=FakeModel))

    output = tmp_path / "qwen.wav"
    module._write_qwen3(
        "qwen3_tts_17b_customvoice",
        "hello from qwen",
        "en",
        "English",
        output,
        {"style_instruction": "warm calm delivery", "max_new_tokens": 256},
    )

    assert output.exists()


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
    module._write_omnivoice("hello from omnivoice", output, None)

    assert output.exists()
    assert output.stat().st_size > 0


def test_omnivoice_backend_uses_selected_reference_voice(monkeypatch, tmp_path: Path):
    module = _load_backend_module()

    reference = tmp_path / "custom_voice.wav"
    reference.write_bytes(b"fake wav placeholder")
    reference.with_suffix(".txt").write_text("custom transcript", encoding="utf-8")

    class FakeOmniVoice:
        @classmethod
        def from_pretrained(cls, model_id, device_map, dtype):
            return cls()

        def generate(self, text, ref_audio, ref_text):
            assert ref_audio == str(reference)
            assert ref_text == "custom transcript"
            return [np.zeros(2400, dtype=np.float32)]

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        float16="float16",
        float32="float32",
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "omnivoice", SimpleNamespace(OmniVoice=FakeOmniVoice))

    output = tmp_path / "omnivoice.wav"
    module._write_omnivoice("hello from omnivoice", output, reference)

    assert output.exists()


def test_vibevoice_backend_writes_wav_from_native_inference(monkeypatch, tmp_path: Path):
    module = _load_backend_module()

    voice_refs = tmp_path / "voice_refs"
    voice_refs.mkdir()
    sf.write(voice_refs / "first_impression.wav", np.zeros(2400, dtype=np.float32), 24000)

    class FakeTensor:
        def __init__(self):
            self.device = None

        def to(self, device):
            self.device = device
            return self

    class FakeProcessor:
        tokenizer = object()

        @classmethod
        def from_pretrained(cls, model_id):
            assert model_id == "microsoft/VibeVoice-1.5B"
            return cls()

        def __call__(self, text, voice_samples, padding, return_tensors, return_attention_mask):
            assert text == ["Speaker 1: hello from vibevoice"]
            assert voice_samples[0][0].endswith("first_impression.wav")
            assert padding is True
            assert return_tensors == "pt"
            assert return_attention_mask is True
            return {"input_ids": FakeTensor()}

        def save_audio(self, audio, output_path):
            sf.write(output_path, np.asarray(audio, dtype=np.float32), 24000)

    class FakeModel:
        @classmethod
        def from_pretrained(cls, model_id, torch_dtype, device_map, attn_implementation):
            assert model_id == "microsoft/VibeVoice-1.5B"
            assert torch_dtype == "float32"
            assert device_map == "cpu"
            assert attn_implementation == "sdpa"
            return cls()

        def eval(self):
            return None

        def set_ddpm_inference_steps(self, num_steps):
            assert num_steps == 4

        def generate(self, **kwargs):
            assert kwargs["cfg_scale"] == 1.6
            assert kwargs["generation_config"] == {"do_sample": False}
            assert kwargs["is_prefill"] is True
            return SimpleNamespace(
                speech_outputs=[np.sin(np.linspace(0, np.pi * 2, 2400, dtype=np.float32))]
            )

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        manual_seed=lambda seed: None,
        bfloat16="bfloat16",
        float32="float32",
        is_tensor=lambda value: isinstance(value, FakeTensor),
    )
    fake_modeling = SimpleNamespace(VibeVoiceForConditionalGenerationInference=FakeModel)
    fake_processor = SimpleNamespace(VibeVoiceProcessor=FakeProcessor)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "vibevoice", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "vibevoice.modular", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "vibevoice.processor", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "vibevoice.modular.modeling_vibevoice_inference", fake_modeling)
    monkeypatch.setitem(sys.modules, "vibevoice.processor.vibevoice_processor", fake_processor)
    monkeypatch.chdir(tmp_path)

    output = tmp_path / "vibevoice.wav"
    module._write_vibevoice_15b(
        "hello from vibevoice",
        output,
        {"cfg_scale": 1.6, "ddpm_steps": 4},
        voice_refs / "first_impression.wav",
    )

    assert output.exists()
    assert output.stat().st_size > 0
