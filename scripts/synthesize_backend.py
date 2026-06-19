from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from aispeechapp.cache_paths import DEFAULT_MODEL_ROOT, configure_model_caches

configure_model_caches()

import soundfile as sf  # noqa: E402

VOXCPM2_SAMPLE_RATE = 48000


def _write_audio_file(output: Path, audio: np.ndarray, sample_rate: int) -> None:
    suffix = output.suffix.lower()
    if suffix == ".wav":
        sf.write(output, audio, sample_rate)
        return

    if suffix != ".mp3":
        raise RuntimeError(f"Unsupported output extension '{output.suffix}'. Use .wav or .mp3.")

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("MP3 output requires ffmpeg to be available on PATH.")

    temp_wav = output.with_name(f"{output.stem}.tmp.wav")
    try:
        sf.write(temp_wav, audio, sample_rate)
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(temp_wav),
                "-codec:a",
                "libmp3lame",
                "-q:a",
                "2",
                str(output),
            ],
            check=True,
        )
    finally:
        temp_wav.unlink(missing_ok=True)


def _write_qwen3(
    candidate: str,
    text: str,
    language_code: str,
    language_hint: str,
    output: Path,
    options: dict[str, object],
) -> None:
    import torch
    from qwen_tts import Qwen3TTSModel

    model_id = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    model = Qwen3TTSModel.from_pretrained(
        model_id,
        device_map="cuda:0" if torch.cuda.is_available() else "cpu",
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        attn_implementation="sdpa",
    )
    speaker = str(options.get("speaker", "Ryan"))
    instruct = str(options.get("style_instruction") or "Speak clearly in a natural studio narration voice.")
    if language_code == "pt-PT":
        instruct = str(
            options.get("style_instruction")
            or (
                "Speak clearly in European Portuguese from Portugal, not Brazilian Portuguese. "
                "Use a natural Portugal accent and pronunciation."
            )
        )
    wavs, sr = model.generate_custom_voice(
        text=text,
        language=language_hint,
        speaker=speaker,
        instruct=instruct,
        max_new_tokens=int(options.get("max_new_tokens", 512)),
    )
    sf.write(output, wavs[0], sr)


def _write_voxcpm2(text: str, output: Path, options: dict[str, object]) -> None:
    from voxcpm import VoxCPM

    model = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False)
    wav = model.generate(
        text=text,
        prompt_wav_path=None,
        prompt_text=None,
        cfg_value=float(options.get("cfg_value", 2.0)),
        inference_timesteps=int(options.get("inference_timesteps", 10)),
        min_len=int(options.get("min_len", 2)),
        max_len=int(options.get("max_len", 4096)),
        normalize=bool(options.get("normalize", True)),
        denoise=bool(options.get("denoise", False)),
        retry_badcase=bool(options.get("retry_badcase", True)),
        retry_badcase_max_times=int(options.get("retry_badcase_max_times", 3)),
        retry_badcase_ratio_threshold=float(options.get("retry_badcase_ratio_threshold", 6.0)),
    )
    sf.write(output, wav, VOXCPM2_SAMPLE_RATE)


def _reference_text_path(reference_wav_path: Path | None) -> Path:
    if reference_wav_path is None:
        return Path("voice_refs/first_impression.txt")
    return reference_wav_path.with_suffix(".txt")


def _write_dots_tts(
    candidate: str,
    text: str,
    output: Path,
    options: dict[str, object],
    reference_wav_path: Path | None,
) -> None:
    from dots_tts.runtime import DotsTtsRuntime

    ref_audio = reference_wav_path or Path("voice_refs/first_impression.wav")
    ref_text = _reference_text_path(reference_wav_path)
    if not ref_audio.exists() or not ref_text.exists():
        raise RuntimeError(
            "dots.tts needs a reference WAV plus a same-name .txt transcript "
            f"for continuation cloning: {ref_audio}, {ref_text}"
        )

    model_id = "rednote-hilab/dots.tts-soar" if candidate == "dots_tts_soar" else "rednote-hilab/dots.tts-mf"
    runtime = DotsTtsRuntime.from_pretrained(model_id, precision="bfloat16")
    result = runtime.generate(
        text=text,
        prompt_audio_path=str(ref_audio),
        prompt_text=ref_text.read_text(encoding="utf-8").strip(),
        num_steps=int(options.get("num_steps", 10)),
        guidance_scale=float(options.get("guidance_scale", 1.2)),
    )
    sf.write(output, result["audio"].float().cpu().squeeze().numpy(), result["sample_rate"])


def _write_indextts2(text: str, output: Path, reference_wav_path: Path | None) -> None:
    from huggingface_hub import snapshot_download
    from indextts.infer_v2 import IndexTTS2

    ref_audio = reference_wav_path or Path("voice_refs/first_impression.wav")
    if not ref_audio.exists():
        raise RuntimeError(f"IndexTTS2 needs a reference WAV for zero-shot voice cloning: {ref_audio}")

    checkpoint_dir = DEFAULT_MODEL_ROOT / "indextts2-runtime" / "checkpoints"
    snapshot_download(
        repo_id="IndexTeam/IndexTTS-2",
        local_dir=checkpoint_dir,
        local_dir_use_symlinks=False,
    )
    tts = IndexTTS2(
        cfg_path=str(checkpoint_dir / "config.yaml"),
        model_dir=str(checkpoint_dir),
        use_fp16=True,
        use_cuda_kernel=False,
        use_deepspeed=False,
    )
    tts.infer(spk_audio_prompt=str(ref_audio), text=text, output_path=str(output), verbose=False)


def _write_omnivoice(text: str, output: Path, reference_wav_path: Path | None) -> None:
    import torch
    from omnivoice import OmniVoice

    ref_audio = reference_wav_path or Path("voice_refs/first_impression.wav")
    ref_text = _reference_text_path(reference_wav_path)
    if not ref_audio.exists() or not ref_text.exists():
        raise RuntimeError(
            "OmniVoice needs a reference WAV plus a same-name .txt transcript "
            f"for voice cloning: {ref_audio}, {ref_text}"
        )

    device_map = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map=device_map, dtype=dtype)
    audio = model.generate(
        text=text,
        ref_audio=str(ref_audio),
        ref_text=ref_text.read_text(encoding="utf-8").strip(),
    )
    _write_audio_file(output, np.asarray(audio[0], dtype=np.float32), 24000)


def _write_vibevoice_15b(
    text: str,
    output: Path,
    options: dict[str, object],
    reference_wav_path: Path | None,
) -> None:
    import torch
    from vibevoice.modular.modeling_vibevoice_inference import (
        VibeVoiceForConditionalGenerationInference,
    )
    from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor

    ref_audio = reference_wav_path or Path("voice_refs/first_impression.wav")
    if not ref_audio.exists():
        raise RuntimeError(f"VibeVoice 1.5B needs a reference WAV for voice cloning: {ref_audio}")

    if "seed" in options:
        seed = int(options["seed"])
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    load_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model_id = str(options.get("model_id", "microsoft/VibeVoice-1.5B"))
    attn_implementation = str(options.get("attn_implementation", "sdpa"))

    processor = VibeVoiceProcessor.from_pretrained(model_id)
    model = VibeVoiceForConditionalGenerationInference.from_pretrained(
        model_id,
        torch_dtype=load_dtype,
        device_map=device,
        attn_implementation=attn_implementation,
    )
    model.eval()
    model.set_ddpm_inference_steps(num_steps=int(options.get("ddpm_steps", 10)))

    speaker_script = text if text.lstrip().lower().startswith("speaker ") else f"Speaker 1: {text}"
    inputs = processor(
        text=[speaker_script.replace("’", "'")],
        voice_samples=[[str(ref_audio)]],
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )
    for key, value in inputs.items():
        if torch.is_tensor(value):
            inputs[key] = value.to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=None,
        cfg_scale=float(options.get("cfg_scale", 1.3)),
        tokenizer=processor.tokenizer,
        generation_config={"do_sample": bool(options.get("do_sample", False))},
        verbose=bool(options.get("verbose", False)),
        is_prefill=bool(options.get("voice_clone", True)),
    )
    if not getattr(outputs, "speech_outputs", None) or outputs.speech_outputs[0] is None:
        raise RuntimeError("VibeVoice generation did not return speech audio.")

    if output.suffix.lower() == ".wav":
        processor.save_audio(outputs.speech_outputs[0], output_path=str(output))
        return

    temp_wav = output.with_name(f"{output.stem}.tmp.wav")
    try:
        processor.save_audio(outputs.speech_outputs[0], output_path=str(temp_wav))
        audio, sample_rate = sf.read(temp_wav)
        _write_audio_file(output, np.asarray(audio), int(sample_rate))
    finally:
        temp_wav.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backend-specific synthesis helper.")
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--text")
    parser.add_argument("--text-file", type=Path)
    parser.add_argument("--language-code", required=True)
    parser.add_argument("--language-hint", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-wav-path", type=Path)
    parser.add_argument("--options-json", default="{}")
    args = parser.parse_args()
    if args.text_file:
        args.text = args.text_file.read_text(encoding="utf-8").strip()
    if not args.text:
        parser.error("Pass --text or --text-file.")
    try:
        args.options = json.loads(args.options_json)
    except json.JSONDecodeError as exc:
        parser.error(f"--options-json must be valid JSON: {exc}")
    if not isinstance(args.options, dict):
        parser.error("--options-json must decode to a JSON object.")
    return args


def main() -> int:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.candidate == "qwen3_tts_17b_customvoice":
        _write_qwen3(
            args.candidate,
            args.text,
            args.language_code,
            args.language_hint,
            args.output,
            args.options,
        )
    elif args.candidate == "voxcpm2":
        _write_voxcpm2(args.text, args.output, args.options)
    elif args.candidate in {"dots_tts_soar", "dots_tts_mf"}:
        _write_dots_tts(args.candidate, args.text, args.output, args.options, args.reference_wav_path)
    elif args.candidate == "indextts2":
        _write_indextts2(args.text, args.output, args.reference_wav_path)
    elif args.candidate == "omnivoice":
        _write_omnivoice(args.text, args.output, args.reference_wav_path)
    elif args.candidate == "microsoft_vibevoice_15b":
        _write_vibevoice_15b(args.text, args.output, args.options, args.reference_wav_path)
    else:
        raise RuntimeError(f"No synthesis implementation for {args.candidate}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
