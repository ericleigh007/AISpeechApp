from __future__ import annotations

import argparse
import sys
from pathlib import Path

from aispeechapp.cache_paths import DEFAULT_MODEL_ROOT, configure_model_caches

configure_model_caches()

import soundfile as sf  # noqa: E402

VOXCPM2_SAMPLE_RATE = 48000


def _write_qwen3(candidate: str, text: str, language_code: str, language_hint: str, output: Path) -> None:
    import torch
    from qwen_tts import Qwen3TTSModel

    model_id = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    model = Qwen3TTSModel.from_pretrained(
        model_id,
        device_map="cuda:0" if torch.cuda.is_available() else "cpu",
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        attn_implementation="sdpa",
    )
    speaker = "Ryan"
    instruct = "Speak clearly in a natural studio narration voice."
    if language_code == "pt-PT":
        instruct = (
            "Speak clearly in European Portuguese from Portugal, not Brazilian Portuguese. "
            "Use a natural Portugal accent and pronunciation."
        )
    wavs, sr = model.generate_custom_voice(
        text=text,
        language=language_hint,
        speaker=speaker,
        instruct=instruct,
        max_new_tokens=512,
    )
    sf.write(output, wavs[0], sr)


def _write_voxcpm2(text: str, output: Path) -> None:
    from voxcpm import VoxCPM

    model = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False)
    wav = model.generate(
        text=text,
        prompt_wav_path=None,
        prompt_text=None,
        cfg_value=2.0,
        inference_timesteps=10,
        normalize=True,
        denoise=False,
        retry_badcase=True,
        retry_badcase_max_times=3,
        retry_badcase_ratio_threshold=6.0,
    )
    sf.write(output, wav, VOXCPM2_SAMPLE_RATE)


def _write_dots_tts(candidate: str, text: str, output: Path) -> None:
    from dots_tts.runtime import DotsTtsRuntime

    ref_audio = Path("voice_refs/first_impression.wav")
    ref_text = Path("voice_refs/first_impression.txt")
    if not ref_audio.exists() or not ref_text.exists():
        raise RuntimeError(
            "dots.tts needs voice_refs/first_impression.wav and "
            "voice_refs/first_impression.txt for continuation cloning."
        )

    model_id = "rednote-hilab/dots.tts-soar" if candidate == "dots_tts_soar" else "rednote-hilab/dots.tts-mf"
    runtime = DotsTtsRuntime.from_pretrained(model_id, precision="bfloat16")
    result = runtime.generate(
        text=text,
        prompt_audio_path=str(ref_audio),
        prompt_text=ref_text.read_text(encoding="utf-8").strip(),
        num_steps=10,
        guidance_scale=1.2,
    )
    sf.write(output, result["audio"].float().cpu().squeeze().numpy(), result["sample_rate"])


def _write_indextts2(text: str, output: Path) -> None:
    from huggingface_hub import snapshot_download
    from indextts.infer_v2 import IndexTTS2

    ref_audio = Path("voice_refs/first_impression.wav")
    if not ref_audio.exists():
        raise RuntimeError("IndexTTS2 needs voice_refs/first_impression.wav for zero-shot voice cloning.")

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backend-specific synthesis helper.")
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--text")
    parser.add_argument("--text-file", type=Path)
    parser.add_argument("--language-code", required=True)
    parser.add_argument("--language-hint", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.text_file:
        args.text = args.text_file.read_text(encoding="utf-8").strip()
    if not args.text:
        parser.error("Pass --text or --text-file.")
    return args


def main() -> int:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.candidate == "qwen3_tts_17b_customvoice":
        _write_qwen3(args.candidate, args.text, args.language_code, args.language_hint, args.output)
    elif args.candidate == "voxcpm2":
        _write_voxcpm2(args.text, args.output)
    elif args.candidate in {"dots_tts_soar", "dots_tts_mf"}:
        _write_dots_tts(args.candidate, args.text, args.output)
    elif args.candidate == "indextts2":
        _write_indextts2(args.text, args.output)
    else:
        raise RuntimeError(f"No synthesis implementation for {args.candidate}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
