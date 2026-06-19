from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Iterable, Protocol

import numpy as np
import soundfile as sf

from aispeechapp.cache_paths import configure_model_caches
from aispeechapp.candidates import PROJECT_ROOT


VOXCPM2_SAMPLE_RATE = 48000
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "voxcpm2_streaming"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "reports" / "voxcpm2_streaming.json"
DEFAULT_HISTORY_PATH = PROJECT_ROOT / "reports" / "voxcpm2_streaming_history.jsonl"


class VoxCPM2StreamingModel(Protocol):
    def generate_streaming(self, *args: object, **kwargs: object) -> Iterable[np.ndarray]:
        ...


class AudioSink(Protocol):
    def write(self, audio: np.ndarray) -> None:
        ...

    def close(self) -> None:
        ...


ModelFactory = Callable[[], VoxCPM2StreamingModel]
AudioSinkFactory = Callable[[int, str | int | None, float, str | float | None], AudioSink]
AudioObserver = Callable[[np.ndarray, int], None]


@dataclass(frozen=True)
class StreamingChunk:
    index: int
    elapsed_s: float
    samples: int
    audio_duration_s: float


@dataclass(frozen=True)
class StreamingSynthesisResult:
    text: str
    output_path: str
    reference_wav_path: str | None
    prompt_wav_path: str | None
    prompt_text: str | None
    sample_rate: int
    cfg_value: float
    inference_timesteps: int
    min_len: int
    max_len: int
    normalize: bool
    denoise: bool
    retry_badcase: bool
    retry_badcase_max_times: int
    retry_badcase_ratio_threshold: float
    seed: int | None
    quantization_method: str
    quantization_targets: str
    quantization_status: str
    torch_compile: bool
    torch_compile_mode: str
    torch_compile_status: str
    audio_normalization: bool
    audio_target_peak: float
    chunk_count: int
    first_chunk_latency_s: float | None
    total_elapsed_s: float
    audio_duration_s: float
    realtime_factor: float | None
    played_to_device: bool
    playback_mode: str
    audio_device: str | int | None
    playback_prebuffer_s: float
    audio_latency: str | float | None
    max_chunk_gap_s: float | None
    max_chunk_gap_over_audio_s: float | None
    chunks: list[StreamingChunk]


@lru_cache(maxsize=1)
def _default_model_factory() -> VoxCPM2StreamingModel:
    return _load_default_voxcpm2_model()


def _load_default_voxcpm2_model() -> VoxCPM2StreamingModel:
    configure_model_caches()
    from voxcpm import VoxCPM

    return VoxCPM.from_pretrained(
        "openbmb/VoxCPM2",
        load_denoiser=False,
        cache_dir=os.environ.get("HUGGINGFACE_HUB_CACHE"),
        optimize=False,
    )


@lru_cache(maxsize=1)
def _prepared_default_model_factory(
    cache_key: str,
    quantization_method: str,
    quantization_targets: str,
) -> tuple[VoxCPM2StreamingModel, str]:
    model = _load_default_voxcpm2_model()
    model, quantization_status = _maybe_quantize_model(
        model,
        method=quantization_method,
        targets=quantization_targets,
    )
    return model, quantization_status


def clear_default_model_caches() -> None:
    _default_model_factory.cache_clear()
    _prepared_default_model_factory.cache_clear()


def _to_mono_float32(chunk: np.ndarray) -> np.ndarray:
    array = np.asarray(chunk)
    if array.ndim == 2:
        array = array.mean(axis=1)
    return array.astype(np.float32, copy=False)


def _maybe_compile_model(
    model: VoxCPM2StreamingModel,
    *,
    enabled: bool,
    mode: str,
) -> tuple[VoxCPM2StreamingModel, str]:
    if not enabled:
        return model, "disabled"
    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on optional torch runtime
        return model, f"unavailable: {type(exc).__name__}: {exc}"
    if not hasattr(torch, "compile"):
        return model, "unavailable: torch.compile is not present"
    try:
        from torch import nn
    except Exception as exc:  # pragma: no cover - depends on optional torch runtime
        return model, f"unavailable: torch.nn import failed: {type(exc).__name__}: {exc}"
    if isinstance(model, nn.Module):
        try:
            return torch.compile(model, mode=mode), f"compiled:model:{mode}"
        except Exception as exc:  # pragma: no cover - depends on backend/model support
            return model, f"failed:model:{type(exc).__name__}: {exc}"

    compiled_names: list[str] = []
    try:
        model_vars = vars(model)
    except TypeError:
        model_vars = {}
    for name, value in model_vars.items():
        if name == "tts_model":
            # VoxCPM checks isinstance(self.tts_model, VoxCPM2Model) before
            # allowing reference_wav_path. Wrapping this attribute directly in
            # torch.compile breaks voice cloning even when the real model is v2.
            continue
        if isinstance(value, nn.Module):
            try:
                setattr(model, name, torch.compile(value, mode=mode))
                compiled_names.append(name)
            except Exception:
                continue
    if compiled_names:
        return model, f"compiled:modules:{','.join(sorted(compiled_names))}:{mode}"
    return model, "not_applicable: no torch.nn.Module target found"


def _set_torch_seed(seed: int | None) -> str:
    if seed is None:
        return "disabled"
    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on optional torch runtime
        return f"unavailable: {type(exc).__name__}: {exc}"
    torch.manual_seed(seed)
    cuda = getattr(torch, "cuda", None)
    if cuda is not None and cuda.is_available():
        cuda.manual_seed_all(seed)
        return "set:torch+cuda"
    return "set:torch"


def _get_nested_attr(root: object, path: str) -> object:
    value = root
    for part in path.split("."):
        value = getattr(value, part)
    return value


def _set_nested_attr(root: object, path: str, value: object) -> None:
    parent_path, _, attr_name = path.rpartition(".")
    parent = _get_nested_attr(root, parent_path) if parent_path else root
    setattr(parent, attr_name, value)


def _quantization_target_paths(targets: str) -> tuple[str, ...]:
    if targets == "base_lm_only":
        return ("tts_model.base_lm",)
    if targets == "lm_and_diffusion":
        return ("tts_model.base_lm", "tts_model.residual_lm", "tts_model.feat_decoder.estimator")
    return ("tts_model.base_lm", "tts_model.residual_lm")


def _maybe_quantize_model(
    model: VoxCPM2StreamingModel,
    *,
    method: str,
    targets: str,
) -> tuple[VoxCPM2StreamingModel, str]:
    if method == "disabled":
        return model, "disabled"
    if method == "bnb_int8_linear":
        return _maybe_quantize_model_bnb(model, method=method, targets=targets)
    if method in {"torchao_int8_weight_only", "torchao_int4_weight_only"}:
        return _maybe_quantize_model_torchao(model, method=method, targets=targets)
    if method != "dynamic_int8_cpu":
        return model, f"unsupported:{method}"
    try:
        import torch
        from torch import nn
        from torch.ao.quantization import quantize_dynamic
    except Exception as exc:  # pragma: no cover - depends on optional torch runtime
        return model, f"unavailable: {type(exc).__name__}: {exc}"

    tts_model = getattr(model, "tts_model", None)
    device = str(getattr(tts_model, "device", "")).lower()
    if device and not device.startswith("cpu"):
        return model, f"skipped:dynamic_int8_cpu requires CPU model, found {device}"

    quantized_paths: list[str] = []
    skipped_paths: list[str] = []
    for path in _quantization_target_paths(targets):
        try:
            module = _get_nested_attr(model, path)
        except AttributeError:
            skipped_paths.append(path)
            continue
        if not isinstance(module, nn.Module):
            skipped_paths.append(path)
            continue
        try:
            quantized_module = quantize_dynamic(
                module.cpu(),
                {nn.Linear},
                dtype=torch.qint8,
            )
            _set_nested_attr(model, path, quantized_module)
            quantized_paths.append(path)
        except Exception:
            skipped_paths.append(path)
    if quantized_paths:
        status = f"dynamic_int8_cpu:{','.join(quantized_paths)}"
        if skipped_paths:
            status += f";skipped:{','.join(skipped_paths)}"
        return model, status
    return model, f"not_applicable: no quantizable targets for {targets}"


def _maybe_quantize_model_torchao(
    model: VoxCPM2StreamingModel,
    *,
    method: str,
    targets: str,
) -> tuple[VoxCPM2StreamingModel, str]:
    try:
        from torch import nn
        from torchao.quantization import Int4WeightOnlyConfig, Int8WeightOnlyConfig, quantize_
    except Exception as exc:  # pragma: no cover - depends on optional torchao runtime
        return model, f"unavailable: torchao: {type(exc).__name__}: {exc}"

    if method == "torchao_int4_weight_only":
        config = Int4WeightOnlyConfig(group_size=128)
    else:
        config = Int8WeightOnlyConfig(version=2)

    quantized_paths: list[str] = []
    skipped_paths: list[str] = []
    for path in _quantization_target_paths(targets):
        try:
            module = _get_nested_attr(model, path)
        except AttributeError:
            skipped_paths.append(path)
            continue
        if not isinstance(module, nn.Module):
            skipped_paths.append(path)
            continue
        try:
            quantize_(module, config)
            quantized_paths.append(path)
        except Exception:
            skipped_paths.append(path)
    if quantized_paths:
        status = f"{method}:{','.join(quantized_paths)}"
        if skipped_paths:
            status += f";skipped:{','.join(skipped_paths)}"
        return model, status
    return model, f"not_applicable: torchao found no quantizable targets for {targets}"


def _replace_linear_with_bnb_int8(module: object) -> tuple[int, int]:
    import torch
    from torch import nn
    import bitsandbytes as bnb

    replaced = 0
    skipped = 0
    for child_name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            device = child.weight.device
            dtype = child.weight.dtype
            replacement = bnb.nn.Linear8bitLt(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
                has_fp16_weights=False,
                threshold=6.0,
                device=device,
            )
            with torch.no_grad():
                replacement.weight.data = child.weight.detach().to(device=device, dtype=torch.float16)
                if child.bias is not None and replacement.bias is not None:
                    replacement.bias.data = child.bias.detach().to(device=device, dtype=dtype)
            replacement.to(device=device)
            setattr(module, child_name, replacement)
            replaced += 1
        else:
            child_replaced, child_skipped = _replace_linear_with_bnb_int8(child)
            replaced += child_replaced
            skipped += child_skipped
    return replaced, skipped


def _maybe_quantize_model_bnb(
    model: VoxCPM2StreamingModel,
    *,
    method: str,
    targets: str,
) -> tuple[VoxCPM2StreamingModel, str]:
    try:
        import bitsandbytes as bnb
        from torch import nn
    except Exception as exc:  # pragma: no cover - depends on optional bitsandbytes runtime
        return model, f"unavailable: bitsandbytes: {type(exc).__name__}: {exc}"
    if not hasattr(bnb.nn, "Linear8bitLt"):
        return model, "unavailable: bitsandbytes Linear8bitLt is not present"

    target_statuses: list[str] = []
    total_replaced = 0
    skipped_paths: list[str] = []
    for path in _quantization_target_paths(targets):
        try:
            module = _get_nested_attr(model, path)
        except AttributeError:
            skipped_paths.append(path)
            continue
        if not isinstance(module, nn.Module):
            skipped_paths.append(path)
            continue
        try:
            replaced, _skipped = _replace_linear_with_bnb_int8(module)
        except Exception:
            skipped_paths.append(path)
            continue
        if replaced:
            total_replaced += replaced
            target_statuses.append(f"{path}:{replaced}")
        else:
            skipped_paths.append(path)
    if target_statuses:
        status = f"{method}:{','.join(target_statuses)}"
        if skipped_paths:
            status += f";skipped:{','.join(skipped_paths)}"
        return model, status
    return model, f"not_applicable: bitsandbytes found no Linear targets for {targets}"


class SoundDeviceSink:
    def __init__(
        self,
        sample_rate: int,
        device: str | int | None = None,
        prebuffer_s: float = 0.45,
        latency: str | float | None = "high",
    ) -> None:
        import sounddevice as sd

        self._sd = sd
        self._sample_rate = sample_rate
        self._device = device
        self._prebuffer_frames = max(0, int(prebuffer_s * sample_rate))
        self._latency = latency
        self._stream = None
        self._pending: list[np.ndarray] = []
        self._pending_frames = 0

    def _start(self) -> None:
        if self._stream is not None:
            return
        self._stream = self._sd.OutputStream(
            samplerate=self._sample_rate,
            channels=1,
            dtype="float32",
            device=self._device,
            latency=self._latency,
        )
        self._stream.start()
        for audio in self._pending:
            self._stream.write(audio.reshape(-1, 1))
        self._pending.clear()
        self._pending_frames = 0

    def write(self, audio: np.ndarray) -> None:
        if self._stream is None:
            self._pending.append(audio.copy())
            self._pending_frames += int(audio.shape[0])
            if self._pending_frames < self._prebuffer_frames:
                return
            self._start()
            return
        self._stream.write(audio.reshape(-1, 1))

    def close(self) -> None:
        if self._stream is None and self._pending:
            self._start()
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()


class StreamingPeakNormalizer:
    def __init__(
        self,
        *,
        target_peak: float = 0.85,
        max_gain: float = 4.0,
        smoothing: float = 0.2,
    ) -> None:
        self._target_peak = max(0.05, min(0.98, target_peak))
        self._max_gain = max(1.0, max_gain)
        self._smoothing = max(0.0, min(1.0, smoothing))
        self._gain = 1.0

    def process(self, audio: np.ndarray) -> np.ndarray:
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak > 1e-5:
            desired_gain = max(0.1, min(self._max_gain, self._target_peak / peak))
            self._gain = (self._gain * (1.0 - self._smoothing)) + (desired_gain * self._smoothing)
        normalized = audio * self._gain
        return np.clip(normalized, -0.98, 0.98).astype(np.float32, copy=False)


def synthesize_voxcpm2_streaming(
    *,
    text: str,
    output_path: Path,
    model_cache_key: str = "voxcpm2",
    reference_wav_path: Path | None = None,
    prompt_wav_path: Path | None = None,
    prompt_text: str | None = None,
    cfg_value: float = 2.0,
    inference_timesteps: int = 10,
    min_len: int = 2,
    max_len: int = 4096,
    normalize: bool = True,
    denoise: bool = False,
    retry_badcase: bool = False,
    retry_badcase_max_times: int = 3,
    retry_badcase_ratio_threshold: float = 6.0,
    seed: int | None = None,
    quantization_method: str = "disabled",
    quantization_targets: str = "lm_only",
    torch_compile: bool = False,
    torch_compile_mode: str = "reduce-overhead",
    audio_normalization: bool = False,
    audio_target_peak: float = 0.85,
    sample_rate: int = VOXCPM2_SAMPLE_RATE,
    play_audio: bool = False,
    playback_mode: str = "after_generation",
    audio_device: str | int | None = None,
    playback_prebuffer_s: float = 0.45,
    audio_latency: str | float | None = "high",
    model_factory: ModelFactory | None = None,
    audio_sink_factory: AudioSinkFactory = SoundDeviceSink,
    audio_observer: AudioObserver | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> StreamingSynthesisResult:
    if not text.strip():
        raise ValueError("text must not be empty")
    if reference_wav_path is not None and not reference_wav_path.exists():
        raise FileNotFoundError(f"reference_wav_path does not exist: {reference_wav_path}")
    if prompt_wav_path is not None and not prompt_wav_path.exists():
        raise FileNotFoundError(f"prompt_wav_path does not exist: {prompt_wav_path}")
    if (prompt_wav_path is None) != (prompt_text is None):
        raise ValueError("prompt_wav_path and prompt_text must be provided together")
    if playback_mode not in {"after_generation", "live"}:
        raise ValueError("playback_mode must be 'after_generation' or 'live'")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    default_model_factory_used = model_factory is None
    if default_model_factory_used and not torch_compile:
        model, quantization_status = _prepared_default_model_factory(
            model_cache_key,
            quantization_method,
            quantization_targets,
        )
    else:
        if default_model_factory_used:
            clear_default_model_caches()
            model = _load_default_voxcpm2_model()
        else:
            model = model_factory()
        model, quantization_status = _maybe_quantize_model(
            model,
            method=quantization_method,
            targets=quantization_targets,
        )
    _set_torch_seed(seed)
    model, torch_compile_status = _maybe_compile_model(
        model,
        enabled=torch_compile,
        mode=torch_compile_mode,
    )
    chunks: list[StreamingChunk] = []
    total_samples = 0
    first_chunk_latency: float | None = None
    start = clock()
    audio_sink = (
        audio_sink_factory(sample_rate, audio_device, playback_prebuffer_s, audio_latency)
        if play_audio and playback_mode == "live"
        else None
    )
    deferred_playback_chunks: list[np.ndarray] = []
    normalizer = (
        StreamingPeakNormalizer(target_peak=audio_target_peak)
        if audio_normalization
        else None
    )

    try:
        with sf.SoundFile(str(output_path), mode="w", samplerate=sample_rate, channels=1) as writer:
            for index, chunk in enumerate(
                model.generate_streaming(
                    text=text,
                    prompt_wav_path=str(prompt_wav_path) if prompt_wav_path else None,
                    prompt_text=prompt_text,
                    reference_wav_path=str(reference_wav_path) if reference_wav_path else None,
                    cfg_value=cfg_value,
                    inference_timesteps=inference_timesteps,
                    min_len=min_len,
                    max_len=max_len,
                    normalize=normalize,
                    denoise=denoise,
                    retry_badcase=retry_badcase,
                    retry_badcase_max_times=retry_badcase_max_times,
                    retry_badcase_ratio_threshold=retry_badcase_ratio_threshold,
                )
            ):
                now = clock()
                audio = _to_mono_float32(chunk)
                if normalizer is not None:
                    audio = normalizer.process(audio)
                if first_chunk_latency is None:
                    first_chunk_latency = round(now - start, 3)
                writer.write(audio)
                if audio_observer is not None:
                    audio_observer(audio, sample_rate)
                if audio_sink is not None:
                    audio_sink.write(audio)
                if play_audio and playback_mode == "after_generation":
                    deferred_playback_chunks.append(audio.copy())
                total_samples += int(audio.shape[0])
                chunks.append(
                    StreamingChunk(
                        index=index,
                        elapsed_s=round(now - start, 3),
                        samples=int(audio.shape[0]),
                        audio_duration_s=round(float(audio.shape[0]) / sample_rate, 3),
                    )
                )
    finally:
        if audio_sink is not None:
            audio_sink.close()
        if default_model_factory_used and torch_compile:
            clear_default_model_caches()

    if play_audio and playback_mode == "after_generation" and deferred_playback_chunks:
        deferred_sink = audio_sink_factory(sample_rate, audio_device, playback_prebuffer_s, audio_latency)
        try:
            for audio in deferred_playback_chunks:
                deferred_sink.write(audio)
        finally:
            deferred_sink.close()

    total_elapsed = round(clock() - start, 3)
    audio_duration = round(total_samples / sample_rate, 3)
    realtime_factor = round(total_elapsed / audio_duration, 4) if audio_duration else None
    chunk_gap_pairs = zip(chunks, chunks[1:], strict=False)
    chunk_gaps = [
        round(second.elapsed_s - first.elapsed_s, 3)
        for first, second in chunk_gap_pairs
    ]
    max_chunk_gap = max(chunk_gaps) if chunk_gaps else None
    max_chunk_audio_duration = max((chunk.audio_duration_s for chunk in chunks), default=0)
    max_chunk_gap_over_audio = (
        round(max_chunk_gap - max_chunk_audio_duration, 3)
        if max_chunk_gap is not None
        else None
    )
    return StreamingSynthesisResult(
        text=text,
        output_path=str(output_path),
        reference_wav_path=str(reference_wav_path) if reference_wav_path else None,
        prompt_wav_path=str(prompt_wav_path) if prompt_wav_path else None,
        prompt_text=prompt_text,
        sample_rate=sample_rate,
        cfg_value=cfg_value,
        inference_timesteps=inference_timesteps,
        min_len=min_len,
        max_len=max_len,
        normalize=normalize,
        denoise=denoise,
        retry_badcase=retry_badcase,
        retry_badcase_max_times=retry_badcase_max_times,
        retry_badcase_ratio_threshold=retry_badcase_ratio_threshold,
        seed=seed,
        quantization_method=quantization_method,
        quantization_targets=quantization_targets,
        quantization_status=quantization_status,
        torch_compile=torch_compile,
        torch_compile_mode=torch_compile_mode,
        torch_compile_status=torch_compile_status,
        audio_normalization=audio_normalization,
        audio_target_peak=audio_target_peak,
        chunk_count=len(chunks),
        first_chunk_latency_s=first_chunk_latency,
        total_elapsed_s=total_elapsed,
        audio_duration_s=audio_duration,
        realtime_factor=realtime_factor,
        played_to_device=play_audio,
        playback_mode=playback_mode,
        audio_device=audio_device,
        playback_prebuffer_s=playback_prebuffer_s,
        audio_latency=audio_latency,
        max_chunk_gap_s=max_chunk_gap,
        max_chunk_gap_over_audio_s=max_chunk_gap_over_audio,
        chunks=chunks,
    )


def write_streaming_report(
    result: StreamingSynthesisResult,
    report_path: Path = DEFAULT_REPORT_PATH,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")


def append_streaming_history(
    result: StreamingSynthesisResult,
    history_path: Path = DEFAULT_HISTORY_PATH,
) -> None:
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VoxCPM2 in streaming synthesis mode.")
    parser.add_argument("--text", required=True)
    parser.add_argument("--reference-wav-path", type=Path)
    parser.add_argument("--prompt-wav-path", type=Path)
    parser.add_argument("--prompt-text")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "voxcpm2_streaming.wav",
    )
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY_PATH)
    parser.add_argument("--play-audio", action="store_true")
    parser.add_argument("--playback-mode", choices=["after_generation", "live"], default="after_generation")
    parser.add_argument("--audio-device")
    parser.add_argument("--playback-prebuffer-s", type=float, default=0.45)
    parser.add_argument("--audio-latency", default="high")
    parser.add_argument("--cfg-value", type=float, default=2.0)
    parser.add_argument("--inference-timesteps", type=int, default=10)
    parser.add_argument("--min-len", type=int, default=2)
    parser.add_argument("--max-len", type=int, default=4096)
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--denoise", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--retry-badcase", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--retry-badcase-max-times", type=int, default=3)
    parser.add_argument("--retry-badcase-ratio-threshold", type=float, default=6.0)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--quantization-method", default="disabled")
    parser.add_argument("--quantization-targets", default="lm_only")
    parser.add_argument("--torch-compile", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--torch-compile-mode", default="reduce-overhead")
    parser.add_argument("--audio-normalization", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--audio-target-peak", type=float, default=0.85)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = synthesize_voxcpm2_streaming(
        text=args.text,
        output_path=args.output,
        reference_wav_path=args.reference_wav_path,
        prompt_wav_path=args.prompt_wav_path,
        prompt_text=args.prompt_text,
        cfg_value=args.cfg_value,
        inference_timesteps=args.inference_timesteps,
        min_len=args.min_len,
        max_len=args.max_len,
        normalize=args.normalize,
        denoise=args.denoise,
        retry_badcase=args.retry_badcase,
        retry_badcase_max_times=args.retry_badcase_max_times,
        retry_badcase_ratio_threshold=args.retry_badcase_ratio_threshold,
        seed=args.seed,
        quantization_method=args.quantization_method,
        quantization_targets=args.quantization_targets,
        torch_compile=args.torch_compile,
        torch_compile_mode=args.torch_compile_mode,
        audio_normalization=args.audio_normalization,
        audio_target_peak=args.audio_target_peak,
        play_audio=args.play_audio,
        playback_mode=args.playback_mode,
        audio_device=args.audio_device,
        playback_prebuffer_s=args.playback_prebuffer_s,
        audio_latency=args.audio_latency,
    )
    write_streaming_report(result, args.report)
    append_streaming_history(result, args.history)
    print(json.dumps(asdict(result), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
