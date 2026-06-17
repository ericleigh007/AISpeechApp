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
    chunk_count: int
    first_chunk_latency_s: float | None
    total_elapsed_s: float
    audio_duration_s: float
    realtime_factor: float | None
    played_to_device: bool
    audio_device: str | int | None
    playback_prebuffer_s: float
    audio_latency: str | float | None
    max_chunk_gap_s: float | None
    max_chunk_gap_over_audio_s: float | None
    chunks: list[StreamingChunk]


@lru_cache(maxsize=1)
def _default_model_factory() -> VoxCPM2StreamingModel:
    configure_model_caches()
    from voxcpm import VoxCPM

    return VoxCPM.from_pretrained(
        "openbmb/VoxCPM2",
        load_denoiser=False,
        cache_dir=os.environ.get("HUGGINGFACE_HUB_CACHE"),
    )


def _to_mono_float32(chunk: np.ndarray) -> np.ndarray:
    array = np.asarray(chunk)
    if array.ndim == 2:
        array = array.mean(axis=1)
    return array.astype(np.float32, copy=False)


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
        if self._pending:
            self._start()
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()


def synthesize_voxcpm2_streaming(
    *,
    text: str,
    output_path: Path,
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
    sample_rate: int = VOXCPM2_SAMPLE_RATE,
    play_audio: bool = False,
    audio_device: str | int | None = None,
    playback_prebuffer_s: float = 0.45,
    audio_latency: str | float | None = "high",
    model_factory: ModelFactory = _default_model_factory,
    audio_sink_factory: AudioSinkFactory = SoundDeviceSink,
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

    output_path.parent.mkdir(parents=True, exist_ok=True)
    model = model_factory()
    chunks: list[StreamingChunk] = []
    total_samples = 0
    first_chunk_latency: float | None = None
    start = clock()
    audio_sink = (
        audio_sink_factory(sample_rate, audio_device, playback_prebuffer_s, audio_latency)
        if play_audio
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
                if first_chunk_latency is None:
                    first_chunk_latency = round(now - start, 3)
                writer.write(audio)
                if audio_sink is not None:
                    audio_sink.write(audio)
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
        chunk_count=len(chunks),
        first_chunk_latency_s=first_chunk_latency,
        total_elapsed_s=total_elapsed,
        audio_duration_s=audio_duration,
        realtime_factor=realtime_factor,
        played_to_device=play_audio,
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
        play_audio=args.play_audio,
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
