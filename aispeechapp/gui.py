from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

import numpy as np

from aispeechapp.candidates import PROJECT_ROOT, load_candidates
from aispeechapp.voxcpm2_streaming import (
    DEFAULT_HISTORY_PATH,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_REPORT_PATH,
    append_streaming_history,
    synthesize_voxcpm2_streaming,
    write_streaming_report,
)


OMNICHAT_VOICES_DIR = PROJECT_ROOT.parent / "OmniChat" / "voices"
DEFAULT_GUI_SETTINGS_PATH = PROJECT_ROOT / "configs" / "gui_settings.local.json"
GUI_SYNTHESIS_CANDIDATE_IDS = {
    "qwen3_tts_17b_customvoice",
    "voxcpm2",
    "voxcpm2_quantized",
    "voxcpm2_torchao_int8",
    "voxcpm2_bnb_int8",
    "dots_tts_soar",
    "dots_tts_mf",
    "indextts2",
    "omnivoice",
    "microsoft_vibevoice_15b",
}


def _load_gui_settings(settings_path: Path = DEFAULT_GUI_SETTINGS_PATH) -> dict:
    if not settings_path.exists():
        return {}
    try:
        payload = json.loads(settings_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _save_gui_settings(settings: dict, settings_path: Path = DEFAULT_GUI_SETTINGS_PATH) -> None:
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(json.dumps(settings, indent=2, sort_keys=True), encoding="utf-8")


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def _capability_label(capabilities: set[str], *names: str) -> str:
    return "yes" if any(name in capabilities for name in names) else "no"


def _language_scope(capabilities: set[str]) -> str:
    if "massively_multilingual" in capabilities:
        return "massive multi"
    if "multilingual" in capabilities:
        return "multi"
    if "bilingual" in capabilities:
        return "bi"
    if "japanese" in capabilities:
        return "ja"
    return "-"


def _audio_rate(capabilities: set[str]) -> str:
    for rate in ("48khz", "24khz"):
        if rate in capabilities:
            return rate.upper()
    return "-"


def _control_summary(capabilities: set[str]) -> str:
    controls = []
    labels = {
        "voice_design": "voice design",
        "emotion_control": "emotion",
        "emotion_tags": "emotion",
        "duration_control": "duration",
        "pronunciation_control": "pronunciation",
        "phoneme_control": "phoneme",
        "style_control": "style",
        "multispeaker": "multi-speaker",
        "long_form": "long-form",
        "low_latency": "low latency",
    }
    for capability, label in labels.items():
        if capability in capabilities and label not in controls:
            controls.append(label)
    return ", ".join(controls) if controls else "-"


def _model_scale(name: object, repo_id: object, explicit_scale: object = "") -> str:
    if str(explicit_scale).strip():
        return str(explicit_scale).strip()
    text = f"{name} {repo_id}"
    patterns = [
        (r"(\d+(?:\.\d+)?)\s*B\b", "B"),
        (r"(\d+(?:\.\d+)?)\s*M\b", "M"),
        (r"(\d+(?:\.\d+)?)B\b", "B"),
        (r"(\d+(?:\.\d+)?)M\b", "M"),
    ]
    for pattern, suffix in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return f"{match.group(1)}{suffix}"
    return "unknown"


SMOKE_TABLE_COLUMNS = [
    "Candidate",
    "Priority",
    "GUI",
    "Clone",
    "Stream",
    "Lang",
    "Audio",
    "Size",
    "Controls",
    "Metadata",
    "Imports",
    "Cache",
    "Status",
]


def _smoke_report_rows(payload: dict) -> list[dict[str, object]]:
    results = payload.get("results", [])
    if not isinstance(results, list):
        return []
    rows = []
    for result in sorted(results, key=lambda item: str(item.get("candidate_id", "")).lower()):
        imports = result.get("import_checks", [])
        imports_ok = all(check.get("available") for check in imports) if isinstance(imports, list) else False
        details = result.get("details", {})
        capabilities = set(details.get("capabilities", []) if isinstance(details, dict) else [])
        candidate_id = str(result.get("candidate_id", ""))
        gui_runnable = candidate_id in GUI_SYNTHESIS_CANDIDATE_IDS
        cells = [
            candidate_id,
            result.get("priority", ""),
            "yes" if gui_runnable else "no",
            _capability_label(capabilities, "voice_clone"),
            _capability_label(capabilities, "streaming", "low_latency"),
            _language_scope(capabilities),
            _audio_rate(capabilities),
            _model_scale(
                result.get("name", ""),
                result.get("repo_id", ""),
                details.get("model_scale", "") if isinstance(details, dict) else "",
            ),
            _control_summary(capabilities),
            _yes_no(bool(result.get("metadata_ok"))),
            _yes_no(imports_ok),
            _yes_no(bool(result.get("cache_present"))),
            result.get("status", ""),
        ]
        rows.append(
            {
                "values": [str(cell) for cell in cells],
                "gui_runnable": gui_runnable,
            }
        )
    return rows


def _text_to_markdown_fallback(text: str) -> str:
    if not text.strip():
        return "_No output._"
    return "```text\n" + text.replace("```", "`\u200b``") + "\n```"


def _format_report_age(payload: dict, *, now: float | None = None) -> str:
    generated_at = payload.get("generated_at_unix")
    if not isinstance(generated_at, int | float):
        return "Report age: unknown"
    current = time.time() if now is None else now
    delta_s = int(current - float(generated_at))
    if delta_s < 0:
        age = "from the future"
    elif delta_s < 60:
        age = f"{delta_s}s old"
    elif delta_s < 3600:
        age = f"{delta_s // 60}m old"
    elif delta_s < 86400:
        age = f"{delta_s // 3600}h {(delta_s % 3600) // 60}m old"
    else:
        age = f"{delta_s // 86400}d {(delta_s % 86400) // 3600}h old"
    generated_text = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(float(generated_at)))
    return f"Report generated: {generated_text} ({age})"


def _run_smoke(candidate_id: str | None = None) -> str:
    import subprocess

    cmd = [sys.executable, "-m", "aispeechapp.smoke", "--metadata-only"]
    if candidate_id is None:
        cmd.append("--all")
    else:
        cmd.extend(["--candidate", candidate_id])
    completed = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    report_path = PROJECT_ROOT / "reports" / "smoke_metadata.json"
    if report_path.exists():
        report_text = report_path.read_text(encoding="utf-8")
        if completed.stderr.strip():
            payload = json.loads(report_text)
            payload["stderr"] = completed.stderr
            return json.dumps(payload)
        return report_text
    return _text_to_markdown_fallback(completed.stdout + completed.stderr)


def _load_latest_report() -> str:
    path = PROJECT_ROOT / "reports" / "smoke_metadata.json"
    if not path.exists():
        return "_No report written yet._"
    return path.read_text(encoding="utf-8")


def _load_audio_devices() -> list[str]:
    try:
        import sounddevice as sd
    except Exception:
        return []

    devices = []
    for index, device in enumerate(sd.query_devices()):
        if int(device.get("max_output_channels", 0)) > 0:
            devices.append(f"{index}: {device['name']}")
    return devices


def _load_voice_references() -> list[tuple[str, str]]:
    if not OMNICHAT_VOICES_DIR.exists():
        return []
    return [
        (path.stem.replace("_", " ").title(), str(path))
        for path in sorted(OMNICHAT_VOICES_DIR.glob("*.wav"))
    ]


def _safe_filename_part(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9]+", "_", value.strip()).strip("_").lower()
    return safe or "unknown"


def _display_voice_name(combo_text: str, reference_wav_path: str) -> str:
    if combo_text and combo_text != "Custom reference WAV":
        return combo_text
    if reference_wav_path:
        return Path(reference_wav_path).stem.replace("_", " ").title()
    return "Custom Voice"


def _default_synthesis_prompt(model_name: str, voice_name: str) -> str:
    return (
        f"This is {model_name}, cloning the voice named {voice_name}. "
        "This sample checks clear diction, natural pacing, expressive volume, "
        "and careful pronunciation across several moderately complex words."
    )


def _default_synthesis_output_path(
    model_name: str,
    voice_name: str,
    *,
    timestamp: str | None = None,
) -> Path:
    generated_at = timestamp or time.strftime("%Y%m%d_%H%M%S")
    filename = f"{_safe_filename_part(voice_name)}__{generated_at}.wav"
    return PROJECT_ROOT / "outputs" / _safe_filename_part(model_name) / filename


def _sanitize_control_instruction(value: str) -> str:
    return re.sub(r"[()（）]", "", value).strip()


def _effective_synthesis_text(
    target_text: str,
    text_controls: tuple[dict[str, object], ...],
    text_control_options: dict[str, str],
) -> str:
    effective_text = target_text
    for control in text_controls:
        control_id = str(control.get("id", ""))
        value = text_control_options.get(control_id, "").strip()
        if not value:
            continue
        if control.get("mode") == "parenthesized_prefix":
            clean = _sanitize_control_instruction(value)
            if clean:
                effective_text = f"({clean}){effective_text}"
    return effective_text


def _text_control_generation_options(
    text_controls: tuple[dict[str, object], ...],
    text_control_options: dict[str, str],
) -> dict[str, str]:
    options: dict[str, str] = {}
    for control in text_controls:
        if control.get("mode") != "generation_option":
            continue
        control_id = str(control.get("id", ""))
        value = text_control_options.get(control_id, "").strip()
        if value:
            options[str(control.get("option_id", control_id))] = value
    return options


def _sidecar_path(output_path: Path) -> Path:
    return output_path.with_suffix(f"{output_path.suffix}.json")


def _write_generation_sidecar(output_path: Path, metadata: dict[str, object]) -> Path:
    sidecar_path = _sidecar_path(output_path)
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return sidecar_path


def _run_voxcpm2_streaming(
    *,
    candidate_id: str = "voxcpm2",
    candidate_name: str,
    voice_name: str,
    text: str,
    reference_wav_path: str,
    output_path: str,
    play_audio: bool,
    audio_device: str | None,
    playback_prebuffer_s: float,
    audio_latency: str | None,
    spoken_text: str | None = None,
    text_control_options: dict[str, str] | None = None,
    generation_options: dict[str, object] | None = None,
) -> str:
    options = generation_options or {}
    result = synthesize_voxcpm2_streaming(
        text=text,
        output_path=Path(output_path),
        model_cache_key=candidate_id,
        reference_wav_path=Path(reference_wav_path) if reference_wav_path else None,
        play_audio=play_audio,
        audio_device=audio_device.split(":", 1)[0] if audio_device else None,
        playback_prebuffer_s=playback_prebuffer_s,
        audio_latency=audio_latency,
        cfg_value=float(options.get("cfg_value", 2.0)),
        inference_timesteps=int(options.get("inference_timesteps", 10)),
        min_len=int(options.get("min_len", 2)),
        max_len=int(options.get("max_len", 4096)),
        normalize=bool(options.get("normalize", True)),
        denoise=bool(options.get("denoise", False)),
        retry_badcase=bool(options.get("retry_badcase", False)),
        retry_badcase_max_times=int(options.get("retry_badcase_max_times", 3)),
        retry_badcase_ratio_threshold=float(options.get("retry_badcase_ratio_threshold", 6.0)),
        seed=int(options.get("seed", 0)),
        quantization_method=str(options.get("quantization_method", "disabled")),
        quantization_targets=str(options.get("quantization_targets", "lm_only")),
        torch_compile=bool(options.get("torch_compile", False)),
        torch_compile_mode=str(options.get("torch_compile_mode", "reduce-overhead")),
        audio_normalization=bool(options.get("audio_normalization", False)),
        audio_target_peak=float(options.get("audio_target_peak", 0.85)),
        playback_mode=str(options.get("playback_mode", "after_generation")),
    )
    write_streaming_report(result, DEFAULT_REPORT_PATH)
    append_streaming_history(result, DEFAULT_HISTORY_PATH)
    payload = result.__dict__ | {"chunks": [chunk.__dict__ for chunk in result.chunks]}
    payload["candidate_id"] = candidate_id
    payload["candidate_name"] = candidate_name
    payload["voice_name"] = voice_name
    payload["time_to_first_output_s"] = result.first_chunk_latency_s
    payload["spoken_text"] = spoken_text or text
    payload["effective_text"] = text
    payload["text_control_options"] = text_control_options or {}
    sidecar_path = _write_generation_sidecar(
        Path(output_path),
        {
            "candidate_id": candidate_id,
            "candidate_name": candidate_name,
            "voice_name": voice_name,
            "reference_wav_path": reference_wav_path,
            "text": text,
            "spoken_text": spoken_text or text,
            "effective_text": text,
            "text_control_options": text_control_options or {},
            "output_path": output_path,
            "status": "complete",
            "played_to_device": result.played_to_device,
            "audio_device": result.audio_device,
            "playback_prebuffer_s": result.playback_prebuffer_s,
            "audio_latency": result.audio_latency,
            "audio_duration_s": result.audio_duration_s,
            "time_to_first_output_s": result.first_chunk_latency_s,
            "quantization_method": result.quantization_method,
            "quantization_targets": result.quantization_targets,
            "quantization_status": result.quantization_status,
            "torch_compile": result.torch_compile,
            "torch_compile_mode": result.torch_compile_mode,
            "torch_compile_status": result.torch_compile_status,
            "generation_options": generation_options or {},
            "result": payload,
            "generated_at_unix": time.time(),
        },
    )
    payload["metadata_path"] = str(sidecar_path)
    return json.dumps(payload, indent=2)


def _run_backend_synthesis(
    *,
    candidate_id: str,
    candidate_name: str,
    voice_name: str,
    text: str,
    output_path: str,
    language_code: str,
    language_hint: str,
    reference_wav_path: str,
    play_audio: bool,
    audio_device: str | None,
    playback_prebuffer_s: float,
    audio_latency: str | None,
    spoken_text: str | None = None,
    text_control_options: dict[str, str] | None = None,
    generation_options: dict[str, object] | None = None,
) -> str:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "synthesize_backend.py"),
        "--candidate",
        candidate_id,
        "--text",
        text,
        "--language-code",
        language_code,
        "--language-hint",
        language_hint,
        "--output",
        output_path,
        "--options-json",
        json.dumps(generation_options or {}),
    ]
    if reference_wav_path:
        cmd.extend(["--reference-wav-path", reference_wav_path])
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.unlink(missing_ok=True)
    started = time.perf_counter()
    first_output_s: float | None = None
    with (
        tempfile.TemporaryFile(mode="w+", encoding="utf-8", errors="replace") as stdout_file,
        tempfile.TemporaryFile(mode="w+", encoding="utf-8", errors="replace") as stderr_file,
    ):
        process = subprocess.Popen(
            cmd,
            cwd=PROJECT_ROOT,
            text=True,
            stdout=stdout_file,
            stderr=stderr_file,
        )
        while process.poll() is None:
            if first_output_s is None and output_file.exists() and output_file.stat().st_size > 0:
                first_output_s = round(time.perf_counter() - started, 3)
            time.sleep(0.05)
        if first_output_s is None and output_file.exists() and output_file.stat().st_size > 0:
            first_output_s = round(time.perf_counter() - started, 3)
        returncode = process.wait()
        stdout_file.seek(0)
        stderr_file.seek(0)
        stdout_text = stdout_file.read()
        stderr_text = stderr_file.read()
    elapsed = round(time.perf_counter() - started, 3)
    payload = {
        "candidate_id": candidate_id,
        "candidate_name": candidate_name,
        "voice_name": voice_name,
        "reference_wav_path": reference_wav_path,
        "output_path": output_path,
        "elapsed_s": elapsed,
        "time_to_first_output_s": first_output_s,
        "returncode": returncode,
        "stdout": stdout_text,
        "stderr": stderr_text,
        "generation_options": generation_options or {},
        "spoken_text": spoken_text or text,
        "effective_text": text,
        "text_control_options": text_control_options or {},
        "played_to_device": False,
        "audio_device": audio_device,
        "playback_prebuffer_s": playback_prebuffer_s,
        "audio_latency": audio_latency,
    }
    if returncode != 0:
        payload["status"] = "failed"
    else:
        payload["status"] = "complete"
        payload["output_exists"] = output_file.exists()
        if play_audio and output_file.exists():
            playback = _play_generated_audio_file(
                output_file,
                audio_device=audio_device,
                playback_prebuffer_s=playback_prebuffer_s,
                audio_latency=audio_latency,
                normalize=bool((generation_options or {}).get("audio_normalization", False)),
                target_peak=float((generation_options or {}).get("audio_target_peak", 0.85)),
            )
            payload.update(playback)
    sidecar_path = _write_generation_sidecar(
        Path(output_path),
        {
            **payload,
            "text": text,
            "spoken_text": spoken_text or text,
            "effective_text": text,
            "text_control_options": text_control_options or {},
            "language_code": language_code,
            "language_hint": language_hint,
            "generated_at_unix": time.time(),
        },
    )
    payload["metadata_path"] = str(sidecar_path)
    return json.dumps(payload, indent=2)


def _play_generated_audio_file(
    output_file: Path,
    *,
    audio_device: str | None,
    playback_prebuffer_s: float,
    audio_latency: str | None,
    normalize: bool,
    target_peak: float,
) -> dict[str, object]:
    import soundfile as sf
    import sounddevice as sd

    audio, sample_rate = sf.read(str(output_file), dtype="float32", always_2d=False)
    samples = np.asarray(audio, dtype=np.float32)
    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    samples = samples.reshape(-1)

    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    if normalize and peak > 0:
        samples = np.clip(samples * (target_peak / peak), -1.0, 1.0).astype(np.float32)

    device = audio_device.split(":", 1)[0] if audio_device else None
    sd.play(samples.reshape(-1, 1), samplerate=int(sample_rate), device=device, blocking=True)
    sd.stop()

    return {
        "played_to_device": True,
        "audio_sample_rate": int(sample_rate),
        "audio_duration_s": round(float(samples.shape[0]) / int(sample_rate), 3)
        if sample_rate
        else 0.0,
        "audio_peak": round(peak, 6),
        "audio_normalization": normalize,
        "audio_target_peak": target_peak,
    }


def create_main_window(
    *,
    run_smoke_func=_run_smoke,
    load_latest_report_func=_load_latest_report,
    load_audio_devices_func=_load_audio_devices,
    load_voice_references_func=_load_voice_references,
    run_voxcpm2_streaming_func=_run_voxcpm2_streaming,
    run_backend_synthesis_func=_run_backend_synthesis,
    settings_path: Path = DEFAULT_GUI_SETTINGS_PATH,
):
    from PySide6 import QtCore, QtGui, QtWidgets


    class AISpeechWindow(QtWidgets.QMainWindow):
        pass

    class SynthesisWorker(QtCore.QObject):
        finished = QtCore.Signal(str, str)

        def __init__(
            self,
            *,
            candidate_id: str,
            backend_kwargs: dict,
        ) -> None:
            super().__init__()
            self._candidate_id = candidate_id
            self._backend_kwargs = backend_kwargs

        @QtCore.Slot()
        def run(self) -> None:
            try:
                result = run_backend_synthesis_func(**self._backend_kwargs)
            except Exception as exc:  # pragma: no cover - exercised through GUI smoke behavior
                result = json.dumps(
                    {
                        "status": "failed",
                        "candidate_id": self._candidate_id,
                        "error": str(exc),
                    },
                    indent=2,
                )
            self.finished.emit(self._candidate_id, result)

    def name(widget: QtWidgets.QWidget, object_name: str, accessible_name: str | None = None):
        widget.setObjectName(object_name)
        widget.setAccessibleName(accessible_name or object_name)
        return widget

    def help_text(widget: QtWidgets.QWidget, text: str):
        widget.setToolTip(text)
        widget.setStatusTip(text)
        return widget

    candidates = load_candidates()
    candidate_by_id = {candidate.id: candidate for candidate in candidates}
    gui_settings = _load_gui_settings(settings_path)

    window = AISpeechWindow()
    name(window, "main_window", "AISpeechApp Main Window")
    window.setWindowTitle("AISpeechApp - Local TTS Comparison Lab")
    window.resize(980, 680)
    window.statusBar().showMessage("Ready")

    central = QtWidgets.QWidget()
    name(central, "central_widget", "AISpeechApp Central Widget")
    window.setCentralWidget(central)
    layout = QtWidgets.QVBoxLayout(central)
    tabs = QtWidgets.QTabWidget()
    name(tabs, "main_tabs", "Main Tabs")
    layout.addWidget(tabs)

    smoke_tab = QtWidgets.QWidget()
    smoke_layout = QtWidgets.QVBoxLayout(smoke_tab)
    controls = QtWidgets.QHBoxLayout()
    candidate_box = QtWidgets.QComboBox()
    name(candidate_box, "candidate_box", "Candidate Selector")
    candidate_box.addItem("All candidates", None)
    for candidate in candidates:
        candidate_box.addItem(f"P{candidate.priority} - {candidate.name}", candidate.id)

    run_button = QtWidgets.QPushButton("Run Metadata Smoke")
    name(run_button, "run_smoke_button", "Run Metadata Smoke")
    refresh_button = QtWidgets.QPushButton("Load Latest Report")
    name(refresh_button, "refresh_smoke_report_button", "Load Latest Report")
    controls.addWidget(candidate_box, 1)
    controls.addWidget(run_button)
    controls.addWidget(refresh_button)

    report_age = QtWidgets.QLabel("Report age: unknown")
    name(report_age, "smoke_report_age_label", "Smoke Report Age")

    output = QtWidgets.QTableWidget()
    name(output, "smoke_output", "Smoke Output")
    output.setColumnCount(len(SMOKE_TABLE_COLUMNS))
    output.setHorizontalHeaderLabels(SMOKE_TABLE_COLUMNS)
    output.setSortingEnabled(True)
    output.setAlternatingRowColors(True)
    output.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
    output.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
    output.verticalHeader().setVisible(False)
    output.horizontalHeader().setStretchLastSection(True)
    output.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
    output.horizontalHeader().setSectionsClickable(True)

    output_details = QtWidgets.QPlainTextEdit()
    name(output_details, "smoke_details_output", "Smoke Details Output")
    output_details.setReadOnly(True)
    output_details.setMaximumHeight(100)
    output_details.setPlainText("Run a smoke check to populate this panel.")

    def set_smoke_output(text: str) -> None:
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            output.setSortingEnabled(False)
            output.setRowCount(0)
            output.setSortingEnabled(True)
            report_age.setText("Report age: unknown")
            output_details.setPlainText(text.strip() or "No output.")
            return

        report_age.setText(_format_report_age(payload))
        output.setSortingEnabled(False)
        rows = _smoke_report_rows(payload)
        output.setRowCount(len(rows))
        gui_background = QtGui.QColor("#dcfce7")
        gui_foreground = QtGui.QColor("#14532d")
        for row_index, row in enumerate(rows):
            values = row["values"]
            assert isinstance(values, list)
            for column_index, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(str(value))
                item.setData(QtCore.Qt.ItemDataRole.UserRole, str(value).lower())
                if row["gui_runnable"]:
                    item.setBackground(gui_background)
                    item.setForeground(gui_foreground)
                output.setItem(row_index, column_index, item)
        output.setSortingEnabled(True)
        output.sortItems(0, QtCore.Qt.SortOrder.AscendingOrder)
        if rows:
            output_details.setPlainText(
                "Green rows are available in the Synthesis tab."
                + (f"\n\nstderr:\n{payload['stderr']}" if payload.get("stderr") else "")
            )
        else:
            output_details.setPlainText(json.dumps(payload, indent=2))

    set_smoke_output(load_latest_report_func())

    smoke_layout.addLayout(controls)
    smoke_layout.addWidget(report_age)
    smoke_layout.addWidget(output, 1)
    smoke_layout.addWidget(output_details)

    def run_selected() -> None:
        run_button.setEnabled(False)
        window.statusBar().showMessage("Running metadata smoke")
        set_smoke_output("_Running smoke check..._")
        QtCore.QTimer.singleShot(10, finish_run)

    def finish_run() -> None:
        candidate_id = candidate_box.currentData()
        set_smoke_output(run_smoke_func(candidate_id))
        run_button.setEnabled(True)
        window.statusBar().showMessage("Metadata smoke complete")

    def refresh() -> None:
        set_smoke_output(load_latest_report_func())
        window.statusBar().showMessage("Smoke report loaded")

    run_button.clicked.connect(run_selected)
    refresh_button.clicked.connect(refresh)

    voxcpm_tab = QtWidgets.QWidget()
    voxcpm_layout = QtWidgets.QVBoxLayout(voxcpm_tab)
    form = QtWidgets.QFormLayout()
    synthesis_candidate = QtWidgets.QComboBox()
    name(synthesis_candidate, "synthesis_candidate_box", "Synthesis Candidate Selector")
    for candidate in sorted(candidates, key=lambda item: item.name.lower()):
        if candidate.id in GUI_SYNTHESIS_CANDIDATE_IDS:
            synthesis_candidate.addItem(f"P{candidate.priority} - {candidate.name}", candidate.id)
    voxcpm_index = synthesis_candidate.findData("voxcpm2")
    if voxcpm_index >= 0:
        synthesis_candidate.setCurrentIndex(voxcpm_index)
    saved_candidate_id = gui_settings.get("selected_synthesis_candidate_id")
    if isinstance(saved_candidate_id, str):
        saved_index = synthesis_candidate.findData(saved_candidate_id)
        if saved_index >= 0:
            synthesis_candidate.setCurrentIndex(saved_index)

    stream_text = QtWidgets.QPlainTextEdit()
    name(stream_text, "stream_text", "Streaming Text")
    help_text(
        stream_text,
        "Text to synthesize. Longer text is useful for stability checks, but it increases generation time "
        "and can expose tail-end artifacts in some models.",
    )
    voice_combo = QtWidgets.QComboBox()
    name(voice_combo, "voice_reference_box", "Voice Reference Selector")
    help_text(
        voice_combo,
        "Reference voice to clone. Cleaner, dry speech usually improves similarity and reduces noise or room echo in output.",
    )
    voice_combo.addItem("Custom reference WAV", "")
    for voice_name, voice_path in load_voice_references_func():
        voice_combo.addItem(voice_name, voice_path)
    reference_row = QtWidgets.QHBoxLayout()
    reference_path = QtWidgets.QLineEdit()
    name(reference_path, "reference_path", "Reference WAV Path")
    help_text(
        reference_path,
        "Path to the reference audio used for cloning. Short, clear WAV samples tend to give better voice transfer.",
    )
    reference_browse = QtWidgets.QPushButton("Browse")
    name(reference_browse, "reference_browse_button", "Browse Reference WAV")
    help_text(reference_browse, "Choose a WAV/FLAC/MP3/M4A file to use as the cloning reference.")
    reference_row.addWidget(reference_path, 1)
    reference_row.addWidget(reference_browse)
    saved_reference_wav_path = str(gui_settings.get("selected_reference_wav_path", "")).strip()
    if saved_reference_wav_path:
        saved_voice_index = voice_combo.findData(saved_reference_wav_path)
        if saved_voice_index >= 0:
            voice_combo.setCurrentIndex(saved_voice_index)
        else:
            voice_combo.setCurrentIndex(0)
        reference_path.setText(saved_reference_wav_path)

    output_row = QtWidgets.QHBoxLayout()
    output_path = QtWidgets.QLineEdit()
    name(output_path, "stream_output_path", "Streaming Output WAV Path")
    help_text(output_path, "Where the generated audio will be saved. A JSON sidecar with settings is written next to it.")
    output_browse = QtWidgets.QPushButton("Browse")
    name(output_browse, "output_browse_button", "Browse Streaming Output WAV")
    help_text(output_browse, "Choose the output audio file path for this generation.")
    output_row.addWidget(output_path, 1)
    output_row.addWidget(output_browse)

    play_audio = QtWidgets.QCheckBox("Stream to audio device")
    name(play_audio, "play_audio_checkbox", "Stream to audio device")
    help_text(
        play_audio,
        "Play the generated audio through the selected device. Turn this off for silent benchmark runs.",
    )
    play_audio.setChecked(True)
    play_audio.setChecked(bool(gui_settings.get("play_audio", True)))
    normalize_audio = QtWidgets.QCheckBox("Normalize output level")
    name(normalize_audio, "normalize_audio_checkbox", "Normalize Output Level")
    help_text(
        normalize_audio,
        "Apply simple peak normalization to saved and played audio. Useful for level matching, but leave off when judging raw model output.",
    )
    normalize_audio.setChecked(bool(gui_settings.get("audio_normalization", False)))
    audio_target_peak = QtWidgets.QDoubleSpinBox()
    name(audio_target_peak, "audio_target_peak_box", "Audio Target Peak")
    help_text(
        audio_target_peak,
        "Peak level used when output normalization is enabled. Higher is louder but leaves less headroom.",
    )
    audio_target_peak.setRange(0.1, 0.98)
    audio_target_peak.setSingleStep(0.05)
    audio_target_peak.setDecimals(2)
    audio_target_peak.setValue(float(gui_settings.get("audio_target_peak", 0.85)))
    audio_device = QtWidgets.QComboBox()
    name(audio_device, "audio_device_box", "Audio Device Selector")
    help_text(audio_device, "Output device for playback. The default device follows the Windows sound settings.")
    audio_device.addItem("Default output device", None)
    for device in load_audio_devices_func():
        audio_device.addItem(device, device)
    saved_audio_device = gui_settings.get("audio_device")
    if isinstance(saved_audio_device, str):
        saved_audio_device_index = audio_device.findData(saved_audio_device)
        if saved_audio_device_index >= 0:
            audio_device.setCurrentIndex(saved_audio_device_index)
    playback_prebuffer = QtWidgets.QDoubleSpinBox()
    name(playback_prebuffer, "playback_prebuffer_box", "Playback Prebuffer Seconds")
    help_text(
        playback_prebuffer,
        "Seconds of live audio to collect before starting playback. More prebuffer reduces stutter but delays first sound.",
    )
    playback_prebuffer.setRange(0.0, 2.0)
    playback_prebuffer.setSingleStep(0.05)
    playback_prebuffer.setDecimals(2)
    playback_prebuffer.setValue(float(gui_settings.get("playback_prebuffer_s", 0.45)))
    playback_mode = QtWidgets.QComboBox()
    name(playback_mode, "playback_mode_box", "Playback Mode")
    playback_mode.addItem("Smooth - play after generation", "after_generation")
    playback_mode.addItem("Live stream - lowest latency", "live")
    playback_mode.setItemData(
        0,
        "Wait for the generated audio first, then play it. Highest reliability, but not true live streaming.",
        QtCore.Qt.ItemDataRole.ToolTipRole,
    )
    playback_mode.setItemData(
        1,
        "Play chunks as VoxCPM2 emits them. Lowest start latency, but it can stutter if generation falls behind realtime.",
        QtCore.Qt.ItemDataRole.ToolTipRole,
    )
    help_text(
        playback_mode,
        "Playback strategy. Live mode applies to VoxCPM2 streaming; file-based backends still play after generation.",
    )
    saved_playback_mode = str(gui_settings.get("playback_mode", "after_generation"))
    saved_playback_mode_index = playback_mode.findData(saved_playback_mode)
    if saved_playback_mode_index >= 0:
        playback_mode.setCurrentIndex(saved_playback_mode_index)
    audio_latency = QtWidgets.QComboBox()
    name(audio_latency, "audio_latency_box", "Audio Latency Mode")
    audio_latency.addItem("High - smoothest", "high")
    audio_latency.addItem("Low - faster start", "low")
    audio_latency.addItem("Default", None)
    saved_audio_latency = gui_settings.get("audio_latency", "high")
    saved_audio_latency_index = audio_latency.findData(saved_audio_latency)
    if saved_audio_latency_index >= 0:
        audio_latency.setCurrentIndex(saved_audio_latency_index)
    audio_latency.setItemData(
        0,
        "Larger device buffer. Best for avoiding dropouts during live playback.",
        QtCore.Qt.ItemDataRole.ToolTipRole,
    )
    audio_latency.setItemData(
        1,
        "Smaller device buffer. Starts faster, but is easier to underrun.",
        QtCore.Qt.ItemDataRole.ToolTipRole,
    )
    audio_latency.setItemData(
        2,
        "Let sounddevice/PortAudio choose the device default latency.",
        QtCore.Qt.ItemDataRole.ToolTipRole,
    )
    help_text(audio_latency, "Audio-device latency hint. High is the safer setting for clean streaming playback.")

    language_code = QtWidgets.QComboBox()
    name(language_code, "language_code_box", "Language Selector")
    help_text(language_code, "Language hint passed to backends that use one. VoxCPM2 mainly follows the text prompt.")
    language_code.addItem("English", ("en", "English"))
    language_code.addItem("European Portuguese", ("pt-PT", "Portuguese"))
    saved_language_code = str(gui_settings.get("language_code", "en"))
    for index in range(language_code.count()):
        item_language_code, _item_language_hint = language_code.itemData(index)
        if item_language_code == saved_language_code:
            language_code.setCurrentIndex(index)
            break

    parameter_box = QtWidgets.QGroupBox("Model Controls")
    name(parameter_box, "generation_parameter_box", "Model Controls")
    parameter_layout = QtWidgets.QFormLayout(parameter_box)
    parameter_widgets: dict[str, QtWidgets.QWidget] = {}
    text_control_box = QtWidgets.QGroupBox("Prompt Controls")
    name(text_control_box, "text_control_box", "Prompt Controls")
    text_control_layout = QtWidgets.QFormLayout(text_control_box)
    text_control_widgets: dict[str, QtWidgets.QPlainTextEdit] = {}
    loading_parameters = False
    last_auto_prompt = ""
    last_auto_output_path = ""

    def clear_parameter_widgets() -> None:
        while parameter_layout.rowCount():
            parameter_layout.removeRow(0)
        parameter_widgets.clear()

    def clear_text_control_widgets() -> None:
        while text_control_layout.rowCount():
            text_control_layout.removeRow(0)
        text_control_widgets.clear()

    def current_candidate_id() -> str:
        return str(synthesis_candidate.currentData())

    def current_model_name() -> str:
        candidate = candidate_by_id.get(current_candidate_id())
        return candidate.name if candidate is not None else str(synthesis_candidate.currentText())

    def current_voice_name() -> str:
        return _display_voice_name(voice_combo.currentText(), reference_path.text().strip())

    def refresh_sample_defaults(*, force: bool = False) -> None:
        nonlocal last_auto_prompt, last_auto_output_path
        model_name = current_model_name()
        voice_name = current_voice_name()
        prompt = _default_synthesis_prompt(model_name, voice_name)
        output = str(_default_synthesis_output_path(model_name, voice_name))
        current_prompt = stream_text.toPlainText()
        current_output = output_path.text().strip()
        if force or not current_prompt.strip() or current_prompt == last_auto_prompt:
            stream_text.setPlainText(prompt)
            last_auto_prompt = prompt
        if force or not current_output or current_output == last_auto_output_path:
            output_path.setText(output)
            last_auto_output_path = output

    def stored_generation_options(candidate_id: str) -> dict:
        by_candidate = gui_settings.get("generation_parameters", {})
        if not isinstance(by_candidate, dict):
            return {}
        options = by_candidate.get(candidate_id, {})
        return options if isinstance(options, dict) else {}

    def stored_text_control_options(candidate_id: str) -> dict[str, str]:
        by_candidate = gui_settings.get("text_controls", {})
        if not isinstance(by_candidate, dict):
            return {}
        options = by_candidate.get(candidate_id, {})
        if not isinstance(options, dict):
            return {}
        return {str(key): str(value) for key, value in options.items()}

    def write_generation_settings() -> None:
        if loading_parameters:
            return
        candidate_id = current_candidate_id()
        gui_settings["selected_synthesis_candidate_id"] = candidate_id
        by_candidate = gui_settings.setdefault("generation_parameters", {})
        if isinstance(by_candidate, dict):
            by_candidate[candidate_id] = selected_generation_options()
        text_by_candidate = gui_settings.setdefault("text_controls", {})
        if isinstance(text_by_candidate, dict):
            text_by_candidate[candidate_id] = selected_text_control_options()
        _save_gui_settings(gui_settings, settings_path)

    def add_parameter_widget(parameter: dict, stored_options: dict) -> None:
        parameter_id = str(parameter["id"])
        parameter_type = parameter.get("type", "float")
        label = str(parameter.get("label", parameter_id))
        value = stored_options.get(parameter_id, parameter.get("default"))
        if parameter_type == "bool":
            widget = QtWidgets.QCheckBox()
            widget.setChecked(bool(value))
            widget.stateChanged.connect(lambda _state: write_generation_settings())
        elif parameter_type == "int":
            widget = QtWidgets.QSpinBox()
            widget.setRange(int(parameter.get("min", 0)), int(parameter.get("max", 999999)))
            widget.setSingleStep(int(parameter.get("step", 1)))
            widget.setValue(int(value))
            widget.valueChanged.connect(lambda _value: write_generation_settings())
        elif parameter_type == "choice":
            widget = QtWidgets.QComboBox()
            for choice in parameter.get("choices", []):
                widget.addItem(str(choice), choice)
            default_index = widget.findData(value)
            if default_index >= 0:
                widget.setCurrentIndex(default_index)
            widget.currentIndexChanged.connect(lambda _index: write_generation_settings())
        else:
            widget = QtWidgets.QDoubleSpinBox()
            widget.setRange(float(parameter.get("min", 0.0)), float(parameter.get("max", 100.0)))
            widget.setSingleStep(float(parameter.get("step", 0.1)))
            widget.setDecimals(3)
            widget.setValue(float(value))
            widget.valueChanged.connect(lambda _value: write_generation_settings())
        name(widget, f"generation_parameter_{parameter_id}", label)
        tooltip = str(parameter.get("description", label))
        help_text(widget, tooltip)
        parameter_widgets[parameter_id] = widget
        parameter_layout.addRow(label, widget)

    def add_text_control_widget(control: dict, stored_options: dict[str, str]) -> None:
        control_id = str(control["id"])
        label = str(control.get("label", control_id))
        widget = QtWidgets.QPlainTextEdit()
        widget.setFixedHeight(max(48, int(control.get("lines", 2)) * 26))
        widget.setPlaceholderText(str(control.get("placeholder", "")))
        widget.setPlainText(str(stored_options.get(control_id, control.get("default", ""))))
        widget.textChanged.connect(write_generation_settings)
        name(widget, f"text_control_{control_id}", label)
        help_text(widget, str(control.get("description", label)))
        text_control_widgets[control_id] = widget
        text_control_layout.addRow(label, widget)

    def add_disabled_text_control_placeholder() -> None:
        widget = QtWidgets.QPlainTextEdit()
        widget.setFixedHeight(52)
        widget.setEnabled(False)
        widget.setPlaceholderText("No separate prompt control is supported by the selected model.")
        widget.setPlainText("")
        name(widget, "text_control_control_instruction", "Control Instruction")
        help_text(widget, "This model does not advertise a separate prompt/style control in candidates.json.")
        text_control_layout.addRow("Control Instruction", widget)

    def selected_generation_options() -> dict[str, object]:
        options: dict[str, object] = {}
        for parameter_id, widget in parameter_widgets.items():
            if isinstance(widget, QtWidgets.QCheckBox):
                options[parameter_id] = widget.isChecked()
            elif isinstance(widget, QtWidgets.QSpinBox):
                options[parameter_id] = widget.value()
            elif isinstance(widget, QtWidgets.QDoubleSpinBox):
                options[parameter_id] = widget.value()
            elif isinstance(widget, QtWidgets.QComboBox):
                options[parameter_id] = widget.currentData()
        return options

    def selected_text_control_options() -> dict[str, str]:
        return {
            control_id: widget.toPlainText().strip()
            for control_id, widget in text_control_widgets.items()
            if widget.toPlainText().strip()
        }

    def refresh_generation_parameters() -> None:
        nonlocal loading_parameters
        loading_parameters = True
        clear_parameter_widgets()
        clear_text_control_widgets()
        candidate_id = current_candidate_id()
        gui_settings["selected_synthesis_candidate_id"] = candidate_id
        refresh_sample_defaults()
        candidate = candidate_by_id.get(candidate_id)
        if candidate is None or not candidate.text_controls:
            add_disabled_text_control_placeholder()
        else:
            stored_text_options = stored_text_control_options(candidate_id)
            for control in candidate.text_controls:
                add_text_control_widget(control, stored_text_options)
        if candidate is None or not candidate.generation_parameters:
            empty_label = QtWidgets.QLabel("No exposed controls for this backend yet.")
            name(empty_label, "generation_parameter_empty", "No Model Controls")
            parameter_layout.addRow(empty_label)
            loading_parameters = False
            _save_gui_settings(gui_settings, settings_path)
            return
        stored_options = stored_generation_options(candidate_id)
        for parameter in candidate.generation_parameters:
            add_parameter_widget(parameter, stored_options)
        loading_parameters = False
        write_generation_settings()

    form.addRow("Model", synthesis_candidate)
    form.addRow("Text", stream_text)
    form.addRow("Language", language_code)
    form.addRow("Voice", voice_combo)
    form.addRow("Reference WAV", reference_row)
    form.addRow("Output File", output_row)
    form.addRow("", play_audio)
    form.addRow("", normalize_audio)
    form.addRow("Target Peak", audio_target_peak)
    form.addRow("Audio Device", audio_device)
    form.addRow("Playback Prebuffer", playback_prebuffer)
    form.addRow("Playback Mode", playback_mode)
    form.addRow("Audio Latency", audio_latency)

    stream_controls = QtWidgets.QHBoxLayout()
    stream_button = QtWidgets.QPushButton()
    name(stream_button, "run_stream_button", "Run VoxCPM2 Streaming")
    stream_button.setToolTip("Run VoxCPM2 streaming generation")
    stream_button.setIcon(window.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_MediaPlay))
    stream_button.setIconSize(QtCore.QSize(32, 32))
    stream_button.setMinimumSize(52, 44)
    load_stream_report = QtWidgets.QPushButton("Load Streaming Report")
    name(load_stream_report, "load_stream_report_button", "Load Streaming Report")
    load_latency_history = QtWidgets.QPushButton("Load Latency History")
    name(load_latency_history, "load_latency_history_button", "Load Latency History")
    stream_controls.addWidget(stream_button)
    stream_controls.addWidget(load_stream_report)
    stream_controls.addWidget(load_latency_history)

    stream_output = QtWidgets.QPlainTextEdit()
    name(stream_output, "stream_output", "Streaming Output")
    stream_output.setReadOnly(True)
    stream_output.setPlainText("Run VoxCPM2 streaming to populate this panel.")
    voxcpm_layout.addLayout(form)
    voxcpm_layout.addWidget(text_control_box)
    voxcpm_layout.addWidget(parameter_box)
    voxcpm_layout.addLayout(stream_controls)
    voxcpm_layout.addWidget(stream_output, 1)
    tabs.addTab(voxcpm_tab, "Synthesis")
    tabs.addTab(smoke_tab, "Diagnostics")

    class SynthesisUiBridge(QtCore.QObject):
        def __init__(
            self,
            thread: QtCore.QThread,
        ) -> None:
            super().__init__(window)
            self._thread = thread

        @QtCore.Slot(str, str)
        def on_finished(self, done_candidate_id: str, result: str) -> None:
            stream_output.setPlainText(result)
            stream_button.setEnabled(True)
            window.statusBar().showMessage(f"{candidate_by_id[done_candidate_id].name} complete")
            self._thread.quit()

    def browse_reference() -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            window,
            "Select reference WAV",
            str(PROJECT_ROOT.parent / "OmniChat" / "voices"),
            "Audio files (*.wav *.flac *.mp3 *.m4a)",
        )
        if path:
            voice_combo.setCurrentIndex(0)
            reference_path.setText(path)
            save_synthesis_session_settings()
            refresh_sample_defaults()

    def browse_output() -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            window,
            "Select output file",
            str(DEFAULT_OUTPUT_DIR / "voxcpm2_streaming_gui.wav"),
            "Audio files (*.wav *.mp3)",
        )
        if path:
            output_path.setText(path)

    def select_voice_reference() -> None:
        selected_path = voice_combo.currentData()
        if selected_path:
            reference_path.setText(str(selected_path))
            window.statusBar().showMessage(f"Voice selected: {voice_combo.currentText()}")
        save_synthesis_session_settings()
        refresh_sample_defaults()

    def save_synthesis_session_settings() -> None:
        selected_language_code, _selected_language_hint = language_code.currentData()
        gui_settings["selected_reference_wav_path"] = reference_path.text().strip()
        gui_settings["language_code"] = selected_language_code
        gui_settings["play_audio"] = play_audio.isChecked()
        gui_settings["audio_device"] = audio_device.currentData()
        gui_settings["playback_prebuffer_s"] = playback_prebuffer.value()
        gui_settings["audio_latency"] = audio_latency.currentData()
        _save_gui_settings(gui_settings, settings_path)

    def run_stream_selected() -> None:
        stream_button.setEnabled(False)
        candidate_id = str(synthesis_candidate.currentData())
        candidate_name = candidate_by_id[candidate_id].name
        window.statusBar().showMessage(f"Running {candidate_name}")
        stream_output.setPlainText(f"Running {candidate_name}...")
        selected_device = audio_device.currentData()
        options = selected_generation_options()
        text_control_options = selected_text_control_options()
        lang_code, lang_hint = language_code.currentData()
        candidate = candidate_by_id[candidate_id]
        spoken_text = stream_text.toPlainText()
        effective_text = _effective_synthesis_text(
            spoken_text,
            candidate.text_controls,
            text_control_options,
        )
        text_option_overrides = _text_control_generation_options(
            candidate.text_controls,
            text_control_options,
        )
        generation_options = options | text_option_overrides

        voxcpm2_kwargs = {
            "candidate_id": candidate_id,
            "candidate_name": candidate_name,
            "voice_name": current_voice_name(),
            "text": effective_text,
            "spoken_text": spoken_text,
            "text_control_options": text_control_options,
            "reference_wav_path": reference_path.text().strip(),
            "output_path": output_path.text().strip(),
            "play_audio": play_audio.isChecked(),
            "audio_device": selected_device,
            "playback_prebuffer_s": playback_prebuffer.value(),
            "audio_latency": audio_latency.currentData(),
            "generation_options": generation_options
            | {
                "audio_normalization": normalize_audio.isChecked(),
                "audio_target_peak": audio_target_peak.value(),
                "playback_mode": playback_mode.currentData(),
            },
        }
        if candidate_id in {"voxcpm2", "voxcpm2_quantized", "voxcpm2_torchao_int8", "voxcpm2_bnb_int8"}:
            QtCore.QTimer.singleShot(10, lambda: finish_voxcpm2_run(voxcpm2_kwargs))
            return

        thread = QtCore.QThread(window)
        worker = SynthesisWorker(
            candidate_id=candidate_id,
            backend_kwargs={
                "candidate_id": candidate_id,
                "candidate_name": candidate_name,
                "voice_name": current_voice_name(),
                "text": effective_text,
                "spoken_text": spoken_text,
                "text_control_options": text_control_options,
                "output_path": output_path.text().strip(),
                "language_code": lang_code,
                "language_hint": lang_hint,
                "reference_wav_path": reference_path.text().strip(),
                "play_audio": play_audio.isChecked(),
                "audio_device": selected_device,
                "playback_prebuffer_s": playback_prebuffer.value(),
                "audio_latency": audio_latency.currentData(),
                "generation_options": generation_options
                | {
                    "audio_normalization": normalize_audio.isChecked(),
                    "audio_target_peak": audio_target_peak.value(),
                },
            },
        )
        worker.moveToThread(thread)
        bridge = SynthesisUiBridge(thread)

        worker.finished.connect(bridge.on_finished, QtCore.Qt.ConnectionType.QueuedConnection)
        worker.finished.connect(worker.deleteLater)
        thread.started.connect(worker.run)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(bridge.deleteLater)
        thread.start()
        window._active_synthesis_thread = thread
        window._active_synthesis_worker = worker
        window._active_synthesis_bridge = bridge

    def finish_voxcpm2_run(voxcpm2_kwargs: dict) -> None:
        try:
            result = run_voxcpm2_streaming_func(**voxcpm2_kwargs)
        except Exception as exc:  # pragma: no cover - exercised through GUI smoke behavior
            result = json.dumps(
                    {
                        "status": "failed",
                        "candidate_id": voxcpm2_kwargs.get("candidate_id", "voxcpm2"),
                        "error": str(exc),
                    },
                indent=2,
            )
        stream_output.setPlainText(result)
        stream_button.setEnabled(True)
        window.statusBar().showMessage(f"{voxcpm2_kwargs.get('candidate_name', 'VoxCPM2')} complete")

    def save_audio_normalization_settings() -> None:
        gui_settings["audio_normalization"] = normalize_audio.isChecked()
        gui_settings["audio_target_peak"] = audio_target_peak.value()
        _save_gui_settings(gui_settings, settings_path)

    def save_playback_mode_settings() -> None:
        gui_settings["playback_mode"] = playback_mode.currentData()
        _save_gui_settings(gui_settings, settings_path)


    def refresh_stream_report() -> None:
        if DEFAULT_REPORT_PATH.exists():
            stream_output.setPlainText(
                json.dumps(json.loads(DEFAULT_REPORT_PATH.read_text(encoding="utf-8")), indent=2)
            )
        else:
            stream_output.setPlainText("No VoxCPM2 streaming report written yet.")
        window.statusBar().showMessage("Streaming report loaded")

    def refresh_latency_history() -> None:
        if not DEFAULT_HISTORY_PATH.exists():
            stream_output.setPlainText("No VoxCPM2 streaming history written yet.")
            return
        rows = []
        for line in DEFAULT_HISTORY_PATH.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            rows.append(
                {
                    "first_chunk_latency_s": item["first_chunk_latency_s"],
                    "total_elapsed_s": item["total_elapsed_s"],
                    "audio_duration_s": item["audio_duration_s"],
                    "realtime_factor": item["realtime_factor"],
                    "chunk_count": item["chunk_count"],
                    "reference_wav_path": item["reference_wav_path"],
                    "output_path": item["output_path"],
                }
            )
        stream_output.setPlainText(json.dumps(rows[-20:], indent=2))
        window.statusBar().showMessage("Latency history loaded")

    reference_browse.clicked.connect(browse_reference)
    voice_combo.currentIndexChanged.connect(select_voice_reference)
    reference_path.editingFinished.connect(save_synthesis_session_settings)
    language_code.currentIndexChanged.connect(lambda _index: save_synthesis_session_settings())
    play_audio.toggled.connect(lambda _enabled: save_synthesis_session_settings())
    audio_device.currentIndexChanged.connect(lambda _index: save_synthesis_session_settings())
    playback_prebuffer.valueChanged.connect(lambda _value: save_synthesis_session_settings())
    audio_latency.currentIndexChanged.connect(lambda _index: save_synthesis_session_settings())
    output_browse.clicked.connect(browse_output)
    synthesis_candidate.currentIndexChanged.connect(refresh_generation_parameters)
    normalize_audio.toggled.connect(lambda _enabled: save_audio_normalization_settings())
    audio_target_peak.valueChanged.connect(lambda _value: save_audio_normalization_settings())
    playback_mode.currentIndexChanged.connect(lambda _index: save_playback_mode_settings())
    stream_button.clicked.connect(run_stream_selected)
    load_stream_report.clicked.connect(refresh_stream_report)
    load_latency_history.clicked.connect(refresh_latency_history)

    window._tabs = tabs
    window._synthesis_tab_index = 0
    window._diagnostics_tab_index = 1
    window._candidate_box = candidate_box
    window._smoke_output = output
    window._smoke_details_output = output_details
    window._smoke_report_age_label = report_age
    window._run_smoke_button = run_button
    window._stream_text = stream_text
    window._synthesis_candidate_box = synthesis_candidate
    window._language_code_box = language_code
    window._generation_parameter_box = parameter_box
    window._generation_parameter_widgets = parameter_widgets
    window._text_control_box = text_control_box
    window._text_control_widgets = text_control_widgets
    window._voice_reference_box = voice_combo
    window._reference_path = reference_path
    window._stream_output_path = output_path
    window._normalize_audio_checkbox = normalize_audio
    window._audio_target_peak_box = audio_target_peak
    window._play_audio_checkbox = play_audio
    window._audio_device_box = audio_device
    window._playback_prebuffer_box = playback_prebuffer
    window._playback_mode_box = playback_mode
    window._audio_latency_box = audio_latency
    window._run_stream_button = stream_button
    window._stream_output = stream_output
    window._load_latency_history_button = load_latency_history

    refresh_generation_parameters()

    return window


def _demo_smoke(candidate_id: str | None = None) -> str:
    return json.dumps(
        {
            "demo": True,
            "selected_candidate": candidate_id or "all",
            "status": "metadata smoke path exercised",
            "results": [
                {
                    "candidate_id": "voxcpm2",
                    "name": "VoxCPM2",
                    "repo_id": "openbmb/VoxCPM2",
                    "priority": 1,
                    "metadata_ok": True,
                    "import_checks": [{"module": "torch", "available": True}],
                    "cache_present": True,
                    "status": "ready_for_synthesis_smoke",
                    "details": {
                        "capabilities": ["tts", "voice_clone", "voice_design", "multilingual", "48khz"],
                        "model_scale": "2B",
                    },
                },
                {
                    "candidate_id": "zonos2",
                    "name": "ZONOS2",
                    "repo_id": "Zyphra/ZONOS2",
                    "priority": 3,
                    "metadata_ok": True,
                    "import_checks": [{"module": "torch", "available": True}],
                    "cache_present": False,
                    "status": "needs_model_download",
                    "details": {"capabilities": ["tts"]},
                },
            ],
        },
        indent=2,
    )


def _demo_streaming(
    *,
    candidate_id: str = "voxcpm2",
    candidate_name: str = "VoxCPM2",
    voice_name: str = "Custom Voice",
    text: str,
    reference_wav_path: str,
    output_path: str,
    play_audio: bool,
    audio_device: str | None,
    spoken_text: str | None = None,
    text_control_options: dict[str, str] | None = None,
    playback_prebuffer_s: float = 0.45,
    audio_latency: str | None = "high",
    generation_options: dict[str, object] | None = None,
) -> str:
    output_file = Path(output_path)
    audio_sha256 = _write_demo_wav(output_file, text=text, reference_wav_path=reference_wav_path)
    payload = {
        "demo": True,
        "candidate_id": candidate_id,
        "candidate_name": candidate_name,
        "voice_name": voice_name,
        "text": text,
        "spoken_text": spoken_text or text,
        "effective_text": text,
        "text_control_options": text_control_options or {},
        "reference_wav_path": reference_wav_path,
        "output_path": output_path,
        "output_sha256": audio_sha256,
        "played_to_device": play_audio,
        "playback_mode": (generation_options or {}).get("playback_mode", "after_generation"),
        "audio_device": audio_device,
        "playback_prebuffer_s": playback_prebuffer_s,
        "audio_latency": audio_latency,
        "generation_options": generation_options or {},
        "first_chunk_latency_s": 0.18,
        "time_to_first_output_s": 0.18,
        "total_elapsed_s": 1.42,
        "audio_duration_s": 1.6,
        "realtime_factor": 0.8875,
        "chunk_count": 10,
    }
    sidecar_path = _write_generation_sidecar(
        output_file,
        payload | {"status": "complete", "generated_at_unix": time.time()},
    )
    payload["metadata_path"] = str(sidecar_path)
    DEFAULT_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_REPORT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with DEFAULT_HISTORY_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")
    return json.dumps(payload, indent=2)


def _write_demo_wav(output_path: Path, *, text: str, reference_wav_path: str) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seed_material = f"{Path(reference_wav_path).name}|{text}".encode("utf-8", errors="replace")
    digest = hashlib.sha256(seed_material).digest()
    base_freq = 220 + digest[0]
    mod_freq = 330 + digest[1]
    sample_rate = 16000
    duration_s = 0.4
    amplitude = 0.22
    frames = bytearray()
    for index in range(int(sample_rate * duration_s)):
        t = index / sample_rate
        value = amplitude * (
            math.sin(2 * math.pi * base_freq * t)
            + 0.35 * math.sin(2 * math.pi * mod_freq * t)
        )
        sample = max(-32768, min(32767, int(value * 32767)))
        frames.extend(sample.to_bytes(2, byteorder="little", signed=True))

    with wave.open(str(output_path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(bytes(frames))

    return hashlib.sha256(output_path.read_bytes()).hexdigest()


def run_visible_demo(window, *, exit_after_ms: int | None = None) -> None:
    from PySide6 import QtCore, QtWidgets

    def find(name: str):
        widget = window.findChild(QtWidgets.QWidget, name)
        if widget is None:
            raise RuntimeError(f"Missing demo widget: {name}")
        return widget

    tabs = find("main_tabs")
    smoke_button = find("run_smoke_button")
    stream_button = find("run_stream_button")
    stream_text = find("stream_text")
    reference_path = find("reference_path")
    output_path = find("stream_output_path")
    play_audio = find("play_audio_checkbox")

    def step_smoke() -> None:
        tabs.setCurrentIndex(window._diagnostics_tab_index)
        smoke_button.click()

    def step_stream() -> None:
        tabs.setCurrentIndex(window._synthesis_tab_index)
        stream_text.setPlainText(
            "Demo mode: VoxCPM2 streams chunks to the selected device while also "
            "saving WAV and latency artifacts for comparison."
        )
        reference_path.setText(str(PROJECT_ROOT.parent / "OmniChat" / "voices" / "pegasus.wav"))
        output_path.setText(str(DEFAULT_OUTPUT_DIR / "demo_streaming.wav"))
        play_audio.setChecked(True)
        stream_button.click()

    def step_history() -> None:
        tabs.setCurrentIndex(window._synthesis_tab_index)
        find("load_latency_history_button").click()

    QtCore.QTimer.singleShot(500, step_smoke)
    QtCore.QTimer.singleShot(1800, step_stream)
    QtCore.QTimer.singleShot(3400, step_history)
    if exit_after_ms is not None:
        QtCore.QTimer.singleShot(exit_after_ms, QtWidgets.QApplication.instance().quit)


def main() -> int:
    try:
        from PySide6 import QtWidgets
    except ImportError:
        print("PySide6 is not installed. Install with: python -m pip install -e .[gui]")
        return 2

    parser = argparse.ArgumentParser(description="AISpeechApp GUI")
    parser.add_argument(
        "--demo-backend",
        action="store_true",
        help="Use deterministic demo backends without automatically driving the GUI.",
    )
    parser.add_argument("--demo", action="store_true", help="Run visible scripted demo mode.")
    parser.add_argument(
        "--demo-exit-ms",
        type=int,
        help="Automatically close demo mode after this many milliseconds.",
    )
    args = parser.parse_args()

    app = QtWidgets.QApplication([sys.argv[0]])
    if args.demo or args.demo_backend:
        window = create_main_window(
            run_smoke_func=_demo_smoke,
            load_audio_devices_func=lambda: ["Demo output device"],
            load_voice_references_func=_load_voice_references,
            run_voxcpm2_streaming_func=_demo_streaming,
        )
        title = "AISpeechApp - Visible Demo Mode" if args.demo else "AISpeechApp - Demo Backend"
        window.setWindowTitle(title)
        window.setAccessibleName(title)
    else:
        window = create_main_window()

    if args.demo:
        run_visible_demo(window, exit_after_ms=args.demo_exit_ms)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
