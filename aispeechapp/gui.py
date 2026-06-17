from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import wave
from pathlib import Path

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
    return completed.stdout + completed.stderr


def _load_latest_report() -> str:
    path = PROJECT_ROOT / "reports" / "smoke_metadata.json"
    if not path.exists():
        return "No report written yet."
    return json.dumps(json.loads(path.read_text(encoding="utf-8")), indent=2)


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


def _run_voxcpm2_streaming(
    *,
    text: str,
    reference_wav_path: str,
    output_path: str,
    play_audio: bool,
    audio_device: str | None,
    playback_prebuffer_s: float,
    audio_latency: str | None,
    generation_options: dict[str, object] | None = None,
) -> str:
    options = generation_options or {}
    result = synthesize_voxcpm2_streaming(
        text=text,
        output_path=Path(output_path),
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
    )
    write_streaming_report(result, DEFAULT_REPORT_PATH)
    append_streaming_history(result, DEFAULT_HISTORY_PATH)
    return json.dumps(result.__dict__ | {"chunks": [chunk.__dict__ for chunk in result.chunks]}, indent=2)


def _run_backend_synthesis(
    *,
    candidate_id: str,
    text: str,
    output_path: str,
    language_code: str,
    language_hint: str,
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
    started = __import__("time").perf_counter()
    completed = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    elapsed = round(__import__("time").perf_counter() - started, 3)
    payload = {
        "candidate_id": candidate_id,
        "output_path": output_path,
        "elapsed_s": elapsed,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "generation_options": generation_options or {},
    }
    if completed.returncode != 0:
        payload["status"] = "failed"
    else:
        payload["status"] = "complete"
        payload["output_exists"] = Path(output_path).exists()
    return json.dumps(payload, indent=2)


def create_main_window(
    *,
    run_smoke_func=_run_smoke,
    load_latest_report_func=_load_latest_report,
    load_audio_devices_func=_load_audio_devices,
    load_voice_references_func=_load_voice_references,
    run_voxcpm2_streaming_func=_run_voxcpm2_streaming,
    run_backend_synthesis_func=_run_backend_synthesis,
):
    from PySide6 import QtCore, QtWidgets


    class AISpeechWindow(QtWidgets.QMainWindow):
        pass

    def name(widget: QtWidgets.QWidget, object_name: str, accessible_name: str | None = None):
        widget.setObjectName(object_name)
        widget.setAccessibleName(accessible_name or object_name)
        return widget

    candidates = load_candidates()
    candidate_by_id = {candidate.id: candidate for candidate in candidates}

    window = AISpeechWindow()
    name(window, "main_window", "AISpeechApp Main Window")
    window.setWindowTitle("AISpeechApp - Local TTS Smoke Lab")
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

    output = QtWidgets.QPlainTextEdit()
    name(output, "smoke_output", "Smoke Output")
    output.setReadOnly(True)
    output.setPlainText("Run a smoke check to populate this panel.")

    smoke_layout.addLayout(controls)
    smoke_layout.addWidget(output, 1)
    tabs.addTab(smoke_tab, "Smoke")

    def run_selected() -> None:
        run_button.setEnabled(False)
        window.statusBar().showMessage("Running metadata smoke")
        output.setPlainText("Running smoke check...")
        QtCore.QTimer.singleShot(10, finish_run)

    def finish_run() -> None:
        candidate_id = candidate_box.currentData()
        output.setPlainText(run_smoke_func(candidate_id))
        run_button.setEnabled(True)
        window.statusBar().showMessage("Metadata smoke complete")

    def refresh() -> None:
        output.setPlainText(load_latest_report_func())
        window.statusBar().showMessage("Smoke report loaded")

    run_button.clicked.connect(run_selected)
    refresh_button.clicked.connect(refresh)

    voxcpm_tab = QtWidgets.QWidget()
    voxcpm_layout = QtWidgets.QVBoxLayout(voxcpm_tab)
    form = QtWidgets.QFormLayout()
    synthesis_candidate = QtWidgets.QComboBox()
    name(synthesis_candidate, "synthesis_candidate_box", "Synthesis Candidate Selector")
    for candidate in candidates:
        if candidate.id in {
            "qwen3_tts_17b_customvoice",
            "voxcpm2",
            "dots_tts_soar",
            "dots_tts_mf",
            "indextts2",
            "omnivoice",
            "microsoft_vibevoice_15b",
        }:
            synthesis_candidate.addItem(f"P{candidate.priority} - {candidate.name}", candidate.id)
    voxcpm_index = synthesis_candidate.findData("voxcpm2")
    if voxcpm_index >= 0:
        synthesis_candidate.setCurrentIndex(voxcpm_index)

    stream_text = QtWidgets.QPlainTextEdit()
    name(stream_text, "stream_text", "Streaming Text")
    stream_text.setPlainText(
        "This is VoxCPM2. Today we test low-latency voice cloning with clear diction, "
        "natural pacing, and no unnecessary room echo."
    )
    voice_combo = QtWidgets.QComboBox()
    name(voice_combo, "voice_reference_box", "Voice Reference Selector")
    voice_combo.addItem("Custom reference WAV", "")
    for voice_name, voice_path in load_voice_references_func():
        voice_combo.addItem(voice_name, voice_path)
    reference_row = QtWidgets.QHBoxLayout()
    reference_path = QtWidgets.QLineEdit()
    name(reference_path, "reference_path", "Reference WAV Path")
    reference_browse = QtWidgets.QPushButton("Browse")
    name(reference_browse, "reference_browse_button", "Browse Reference WAV")
    reference_row.addWidget(reference_path, 1)
    reference_row.addWidget(reference_browse)

    output_row = QtWidgets.QHBoxLayout()
    output_path = QtWidgets.QLineEdit(str(DEFAULT_OUTPUT_DIR / "voxcpm2_streaming_gui.wav"))
    name(output_path, "stream_output_path", "Streaming Output WAV Path")
    output_browse = QtWidgets.QPushButton("Browse")
    name(output_browse, "output_browse_button", "Browse Streaming Output WAV")
    output_row.addWidget(output_path, 1)
    output_row.addWidget(output_browse)

    play_audio = QtWidgets.QCheckBox("Stream to audio device")
    name(play_audio, "play_audio_checkbox", "Stream to audio device")
    play_audio.setChecked(True)
    audio_device = QtWidgets.QComboBox()
    name(audio_device, "audio_device_box", "Audio Device Selector")
    audio_device.addItem("Default output device", None)
    for device in load_audio_devices_func():
        audio_device.addItem(device, device)
    playback_prebuffer = QtWidgets.QDoubleSpinBox()
    name(playback_prebuffer, "playback_prebuffer_box", "Playback Prebuffer Seconds")
    playback_prebuffer.setRange(0.0, 2.0)
    playback_prebuffer.setSingleStep(0.05)
    playback_prebuffer.setDecimals(2)
    playback_prebuffer.setValue(0.45)
    audio_latency = QtWidgets.QComboBox()
    name(audio_latency, "audio_latency_box", "Audio Latency Mode")
    audio_latency.addItem("High - smoothest", "high")
    audio_latency.addItem("Low - faster start", "low")
    audio_latency.addItem("Default", None)

    language_code = QtWidgets.QComboBox()
    name(language_code, "language_code_box", "Language Selector")
    language_code.addItem("English", ("en", "English"))
    language_code.addItem("European Portuguese", ("pt-PT", "Portuguese"))

    parameter_box = QtWidgets.QGroupBox("Model Controls")
    name(parameter_box, "generation_parameter_box", "Model Controls")
    parameter_layout = QtWidgets.QFormLayout(parameter_box)
    parameter_widgets: dict[str, QtWidgets.QWidget] = {}

    def clear_parameter_widgets() -> None:
        while parameter_layout.rowCount():
            parameter_layout.removeRow(0)
        parameter_widgets.clear()

    def add_parameter_widget(parameter: dict) -> None:
        parameter_id = str(parameter["id"])
        parameter_type = parameter.get("type", "float")
        label = str(parameter.get("label", parameter_id))
        if parameter_type == "bool":
            widget = QtWidgets.QCheckBox()
            widget.setChecked(bool(parameter.get("default", False)))
        elif parameter_type == "int":
            widget = QtWidgets.QSpinBox()
            widget.setRange(int(parameter.get("min", 0)), int(parameter.get("max", 999999)))
            widget.setSingleStep(int(parameter.get("step", 1)))
            widget.setValue(int(parameter.get("default", 0)))
        elif parameter_type == "choice":
            widget = QtWidgets.QComboBox()
            for choice in parameter.get("choices", []):
                widget.addItem(str(choice), choice)
            default_index = widget.findData(parameter.get("default"))
            if default_index >= 0:
                widget.setCurrentIndex(default_index)
        else:
            widget = QtWidgets.QDoubleSpinBox()
            widget.setRange(float(parameter.get("min", 0.0)), float(parameter.get("max", 100.0)))
            widget.setSingleStep(float(parameter.get("step", 0.1)))
            widget.setDecimals(3)
            widget.setValue(float(parameter.get("default", 0.0)))
        name(widget, f"generation_parameter_{parameter_id}", label)
        widget.setToolTip(str(parameter.get("description", label)))
        parameter_widgets[parameter_id] = widget
        parameter_layout.addRow(label, widget)

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

    def refresh_generation_parameters() -> None:
        clear_parameter_widgets()
        candidate = candidate_by_id.get(str(synthesis_candidate.currentData()))
        if candidate is None or not candidate.generation_parameters:
            empty_label = QtWidgets.QLabel("No exposed controls for this backend yet.")
            name(empty_label, "generation_parameter_empty", "No Model Controls")
            parameter_layout.addRow(empty_label)
            return
        for parameter in candidate.generation_parameters:
            add_parameter_widget(parameter)

    form.addRow("Model", synthesis_candidate)
    form.addRow("Text", stream_text)
    form.addRow("Language", language_code)
    form.addRow("Voice", voice_combo)
    form.addRow("Reference WAV", reference_row)
    form.addRow("Output File", output_row)
    form.addRow("", play_audio)
    form.addRow("Audio Device", audio_device)
    form.addRow("Playback Prebuffer", playback_prebuffer)
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
    voxcpm_layout.addWidget(parameter_box)
    voxcpm_layout.addLayout(stream_controls)
    voxcpm_layout.addWidget(stream_output, 1)
    tabs.addTab(voxcpm_tab, "Synthesis")

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

    def run_stream_selected() -> None:
        stream_button.setEnabled(False)
        candidate_id = str(synthesis_candidate.currentData())
        candidate_name = candidate_by_id[candidate_id].name
        window.statusBar().showMessage(f"Running {candidate_name}")
        stream_output.setPlainText(f"Running {candidate_name}...")
        QtCore.QTimer.singleShot(10, finish_stream_run)

    def finish_stream_run() -> None:
        selected_device = audio_device.currentData()
        candidate_id = str(synthesis_candidate.currentData())
        options = selected_generation_options()
        lang_code, lang_hint = language_code.currentData()
        if candidate_id == "voxcpm2":
            stream_output.setPlainText(
                run_voxcpm2_streaming_func(
                    text=stream_text.toPlainText(),
                    reference_wav_path=reference_path.text().strip(),
                    output_path=output_path.text().strip(),
                    play_audio=play_audio.isChecked(),
                    audio_device=selected_device,
                    playback_prebuffer_s=playback_prebuffer.value(),
                    audio_latency=audio_latency.currentData(),
                    generation_options=options,
                )
            )
        else:
            stream_output.setPlainText(
                run_backend_synthesis_func(
                    candidate_id=candidate_id,
                    text=stream_text.toPlainText(),
                    output_path=output_path.text().strip(),
                    language_code=lang_code,
                    language_hint=lang_hint,
                    generation_options=options,
                )
            )
        stream_button.setEnabled(True)
        window.statusBar().showMessage(f"{candidate_by_id[candidate_id].name} complete")

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
    output_browse.clicked.connect(browse_output)
    synthesis_candidate.currentIndexChanged.connect(refresh_generation_parameters)
    stream_button.clicked.connect(run_stream_selected)
    load_stream_report.clicked.connect(refresh_stream_report)
    load_latency_history.clicked.connect(refresh_latency_history)

    window._tabs = tabs
    window._candidate_box = candidate_box
    window._smoke_output = output
    window._run_smoke_button = run_button
    window._stream_text = stream_text
    window._synthesis_candidate_box = synthesis_candidate
    window._language_code_box = language_code
    window._generation_parameter_box = parameter_box
    window._generation_parameter_widgets = parameter_widgets
    window._voice_reference_box = voice_combo
    window._reference_path = reference_path
    window._stream_output_path = output_path
    window._play_audio_checkbox = play_audio
    window._audio_device_box = audio_device
    window._playback_prebuffer_box = playback_prebuffer
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
        },
        indent=2,
    )


def _demo_streaming(
    *,
    text: str,
    reference_wav_path: str,
    output_path: str,
    play_audio: bool,
    audio_device: str | None,
    playback_prebuffer_s: float = 0.45,
    audio_latency: str | None = "high",
    generation_options: dict[str, object] | None = None,
) -> str:
    output_file = Path(output_path)
    audio_sha256 = _write_demo_wav(output_file, text=text, reference_wav_path=reference_wav_path)
    payload = {
        "demo": True,
        "text": text,
        "reference_wav_path": reference_wav_path,
        "output_path": output_path,
        "output_sha256": audio_sha256,
        "played_to_device": play_audio,
        "audio_device": audio_device,
        "playback_prebuffer_s": playback_prebuffer_s,
        "audio_latency": audio_latency,
        "generation_options": generation_options or {},
        "first_chunk_latency_s": 0.18,
        "total_elapsed_s": 1.42,
        "audio_duration_s": 1.6,
        "realtime_factor": 0.8875,
        "chunk_count": 10,
    }
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
        tabs.setCurrentIndex(0)
        smoke_button.click()

    def step_stream() -> None:
        tabs.setCurrentIndex(1)
        stream_text.setPlainText(
            "Demo mode: VoxCPM2 streams chunks to the selected device while also "
            "saving WAV and latency artifacts for comparison."
        )
        reference_path.setText(str(PROJECT_ROOT.parent / "OmniChat" / "voices" / "pegasus.wav"))
        output_path.setText(str(DEFAULT_OUTPUT_DIR / "demo_streaming.wav"))
        play_audio.setChecked(True)
        stream_button.click()

    def step_history() -> None:
        tabs.setCurrentIndex(1)
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
