from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PySide6 import QtTest, QtWidgets

from aispeechapp.gui import create_main_window


def _app() -> QtWidgets.QApplication:
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _find(window: QtWidgets.QWidget, object_name: str):
    widget = window.findChild(QtWidgets.QWidget, object_name)
    assert widget is not None, object_name
    return widget


def _wait_until(predicate, timeout_ms: int = 1000):
    elapsed = 0
    while elapsed < timeout_ms:
        if predicate():
            return
        QtTest.QTest.qWait(25)
        elapsed += 25
    assert predicate()


def test_gui_smoke_button_runs_injected_backend(tmp_path):
    _app()
    calls: list[str | None] = []
    window = create_main_window(
        run_smoke_func=lambda candidate_id: calls.append(candidate_id) or "smoke ok",
        load_audio_devices_func=lambda: [],
        load_voice_references_func=lambda: [],
        settings_path=tmp_path / "settings.local.json",
    )

    assert isinstance(window, QtWidgets.QMainWindow)
    button = _find(window, "run_smoke_button")
    output = _find(window, "smoke_output")
    button.click()
    QtTest.QTest.qWait(30)

    assert calls == [None]
    assert output.toPlainText() == "smoke ok"
    assert button.isEnabled()
    assert window.statusBar().currentMessage() == "Metadata smoke complete"


@dataclass
class StreamCall:
    text: str
    reference_wav_path: str
    output_path: str
    play_audio: bool
    audio_device: str | None
    playback_prebuffer_s: float
    audio_latency: str | None
    generation_options: dict[str, object]
    audio_observer: object | None


def test_gui_voxcpm2_streaming_button_passes_reference_device_and_output(tmp_path):
    _app()
    calls: list[StreamCall] = []

    def fake_stream(**kwargs):
        kwargs["audio_observer"](np.ones(512, dtype=np.float32) * 0.25, 16000)
        calls.append(StreamCall(**kwargs))
        return "stream ok"

    window = create_main_window(
        load_audio_devices_func=lambda: ["7: Test Speakers"],
        load_voice_references_func=lambda: [("Voice One", "C:/refs/voice_one.wav")],
        run_voxcpm2_streaming_func=fake_stream,
        settings_path=tmp_path / "settings.local.json",
    )

    tabs = _find(window, "main_tabs")
    tabs.setCurrentIndex(1)
    _find(window, "stream_text").setPlainText("hello streaming")
    voice_box = _find(window, "voice_reference_box")
    voice_box.setCurrentIndex(1)
    _find(window, "stream_output_path").setText("C:/out/generated.wav")
    audio_device = _find(window, "audio_device_box")
    audio_device.setCurrentIndex(1)
    _find(window, "playback_prebuffer_box").setValue(0.65)
    latency = _find(window, "audio_latency_box")
    latency.setCurrentIndex(1)
    guidance = _find(window, "generation_parameter_cfg_value")
    guidance.setValue(2.7)
    steps = _find(window, "generation_parameter_inference_timesteps")
    steps.setValue(14)

    button = _find(window, "run_stream_button")
    output = _find(window, "stream_output")
    button.click()
    assert not button.isEnabled()
    _wait_until(lambda: bool(calls) and output.toPlainText() == "stream ok")

    assert calls[0] == StreamCall(
        text="hello streaming",
        reference_wav_path="C:/refs/voice_one.wav",
        output_path="C:/out/generated.wav",
        play_audio=True,
        audio_device="7: Test Speakers",
        playback_prebuffer_s=0.65,
        audio_latency="low",
        generation_options={
            "cfg_value": 2.7,
            "inference_timesteps": 14,
            "normalize": True,
            "denoise": False,
            "retry_badcase": False,
            "retry_badcase_max_times": 3,
            "retry_badcase_ratio_threshold": 6.0,
            "min_len": 2,
            "max_len": 4096,
        },
        audio_observer=calls[0].audio_observer,
    )
    assert callable(calls[0].audio_observer)
    assert output.toPlainText() == "stream ok"
    assert button.isEnabled()
    assert window.statusBar().currentMessage() == "VoxCPM2 complete"
    assert _find(window, "audio_visualizer").sample_count == 512
    assert _find(window, "audio_visualizer").spectrum_peak > 0.0


def test_gui_audio_visualizer_updates_from_audio(tmp_path):
    _app()
    window = create_main_window(
        load_audio_devices_func=lambda: [],
        load_voice_references_func=lambda: [],
        settings_path=tmp_path / "settings.local.json",
    )
    visualizer = _find(window, "audio_visualizer")

    visualizer.append_audio(np.sin(np.linspace(0, np.pi * 8, 2048, dtype=np.float32)), 48000)
    pixmap = visualizer.grab()

    assert visualizer.sample_count == 2048
    assert visualizer.spectrum_peak > 0.0
    assert not pixmap.isNull()


def test_gui_backend_candidate_uses_dynamic_controls(tmp_path):
    _app()
    calls: list[dict] = []

    def fake_backend(**kwargs):
        calls.append(kwargs)
        return "backend ok"

    window = create_main_window(
        load_audio_devices_func=lambda: [],
        load_voice_references_func=lambda: [],
        run_backend_synthesis_func=fake_backend,
        settings_path=tmp_path / "settings.local.json",
    )

    tabs = _find(window, "main_tabs")
    tabs.setCurrentIndex(1)
    candidate_box = _find(window, "synthesis_candidate_box")
    candidate_box.setCurrentIndex(candidate_box.findData("dots_tts_soar"))
    _find(window, "stream_text").setPlainText("hello dots")
    _find(window, "stream_output_path").setText(str(tmp_path / "dots.wav"))
    _find(window, "generation_parameter_num_steps").setValue(12)
    _find(window, "generation_parameter_guidance_scale").setValue(1.4)

    button = _find(window, "run_stream_button")
    output = _find(window, "stream_output")
    button.click()
    _wait_until(lambda: bool(calls) and output.toPlainText() == "backend ok")

    assert calls == [
        {
            "candidate_id": "dots_tts_soar",
            "text": "hello dots",
            "output_path": str(tmp_path / "dots.wav"),
            "language_code": "en",
            "language_hint": "English",
            "generation_options": {"num_steps": 12, "guidance_scale": 1.4},
        }
    ]
    assert output.toPlainText() == "backend ok"
    assert window.statusBar().currentMessage() == "dots.tts soar complete"


def test_gui_generation_controls_persist_per_model(tmp_path):
    _app()
    settings_path = tmp_path / "gui_settings.local.json"

    first_window = create_main_window(
        load_audio_devices_func=lambda: [],
        load_voice_references_func=lambda: [],
        settings_path=settings_path,
    )
    first_candidate_box = _find(first_window, "synthesis_candidate_box")
    first_candidate_box.setCurrentIndex(first_candidate_box.findData("voxcpm2"))
    _find(first_window, "generation_parameter_cfg_value").setValue(2.6)
    _find(first_window, "generation_parameter_inference_timesteps").setValue(16)
    first_window.close()

    second_window = create_main_window(
        load_audio_devices_func=lambda: [],
        load_voice_references_func=lambda: [],
        settings_path=settings_path,
    )
    second_candidate_box = _find(second_window, "synthesis_candidate_box")

    assert second_candidate_box.currentData() == "voxcpm2"
    assert _find(second_window, "generation_parameter_cfg_value").value() == 2.6
    assert _find(second_window, "generation_parameter_inference_timesteps").value() == 16

    second_candidate_box.setCurrentIndex(second_candidate_box.findData("dots_tts_soar"))
    _find(second_window, "generation_parameter_num_steps").setValue(13)
    second_window.close()

    third_window = create_main_window(
        load_audio_devices_func=lambda: [],
        load_voice_references_func=lambda: [],
        settings_path=settings_path,
    )
    third_candidate_box = _find(third_window, "synthesis_candidate_box")
    assert third_candidate_box.currentData() == "dots_tts_soar"
    assert _find(third_window, "generation_parameter_num_steps").value() == 13

    third_candidate_box.setCurrentIndex(third_candidate_box.findData("voxcpm2"))
    assert _find(third_window, "generation_parameter_cfg_value").value() == 2.6
    assert _find(third_window, "generation_parameter_inference_timesteps").value() == 16
