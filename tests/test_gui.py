from __future__ import annotations

from dataclasses import dataclass

from PySide6 import QtTest, QtWidgets

from aispeechapp.gui import create_main_window


def _app() -> QtWidgets.QApplication:
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _find(window: QtWidgets.QWidget, object_name: str):
    widget = window.findChild(QtWidgets.QWidget, object_name)
    assert widget is not None, object_name
    return widget


def test_gui_smoke_button_runs_injected_backend():
    _app()
    calls: list[str | None] = []
    window = create_main_window(
        run_smoke_func=lambda candidate_id: calls.append(candidate_id) or "smoke ok",
        load_audio_devices_func=lambda: [],
        load_voice_references_func=lambda: [],
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


def test_gui_voxcpm2_streaming_button_passes_reference_device_and_output():
    _app()
    calls: list[StreamCall] = []

    def fake_stream(**kwargs):
        calls.append(StreamCall(**kwargs))
        return "stream ok"

    window = create_main_window(
        load_audio_devices_func=lambda: ["7: Test Speakers"],
        load_voice_references_func=lambda: [("Voice One", "C:/refs/voice_one.wav")],
        run_voxcpm2_streaming_func=fake_stream,
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

    button = _find(window, "run_stream_button")
    output = _find(window, "stream_output")
    button.click()
    QtTest.QTest.qWait(30)

    assert calls == [
        StreamCall(
            text="hello streaming",
            reference_wav_path="C:/refs/voice_one.wav",
            output_path="C:/out/generated.wav",
            play_audio=True,
            audio_device="7: Test Speakers",
            playback_prebuffer_s=0.65,
            audio_latency="low",
        )
    ]
    assert output.toPlainText() == "stream ok"
    assert button.isEnabled()
    assert window.statusBar().currentMessage() == "VoxCPM2 streaming complete"
