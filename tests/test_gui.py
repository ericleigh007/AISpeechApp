from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from PySide6 import QtTest, QtWidgets

from aispeechapp.gui import (
    _effective_synthesis_text,
    _format_report_age,
    _smoke_report_rows,
    _text_control_generation_options,
    create_main_window,
)


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
    details = _find(window, "smoke_details_output")
    button.click()
    QtTest.QTest.qWait(30)

    assert calls == [None]
    assert output.rowCount() == 0
    assert details.toPlainText() == "smoke ok"
    assert button.isEnabled()
    assert window.statusBar().currentMessage() == "Metadata smoke complete"


def test_gui_smoke_output_uses_native_sortable_table(tmp_path):
    _app()
    window = create_main_window(
        load_audio_devices_func=lambda: [],
        load_voice_references_func=lambda: [],
        settings_path=tmp_path / "settings.local.json",
    )

    output = _find(window, "smoke_output")

    assert isinstance(output, QtWidgets.QTableWidget)
    assert output.isSortingEnabled()
    assert output.selectionBehavior() == QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows


def test_gui_smoke_json_populates_highlighted_sortable_rows(tmp_path):
    _app()
    payload = {
        "results": [
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
            {
                "candidate_id": "voxcpm2",
                "name": "VoxCPM2",
                "repo_id": "openbmb/VoxCPM2",
                "priority": 1,
                "metadata_ok": True,
                "import_checks": [{"module": "torch", "available": True}],
                "cache_present": True,
                "status": "ready_for_synthesis_smoke",
                "details": {"capabilities": ["tts", "voice_clone"], "model_scale": "2B"},
            },
        ]
    }
    window = create_main_window(
        run_smoke_func=lambda _candidate_id: json.dumps(payload),
        load_audio_devices_func=lambda: [],
        load_voice_references_func=lambda: [],
        settings_path=tmp_path / "settings.local.json",
    )
    output = _find(window, "smoke_output")

    _find(window, "run_smoke_button").click()
    QtTest.QTest.qWait(30)

    assert output.rowCount() == 2
    assert output.item(0, 0).text() == "voxcpm2"
    assert output.item(1, 0).text() == "zonos2"
    assert output.item(0, 2).text() == "yes"
    assert output.item(1, 2).text() == "no"
    assert output.item(0, 0).background().color().name().lower() == "#dcfce7"
    assert output.item(0, 0).foreground().color().name().lower() == "#14532d"


def test_gui_diagnostics_loads_latest_smoke_report_on_startup(tmp_path):
    _app()
    payload = {
        "generated_at_unix": 1000.0,
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
                "details": {"capabilities": ["tts", "voice_clone"], "model_scale": "2B"},
            }
        ]
    }
    window = create_main_window(
        load_latest_report_func=lambda: json.dumps(payload),
        load_audio_devices_func=lambda: [],
        load_voice_references_func=lambda: [],
        settings_path=tmp_path / "settings.local.json",
    )
    output = _find(window, "smoke_output")
    report_age = _find(window, "smoke_report_age_label")

    assert output.rowCount() == 1
    assert output.item(0, 0).text() == "voxcpm2"
    assert output.item(0, 2).text() == "yes"
    assert report_age.text().startswith("Report generated:")


def test_report_age_formatter_uses_timestamp_and_age():
    assert _format_report_age({"generated_at_unix": 1000.0}, now=1125.0).endswith("(2m old)")
    assert _format_report_age({}, now=1125.0) == "Report age: unknown"


def test_smoke_report_rows_format_alpha_sorted_capability_table():
    rows = _smoke_report_rows(
        {
            "results": [
                {
                    "candidate_id": "voxcpm2",
                    "name": "VoxCPM2",
                    "repo_id": "openbmb/VoxCPM2",
                    "priority": 1,
                    "metadata_ok": True,
                    "import_checks": [{"module": "voxcpm", "available": True}],
                    "cache_present": True,
                    "status": "ready_for_synthesis_smoke",
                    "details": {
                        "capabilities": ["tts", "voice_clone", "voice_design", "multilingual", "48khz"],
                        "model_scale": "2B",
                    },
                },
                {
                    "candidate_id": "omnivoice",
                    "name": "OmniVoice",
                    "repo_id": "k2-fsa/OmniVoice",
                    "priority": 2,
                    "metadata_ok": True,
                    "import_checks": [{"module": "missing", "available": False}],
                    "cache_present": False,
                    "status": "needs_install",
                    "details": {
                        "capabilities": [
                            "tts",
                            "voice_clone",
                            "voice_design",
                            "massively_multilingual",
                        ]
                    },
                },
            ]
        }
    )

    assert [row["values"][0] for row in rows] == ["omnivoice", "voxcpm2"]
    assert rows[0]["gui_runnable"] is True
    assert rows[1]["gui_runnable"] is True
    assert rows[1]["values"] == [
        "voxcpm2",
        "1",
        "yes",
        "yes",
        "no",
        "multi",
        "48KHZ",
        "2B",
        "voice design",
        "yes",
        "yes",
        "yes",
        "ready_for_synthesis_smoke",
    ]
    assert rows[0]["values"] == [
        "omnivoice",
        "2",
        "yes",
        "yes",
        "no",
        "massive multi",
        "-",
        "unknown",
        "voice design",
        "yes",
        "no",
        "no",
        "needs_install",
    ]


def test_gui_opens_to_synthesis_before_diagnostics(tmp_path):
    _app()
    window = create_main_window(
        load_audio_devices_func=lambda: [],
        load_voice_references_func=lambda: [],
        settings_path=tmp_path / "settings.local.json",
    )
    tabs = _find(window, "main_tabs")

    assert tabs.currentIndex() == window._synthesis_tab_index
    assert tabs.tabText(window._synthesis_tab_index) == "Synthesis"
    assert tabs.tabText(window._diagnostics_tab_index) == "Diagnostics"


def test_gui_synthesis_model_list_is_alpha_sorted(tmp_path):
    _app()
    window = create_main_window(
        load_audio_devices_func=lambda: [],
        load_voice_references_func=lambda: [],
        settings_path=tmp_path / "settings.local.json",
    )
    model_box = _find(window, "synthesis_candidate_box")
    labels = [model_box.itemText(index).split(" - ", 1)[1] for index in range(model_box.count())]

    assert labels == sorted(labels, key=str.lower)


def test_gui_common_and_dynamic_controls_expose_help_tooltips(tmp_path):
    _app()
    window = create_main_window(
        load_audio_devices_func=lambda: [],
        load_voice_references_func=lambda: [],
        settings_path=tmp_path / "settings.local.json",
    )

    assert "Longer text" in _find(window, "stream_text").toolTip()
    assert "JSON sidecar" in _find(window, "stream_output_path").toolTip()
    assert "stutter" in _find(window, "playback_prebuffer_box").toolTip()
    assert "VoxCPM2 streaming" in _find(window, "playback_mode_box").toolTip()
    assert "clean streaming playback" in _find(window, "audio_latency_box").toolTip()
    assert "ringing" in _find(window, "generation_parameter_cfg_value").toolTip()
    assert "latency" in _find(window, "generation_parameter_inference_timesteps").toolTip()
    assert "pace" in _find(window, "text_control_control_instruction").toolTip()


def test_effective_text_uses_parenthesized_control_instruction():
    controls = (
        {
            "id": "control_instruction",
            "mode": "parenthesized_prefix",
        },
    )

    assert _effective_synthesis_text(
        "Hello.",
        controls,
        {"control_instruction": "(slow natural pace)"},
    ) == "(slow natural pace)Hello."


def test_text_controls_can_be_passed_as_generation_options():
    controls = (
        {
            "id": "style_instruction",
            "mode": "generation_option",
            "option_id": "style_instruction",
        },
    )

    assert _text_control_generation_options(
        controls,
        {"style_instruction": "warm narration"},
    ) == {"style_instruction": "warm narration"}


@dataclass
class StreamCall:
    candidate_id: str
    candidate_name: str
    voice_name: str
    text: str
    spoken_text: str | None
    text_control_options: dict[str, str]
    reference_wav_path: str
    output_path: str
    play_audio: bool
    audio_device: str | None
    playback_prebuffer_s: float
    audio_latency: str | None
    generation_options: dict[str, object]


def test_gui_voxcpm2_streaming_button_passes_reference_device_and_output(tmp_path):
    _app()
    calls: list[StreamCall] = []

    def fake_stream(**kwargs):
        calls.append(StreamCall(**kwargs))
        return "stream ok"

    window = create_main_window(
        load_audio_devices_func=lambda: ["7: Test Speakers"],
        load_voice_references_func=lambda: [("Voice One", "C:/refs/voice_one.wav")],
        run_voxcpm2_streaming_func=fake_stream,
        settings_path=tmp_path / "settings.local.json",
    )

    tabs = _find(window, "main_tabs")
    tabs.setCurrentIndex(window._synthesis_tab_index)
    _find(window, "stream_text").setPlainText("hello streaming")
    _find(window, "text_control_control_instruction").setPlainText("slow natural pace")
    voice_box = _find(window, "voice_reference_box")
    voice_box.setCurrentIndex(1)
    _find(window, "stream_output_path").setText("C:/out/generated.wav")
    audio_device = _find(window, "audio_device_box")
    audio_device.setCurrentIndex(1)
    _find(window, "playback_prebuffer_box").setValue(0.65)
    _find(window, "normalize_audio_checkbox").setChecked(True)
    _find(window, "audio_target_peak_box").setValue(0.9)
    latency = _find(window, "audio_latency_box")
    latency.setCurrentIndex(1)
    guidance = _find(window, "generation_parameter_cfg_value")
    guidance.setValue(2.7)
    steps = _find(window, "generation_parameter_inference_timesteps")
    steps.setValue(14)
    seed = _find(window, "generation_parameter_seed")
    seed.setValue(22)

    button = _find(window, "run_stream_button")
    output = _find(window, "stream_output")
    button.click()
    assert not button.isEnabled()
    _wait_until(lambda: bool(calls) and output.toPlainText() == "stream ok")

    assert calls[0] == StreamCall(
        candidate_id="voxcpm2",
        candidate_name="VoxCPM2",
        voice_name="Voice One",
        text="(slow natural pace)hello streaming",
        spoken_text="hello streaming",
        text_control_options={"control_instruction": "slow natural pace"},
        reference_wav_path="C:/refs/voice_one.wav",
        output_path="C:/out/generated.wav",
        play_audio=True,
        audio_device="7: Test Speakers",
        playback_prebuffer_s=0.65,
        audio_latency="low",
        generation_options={
            "cfg_value": 2.7,
            "inference_timesteps": 14,
            "seed": 22,
            "normalize": True,
            "denoise": False,
            "retry_badcase": False,
            "retry_badcase_max_times": 3,
            "retry_badcase_ratio_threshold": 6.0,
            "torch_compile": False,
            "torch_compile_mode": "reduce-overhead",
            "min_len": 2,
            "max_len": 4096,
            "audio_normalization": True,
            "audio_target_peak": 0.9,
            "playback_mode": "after_generation",
        },
    )
    assert output.toPlainText() == "stream ok"
    assert button.isEnabled()
    assert window.statusBar().currentMessage() == "VoxCPM2 complete"


def test_gui_quantized_voxcpm2_candidate_uses_isolated_streaming_path(tmp_path):
    _app()
    calls: list[StreamCall] = []

    def fake_stream(**kwargs):
        calls.append(StreamCall(**kwargs))
        return "quant stream ok"

    window = create_main_window(
        load_audio_devices_func=lambda: [],
        load_voice_references_func=lambda: [],
        run_voxcpm2_streaming_func=fake_stream,
        settings_path=tmp_path / "settings.local.json",
    )

    candidate_box = _find(window, "synthesis_candidate_box")
    candidate_box.setCurrentIndex(candidate_box.findData("voxcpm2_quantized"))
    _find(window, "stream_text").setPlainText("hello quantized")
    _find(window, "generation_parameter_quantization_method").setCurrentIndex(
        _find(window, "generation_parameter_quantization_method").findData("dynamic_int8_cpu")
    )
    _find(window, "generation_parameter_quantization_targets").setCurrentIndex(
        _find(window, "generation_parameter_quantization_targets").findData("base_lm_only")
    )
    _find(window, "run_stream_button").click()
    _wait_until(lambda: bool(calls))

    assert calls[0].candidate_id == "voxcpm2_quantized"
    assert calls[0].candidate_name == "VoxCPM2 Quantized"
    assert calls[0].generation_options["quantization_method"] == "dynamic_int8_cpu"
    assert calls[0].generation_options["quantization_targets"] == "base_lm_only"
    assert window.statusBar().currentMessage() == "VoxCPM2 Quantized complete"


def test_gui_torchao_voxcpm2_candidate_uses_real_quantization_method(tmp_path):
    _app()
    calls: list[StreamCall] = []

    def fake_stream(**kwargs):
        calls.append(StreamCall(**kwargs))
        return "torchao stream ok"

    window = create_main_window(
        load_audio_devices_func=lambda: [],
        load_voice_references_func=lambda: [],
        run_voxcpm2_streaming_func=fake_stream,
        settings_path=tmp_path / "settings.local.json",
    )

    candidate_box = _find(window, "synthesis_candidate_box")
    candidate_box.setCurrentIndex(candidate_box.findData("voxcpm2_torchao_int8"))
    _find(window, "stream_text").setPlainText("hello torchao")
    _find(window, "run_stream_button").click()
    _wait_until(lambda: bool(calls))

    assert calls[0].candidate_id == "voxcpm2_torchao_int8"
    assert calls[0].candidate_name == "VoxCPM2 TorchAO Int8"
    assert calls[0].generation_options["quantization_method"] == "torchao_int8_weight_only"
    assert calls[0].generation_options["quantization_targets"] == "lm_only"
    assert window.statusBar().currentMessage() == "VoxCPM2 TorchAO Int8 complete"


def test_gui_bnb_voxcpm2_candidate_uses_bnb_quantization_method(tmp_path):
    _app()
    calls: list[StreamCall] = []

    def fake_stream(**kwargs):
        calls.append(StreamCall(**kwargs))
        return "bnb stream ok"

    window = create_main_window(
        load_audio_devices_func=lambda: [],
        load_voice_references_func=lambda: [],
        run_voxcpm2_streaming_func=fake_stream,
        settings_path=tmp_path / "settings.local.json",
    )

    candidate_box = _find(window, "synthesis_candidate_box")
    candidate_box.setCurrentIndex(candidate_box.findData("voxcpm2_bnb_int8"))
    _find(window, "stream_text").setPlainText("hello bnb")
    _find(window, "run_stream_button").click()
    _wait_until(lambda: bool(calls))

    assert calls[0].candidate_id == "voxcpm2_bnb_int8"
    assert calls[0].candidate_name == "VoxCPM2 BNB Int8"
    assert calls[0].generation_options["quantization_method"] == "bnb_int8_linear"
    assert calls[0].generation_options["quantization_targets"] == "lm_only"
    assert window.statusBar().currentMessage() == "VoxCPM2 BNB Int8 complete"


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
    tabs.setCurrentIndex(window._synthesis_tab_index)
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
            "candidate_name": "dots.tts soar",
            "voice_name": "Custom Voice",
            "text": "hello dots",
            "spoken_text": "hello dots",
            "text_control_options": {},
            "output_path": str(tmp_path / "dots.wav"),
            "language_code": "en",
            "language_hint": "English",
            "reference_wav_path": "",
            "play_audio": True,
            "audio_device": None,
            "playback_prebuffer_s": 0.45,
            "audio_latency": "high",
            "generation_options": {
                "num_steps": 12,
                "guidance_scale": 1.4,
                "audio_normalization": False,
                "audio_target_peak": 0.85,
            },
        }
    ]
    assert output.toPlainText() == "backend ok"
    assert window.statusBar().currentMessage() == "dots.tts soar complete"


def test_gui_disables_prompt_control_for_unsupported_model(tmp_path):
    _app()
    window = create_main_window(
        load_audio_devices_func=lambda: [],
        load_voice_references_func=lambda: [],
        settings_path=tmp_path / "settings.local.json",
    )

    candidate_box = _find(window, "synthesis_candidate_box")
    candidate_box.setCurrentIndex(candidate_box.findData("dots_tts_soar"))
    control = _find(window, "text_control_control_instruction")

    assert not control.isEnabled()
    assert "No separate prompt control" in control.placeholderText()


def test_gui_qwen_style_instruction_is_passed_as_backend_option(tmp_path):
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

    candidate_box = _find(window, "synthesis_candidate_box")
    candidate_box.setCurrentIndex(candidate_box.findData("qwen3_tts_17b_customvoice"))
    _find(window, "stream_text").setPlainText("hello qwen")
    _find(window, "text_control_style_instruction").setPlainText("warm calm delivery")
    _find(window, "stream_output_path").setText(str(tmp_path / "qwen.wav"))
    _find(window, "run_stream_button").click()
    _wait_until(lambda: bool(calls))

    assert calls[0]["text"] == "hello qwen"
    assert calls[0]["spoken_text"] == "hello qwen"
    assert calls[0]["text_control_options"] == {"style_instruction": "warm calm delivery"}
    assert calls[0]["generation_options"]["style_instruction"] == "warm calm delivery"


def test_gui_default_prompt_and_output_use_model_and_voice_names(tmp_path):
    _app()
    voice_path = tmp_path / "familiar_voice.wav"
    sf.write(voice_path, np.zeros(1200, dtype=np.float32), 24000)

    window = create_main_window(
        load_audio_devices_func=lambda: [],
        load_voice_references_func=lambda: [("Familiar Voice", str(voice_path))],
        settings_path=tmp_path / "settings.local.json",
    )

    voice_box = _find(window, "voice_reference_box")
    voice_box.setCurrentIndex(voice_box.findText("Familiar Voice"))
    candidate_box = _find(window, "synthesis_candidate_box")
    candidate_box.setCurrentIndex(candidate_box.findData("microsoft_vibevoice_15b"))

    prompt = _find(window, "stream_text").toPlainText()
    output_path = Path(_find(window, "stream_output_path").text())

    assert "This is VibeVoice 1.5B" in prompt
    assert "Familiar Voice" in prompt
    assert output_path.parent.name == "vibevoice_1_5b"
    assert output_path.name.startswith("familiar_voice__")
    assert output_path.name.endswith(".wav")


def test_gui_persists_selected_known_voice_and_session_controls(tmp_path):
    _app()
    settings_path = tmp_path / "settings.local.json"
    voice_path = tmp_path / "known_voice.wav"
    sf.write(voice_path, np.zeros(1200, dtype=np.float32), 24000)

    first_window = create_main_window(
        load_audio_devices_func=lambda: ["5: Test Speakers"],
        load_voice_references_func=lambda: [("Known Voice", str(voice_path))],
        settings_path=settings_path,
    )
    voice_box = _find(first_window, "voice_reference_box")
    voice_box.setCurrentIndex(voice_box.findText("Known Voice"))
    _find(first_window, "play_audio_checkbox").setChecked(False)
    audio_device = _find(first_window, "audio_device_box")
    audio_device.setCurrentIndex(audio_device.findData("5: Test Speakers"))
    _find(first_window, "playback_prebuffer_box").setValue(0.75)
    latency = _find(first_window, "audio_latency_box")
    latency.setCurrentIndex(latency.findData("low"))
    language = _find(first_window, "language_code_box")
    language.setCurrentIndex(1)
    first_window.close()

    second_window = create_main_window(
        load_audio_devices_func=lambda: ["5: Test Speakers"],
        load_voice_references_func=lambda: [("Known Voice", str(voice_path))],
        settings_path=settings_path,
    )

    assert _find(second_window, "voice_reference_box").currentText() == "Known Voice"
    assert _find(second_window, "reference_path").text() == str(voice_path)
    assert _find(second_window, "play_audio_checkbox").isChecked() is False
    assert _find(second_window, "audio_device_box").currentData() == "5: Test Speakers"
    assert _find(second_window, "playback_prebuffer_box").value() == 0.75
    assert _find(second_window, "audio_latency_box").currentData() == "low"
    selected_language_code, _selected_language_hint = _find(second_window, "language_code_box").currentData()
    assert selected_language_code == "pt-PT"


def test_gui_persists_custom_reference_wav_path(tmp_path):
    _app()
    settings_path = tmp_path / "settings.local.json"
    custom_voice = tmp_path / "custom_voice.wav"
    sf.write(custom_voice, np.zeros(1200, dtype=np.float32), 24000)

    first_window = create_main_window(
        load_audio_devices_func=lambda: [],
        load_voice_references_func=lambda: [],
        settings_path=settings_path,
    )
    reference_path = _find(first_window, "reference_path")
    reference_path.setText(str(custom_voice))
    reference_path.editingFinished.emit()
    first_window.close()

    second_window = create_main_window(
        load_audio_devices_func=lambda: [],
        load_voice_references_func=lambda: [],
        settings_path=settings_path,
    )

    assert _find(second_window, "voice_reference_box").currentText() == "Custom reference WAV"
    assert _find(second_window, "reference_path").text() == str(custom_voice)


def test_backend_synthesis_plays_completed_audio_file(monkeypatch, tmp_path: Path):
    from aispeechapp import gui

    output = tmp_path / "backend.wav"
    sf.write(output, np.ones(2400, dtype=np.float32) * 0.1, 24000)
    writes: list[np.ndarray] = []

    class FakeSoundDevice:
        def play(self, audio, samplerate, device, blocking):
            assert samplerate == 24000
            assert device == "3"
            assert blocking is True
            writes.append(audio.copy())

        def stop(self):
            return None

    monkeypatch.setitem(sys.modules, "sounddevice", FakeSoundDevice())

    class FakeProcess:
        def __init__(self, cmd, **_kwargs):
            output_arg = Path(cmd[cmd.index("--output") + 1])
            sf.write(output_arg, np.ones(2400, dtype=np.float32) * 0.1, 24000)
            self.returncode = 0

        def poll(self):
            return self.returncode

        def wait(self):
            return self.returncode

    monkeypatch.setattr(gui.subprocess, "Popen", FakeProcess)

    payload = json.loads(
        gui._run_backend_synthesis(
            candidate_id="microsoft_vibevoice_15b",
            candidate_name="VibeVoice 1.5B",
            voice_name="Test Voice",
            text="hello",
            output_path=str(output),
            language_code="en",
            language_hint="English",
            reference_wav_path=str(tmp_path / "voice.wav"),
            play_audio=True,
            audio_device="3: Speakers",
            playback_prebuffer_s=0.25,
            audio_latency="low",
            generation_options={"audio_normalization": True, "audio_target_peak": 0.8},
        )
    )

    assert payload["status"] == "complete"
    assert payload["played_to_device"] is True
    assert payload["metadata_path"].endswith(".wav.json")
    assert payload["time_to_first_output_s"] is not None
    assert payload["audio_sample_rate"] == 24000
    assert payload["audio_duration_s"] == 0.1
    assert writes
    assert np.isclose(float(np.max(np.abs(writes[0]))), 0.8)
    sidecar = json.loads(Path(payload["metadata_path"]).read_text(encoding="utf-8"))
    assert sidecar["candidate_name"] == "VibeVoice 1.5B"
    assert sidecar["voice_name"] == "Test Voice"
    assert sidecar["time_to_first_output_s"] == payload["time_to_first_output_s"]
    assert sidecar["generation_options"]["audio_target_peak"] == 0.8


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
    _find(first_window, "generation_parameter_seed").setValue(1234)
    _find(first_window, "normalize_audio_checkbox").setChecked(False)
    _find(first_window, "audio_target_peak_box").setValue(0.7)
    playback_mode = _find(first_window, "playback_mode_box")
    playback_mode.setCurrentIndex(playback_mode.findData("live"))
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
    assert _find(second_window, "generation_parameter_seed").value() == 1234
    assert _find(second_window, "normalize_audio_checkbox").isChecked() is False
    assert _find(second_window, "audio_target_peak_box").value() == 0.7
    assert _find(second_window, "playback_mode_box").currentData() == "live"

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
