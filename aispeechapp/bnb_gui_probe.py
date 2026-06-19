from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from PySide6.QtWidgets import QApplication

from aispeechapp.candidates import PROJECT_ROOT
from aispeechapp.gui import create_main_window


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "bnb_gui_probe"
DEFAULT_SETTINGS_PATH = PROJECT_ROOT / "outputs" / "bnb_gui_probe_settings.local.json"


def stop_existing_gui_processes() -> list[int]:
    script = f"""
$selfPid = {os.getpid()}
$matches = Get-CimInstance Win32_Process | Where-Object {{
    $_.ProcessId -ne $selfPid -and
    $_.Name -match 'python|pythonw' -and
    ($_.CommandLine -match 'aispeechapp\\.gui|aispeech-gui')
}}
$matches | ForEach-Object {{
    Stop-Process -Id $_.ProcessId -Force
    $_.ProcessId
}}
"""
    completed = subprocess.run(
        ["powershell", "-NoProfile", "-Command", script],
        text=True,
        capture_output=True,
        check=False,
    )
    stopped = []
    for line in completed.stdout.splitlines():
        line = line.strip()
        if line.isdigit():
            stopped.append(int(line))
    return stopped


def process_events(app: QApplication, seconds: float = 0.1) -> None:
    deadline = time.time() + max(0.0, seconds)
    while time.time() < deadline:
        app.processEvents()
        time.sleep(0.02)


def _select_combo_data(combo, value: str) -> None:
    index = combo.findData(value)
    if index < 0:
        raise RuntimeError(f"Could not find combo data: {value}")
    combo.setCurrentIndex(index)


def run_probe(
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    output_path: Path | None = None,
    reference_wav_path: Path | None = None,
    stop_existing: bool = False,
    timeout_s: float = 900.0,
) -> dict[str, object]:
    stopped = stop_existing_gui_processes() if stop_existing else []
    output_dir.mkdir(parents=True, exist_ok=True)
    output_wav = output_dir / "voxcpm2_bnb_int8_probe.wav"
    report_path = output_path or output_dir / "bnb_gui_probe.json"
    reference = reference_wav_path or PROJECT_ROOT.parent / "OmniChat" / "voices" / "eric_snyder.wav"
    if not reference.exists():
        raise FileNotFoundError(f"Reference WAV does not exist: {reference}")

    app = QApplication.instance() or QApplication([])
    window = create_main_window(settings_path=DEFAULT_SETTINGS_PATH)
    window.setWindowTitle("AISpeechApp - BNB GUI Probe")
    window.show()
    process_events(app, 0.25)

    _select_combo_data(window._synthesis_candidate_box, "voxcpm2_bnb_int8")
    window._reference_path.setText(str(reference))
    window._stream_text.setPlainText(
        "BNB GUI probe: this short sentence verifies that BitsAndBytes int8 is active."
    )
    window._stream_output_path.setText(str(output_wav))
    window._play_audio_checkbox.setChecked(False)
    if "inference_timesteps" in window._generation_parameter_widgets:
        window._generation_parameter_widgets["inference_timesteps"].setValue(6)
    if "seed" in window._generation_parameter_widgets:
        window._generation_parameter_widgets["seed"].setValue(123)
    if "quantization_method" in window._generation_parameter_widgets:
        _select_combo_data(window._generation_parameter_widgets["quantization_method"], "bnb_int8_linear")
    if "quantization_targets" in window._generation_parameter_widgets:
        _select_combo_data(window._generation_parameter_widgets["quantization_targets"], "lm_only")

    screenshot_before = output_dir / "bnb_gui_probe_before.png"
    window.grab().save(str(screenshot_before))
    window._run_stream_button.click()

    started = time.time()
    payload: dict[str, object] | None = None
    while time.time() - started < timeout_s:
        app.processEvents()
        text = window._stream_output.toPlainText()
        if text.strip().startswith("{") and not text.strip().startswith('{"status": "failed"'):
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                payload = None
            if payload is not None and payload.get("candidate_id") == "voxcpm2_bnb_int8":
                break
        if text.strip().startswith("{"):
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                payload = None
            if isinstance(payload, dict) and payload.get("status") == "failed":
                break
        time.sleep(0.1)

    screenshot_after = output_dir / "bnb_gui_probe_after.png"
    window.grab().save(str(screenshot_after))
    status_message = window.statusBar().currentMessage()
    window.close()
    app.processEvents()

    if payload is None:
        payload = {
            "status": "failed",
            "error": "Timed out waiting for BNB GUI probe output.",
            "stream_output": window._stream_output.toPlainText(),
        }

    sidecar_payload = {}
    metadata_path = payload.get("metadata_path") if isinstance(payload, dict) else None
    if isinstance(metadata_path, str) and Path(metadata_path).exists():
        sidecar_payload = json.loads(Path(metadata_path).read_text(encoding="utf-8"))

    quantization_status = str(
        payload.get("quantization_status")
        or sidecar_payload.get("quantization_status")
        or sidecar_payload.get("result", {}).get("quantization_status")
        or ""
    )
    passed = (
        payload.get("candidate_id") == "voxcpm2_bnb_int8"
        and payload.get("metadata_path")
        and Path(str(payload["metadata_path"])).exists()
        and output_wav.exists()
        and quantization_status.startswith("bnb_int8_linear:")
    )
    report = {
        "passed": bool(passed),
        "stopped_process_ids": stopped,
        "status_message": status_message,
        "output_wav": str(output_wav),
        "output_exists": output_wav.exists(),
        "metadata_path": payload.get("metadata_path"),
        "metadata_exists": bool(payload.get("metadata_path") and Path(str(payload["metadata_path"])).exists()),
        "quantization_status": quantization_status,
        "payload": payload,
        "screenshots": {
            "before": str(screenshot_before),
            "after": str(screenshot_after),
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a real AISpeechApp GUI BNB quantization probe.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reference-wav-path", type=Path)
    parser.add_argument("--stop-existing", action="store_true")
    parser.add_argument("--timeout-s", type=float, default=900.0)
    args = parser.parse_args(argv)

    report = run_probe(
        output_dir=args.output_dir,
        output_path=args.output,
        reference_wav_path=args.reference_wav_path,
        stop_existing=args.stop_existing,
        timeout_s=args.timeout_s,
    )
    print(json.dumps(report), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
