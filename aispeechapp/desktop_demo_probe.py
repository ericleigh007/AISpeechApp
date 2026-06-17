from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from PySide6.QtWidgets import QApplication

from aispeechapp.candidates import PROJECT_ROOT
from aispeechapp.gui import _demo_smoke, _demo_streaming, create_main_window
from aispeechapp.voxcpm2_streaming import DEFAULT_OUTPUT_DIR


def process_events(app: QApplication, seconds: float = 0.1) -> None:
    deadline = time.time() + max(0.0, seconds)
    while time.time() < deadline:
        app.processEvents()
        time.sleep(0.02)


def save_screenshot(app: QApplication, window, output_dir: Path, name: str) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    process_events(app, 0.1)
    path = output_dir / f"{name}.png"
    window.grab().save(str(path))
    return str(path)


def run_probe(output_dir: Path, output_path: Path) -> dict:
    app = QApplication.instance() or QApplication([])
    window = create_main_window(
        run_smoke_func=_demo_smoke,
        load_audio_devices_func=lambda: ["Demo output device"],
        run_voxcpm2_streaming_func=_demo_streaming,
        settings_path=output_dir / "desktop_demo_probe_settings.local.json",
    )
    window.setWindowTitle("AISpeechApp - Desktop Demo Probe")
    window.show()
    process_events(app, 0.2)

    screenshots: dict[str, str] = {
        "startup": save_screenshot(app, window, output_dir, "00_startup")
    }
    results: list[dict] = []

    window._tabs.setCurrentIndex(0)
    window._run_smoke_button.click()
    process_events(app, 0.2)
    smoke_text = window._smoke_output.toPlainText()
    results.append(
        {
            "act": 1,
            "title": "Metadata smoke",
            "passed": "metadata smoke path exercised" in smoke_text,
            "output_chars": len(smoke_text),
            "status": window.statusBar().currentMessage(),
        }
    )
    screenshots["smoke"] = save_screenshot(app, window, output_dir, "01_metadata_smoke")

    prompts = [
        "Probe prompt one: crisp consonants, changing rhythm, and a natural pause.",
        "Probe prompt two: numbers 17 and 42, Portuguese names, and careful pacing.",
    ]
    voxcpm_index = window._synthesis_candidate_box.findData("voxcpm2")
    if voxcpm_index >= 0:
        window._synthesis_candidate_box.setCurrentIndex(voxcpm_index)
        process_events(app, 0.05)
    voices = ["Pegasus", "Eric Snyder"]
    generated: list[dict] = []
    for voice_name in voices:
        voice_index = window._voice_reference_box.findText(voice_name)
        if voice_index < 0:
            continue
        window._voice_reference_box.setCurrentIndex(voice_index)
        process_events(app, 0.05)
        for prompt_index, prompt in enumerate(prompts, start=1):
            window._tabs.setCurrentIndex(1)
            window._stream_text.setPlainText(prompt)
            safe_voice = voice_name.lower().replace(" ", "_")
            output_wav = DEFAULT_OUTPUT_DIR / f"desktop_probe_{safe_voice}_{prompt_index}.wav"
            window._stream_output_path.setText(str(output_wav))
            window._play_audio_checkbox.setChecked(True)
            window._run_stream_button.click()
            process_events(app, 0.2)
            payload = json.loads(window._stream_output.toPlainText())
            generated.append(
                {
                    "voice": voice_name,
                    "prompt_index": prompt_index,
                    "reference_wav_path": payload.get("reference_wav_path"),
                    "output_path": payload.get("output_path"),
                    "output_exists": Path(payload.get("output_path", "")).exists(),
                    "output_sha256": payload.get("output_sha256"),
                    "first_chunk_latency_s": payload.get("first_chunk_latency_s"),
                    "realtime_factor": payload.get("realtime_factor"),
                    "chunk_count": payload.get("chunk_count"),
                }
            )

    per_prompt_voice_hashes_differ = []
    for prompt_index in range(1, len(prompts) + 1):
        hashes = {
            item["voice"]: item["output_sha256"]
            for item in generated
            if item["prompt_index"] == prompt_index and item.get("output_sha256")
        }
        per_prompt_voice_hashes_differ.append(
            len(hashes) == len(voices) and len(set(hashes.values())) == len(voices)
        )
    results.append(
        {
            "act": 2,
            "title": "VoxCPM2 streaming voice/prompt matrix",
            "passed": len(generated) == len(prompts) * len(voices)
            and all(item["output_exists"] for item in generated)
            and all(per_prompt_voice_hashes_differ),
            "prompts": prompts,
            "voices": voices,
            "generated": generated,
            "per_prompt_voice_hashes_differ": per_prompt_voice_hashes_differ,
            "status": window.statusBar().currentMessage(),
        }
    )
    screenshots["streaming"] = save_screenshot(app, window, output_dir, "02_streaming")

    window._load_latency_history_button.click()
    process_events(app, 0.2)
    history_text = window._stream_output.toPlainText()
    results.append(
        {
            "act": 3,
            "title": "Latency history",
            "passed": "realtime_factor" in history_text and "0.8875" in history_text,
            "output_chars": len(history_text),
            "status": window.statusBar().currentMessage(),
        }
    )
    screenshots["history"] = save_screenshot(app, window, output_dir, "03_latency_history")

    passed = sum(1 for item in results if item["passed"])
    report = {
        "overall_passed": passed == len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "results": results,
        "screenshots": screenshots,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    window.close()
    app.processEvents()
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run AISpeechApp desktop demo probe.")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "desktop_demo")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    output_path = args.output or args.output_dir / "desktop_demo_probe.json"
    report = run_probe(args.output_dir, output_path)
    print(json.dumps(report), flush=True)
    return 0 if report["overall_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
