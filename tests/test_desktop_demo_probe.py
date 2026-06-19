from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_desktop_demo_probe_exercises_real_pyside_window(tmp_path: Path):
    output_dir = tmp_path / "desktop_demo"
    output_path = output_dir / "desktop_demo_probe.json"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "aispeechapp.desktop_demo_probe",
            "--output-dir",
            str(output_dir),
            "--output",
            str(output_path),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, (
        f"Desktop demo probe failed\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
    )
    assert output_path.exists()

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["overall_passed"] is True
    assert payload["passed"] == 3
    assert payload["failed"] == 0
    voice_matrix = payload["results"][1]
    assert voice_matrix["title"] == "VoxCPM2 streaming voice/prompt matrix"
    assert len(voice_matrix["generated"]) == 4
    assert voice_matrix["per_prompt_voice_hashes_differ"] == [True, True]
    for item in voice_matrix["generated"]:
        assert Path(item["output_path"]).exists()
        assert item["output_sha256"]
        assert Path(item["metadata_path"]).exists()
        assert item["time_to_first_output_s"] is not None
    for path in payload["screenshots"].values():
        assert Path(path).exists()
