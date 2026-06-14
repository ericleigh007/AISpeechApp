# AISpeechApp

Local-only TTS comparison lab for OmniChat's future speech-output layer.

The first milestone is not a polished benchmark. It is a smoke-test harness that
answers:

- Can the model install locally?
- Can it load on this machine?
- Can it synthesize a short sentence?
- Can it clone or condition from a reference voice?
- What are the rough latency, VRAM, sample-rate, and output-quality notes?

The project intentionally lives outside OmniChat so model-specific packages,
CUDA quirks, generated WAV files, and heavyweight experiments do not destabilize
the assistant app.

## Candidate Tiers

Initial high-priority candidates:

| Candidate | Repo | Why it matters |
| --- | --- | --- |
| Qwen3-TTS CustomVoice | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | Strong current local baseline for voice cloning and voice design. |
| VoxCPM2 | `openbmb/VoxCPM2` | 2B, 48 kHz, multilingual, controllable voice cloning. |
| IndexTTS2 | `IndexTeam/IndexTTS-2` | Expressive zero-shot cloning with duration/emotion control. |
| Fish Speech S2 Pro | `fishaudio/s2-pro` | High-end expressive/cloning comparison point. |
| OmniVoice | `k2-fsa/OmniVoice` | New multilingual zero-shot cloning and voice design candidate. |
| MOSS-TTS v1.5 | `OpenMOSS-Team/MOSS-TTS-v1.5` | Newer long-form/cloning candidate with pronunciation control claims. |

Watch-list candidates:

- `bosonai/higgs-audio-v3-tts-4b`
- `Zyphra/ZONOS2`
- `rednote-hilab/dots.tts-soar`
- `Supertone/supertonic-3`
- `LongCat-AudioDiT-3.5B` derived/checkpoint mirrors

## Quick Start

```powershell
cd C:\Users\ericl\Documents\ai-agents\Claude\AISpeechApp
.\scripts\bootstrap.ps1
.\.venv\Scripts\python.exe -m aispeechapp.smoke --list
.\.venv\Scripts\python.exe -m aispeechapp.smoke --all --metadata-only
```

On this workstation, the Windows `py -3.11` launcher is not registered, so the
bootstrap script prefers the uv-managed CPython 3.11/3.12 runtimes under
`%APPDATA%\uv\python`.

For development after bootstrap:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev,gui,metrics]"
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
```

The desktop GUI uses the same framework style as OmniChat RT: native PySide6,
a `QMainWindow` shell, injectable backends for tests, and an app-borne demo
probe that drives the real window and saves screenshots plus a JSON report.

```powershell
.\launch.bat
```

Run the deterministic visible demo backend:

```powershell
.\launch_demo.bat
```

Run the app-borne desktop probe:

```powershell
.\run_desktop_probe.bat
```

The probe exercises the actual PySide6 window, captures screenshots, runs a
small two-prompt/two-voice streaming matrix, verifies that each selected voice
produces a distinct WAV artifact, and writes
`outputs\desktop_demo\desktop_demo_probe.json`.

The VoxCPM2 streaming tab keeps the model in-process for the GUI session, so
the first real generation may still pay model load time but later generations
reuse the loaded model. Playback defaults favor smooth output: `0.45s`
prebuffer and high PortAudio latency. Lower those controls only when tuning for
minimum start latency.

## Repository Policy

- Use the local `.venv` only.
- Keep model weights/caches on the large model disk, not the system drive.
- Do not commit generated audio, screenshots, reports, local voice samples, or model weights.
- Keep the GUI native PySide6, matching OmniChat RT's desktop architecture.
- Preserve app-borne demo tests for visible GUI workflows.

## Smoke Levels

| Level | Meaning |
| --- | --- |
| `metadata` | Query Hugging Face metadata and confirm the candidate exists. |
| `import` | Confirm declared Python packages can be imported. |
| `cache` | Confirm the model is already available locally or report download need. |
| `synthesis` | Generate one short neutral WAV. |
| `clone` | Generate one short WAV using a reference voice. |

Only `metadata` and import/cache readiness are implemented in the initial
scaffold. Synthesis and clone smokes will be added backend by backend after we
decide install order.
