# AISpeechApp Agent Rules

## Python Environment

- Use the repository-local `.venv` for all Python commands.
- On Windows, run Python as `.venv\Scripts\python.exe`.
- Do not fall back to another interpreter silently. If `.venv` is missing or broken, report it.

## Model And Artifact Storage

- Keep heavyweight model weights and caches off the system drive. Prefer `D:\AI\Models` and the cache settings in `aispeechapp.cache_paths`.
- Do not commit generated WAVs, screenshots, reports, model weights, or local voice reference files.
- Keep `outputs/`, `reports/`, and `voice_refs/` ignored except for placeholder/documentation files.

## Desktop GUI Direction

- The desktop GUI should follow OmniChat RT's native PySide6 pattern: a `QMainWindow` shell, injectable backends, and app-borne demo probes.
- Do not replace the native desktop app with a browser UI.
- GUI changes must remain testable without requiring manual human testing.
- Prefer app-borne probes that instantiate and drive the real PySide6 window, capture screenshots, and write JSON reports.

## Testing Expectations

- Run focused tests after changes, then the full relevant suite before committing.
- Use:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
```

- The desktop probe command is:

```powershell
.\run_desktop_probe.bat
```

## Current Product Notes

- VoxCPM2 is the leading local TTS/voice-cloning candidate so far.
- The GUI keeps the VoxCPM2 model in-process for the GUI session to avoid reloading on each generation.
- Playback defaults favor smoothness: `0.45s` prebuffer and high PortAudio latency. Lower these only when tuning for minimum start latency.
