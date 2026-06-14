from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from aispeechapp.cache_paths import configure_model_caches

_CACHE_ENV = configure_model_caches()

from huggingface_hub import HfApi, scan_cache_dir  # noqa: E402
from huggingface_hub.errors import HfHubHTTPError  # noqa: E402

from aispeechapp.candidates import Candidate, PROJECT_ROOT, get_candidate, load_candidates  # noqa: E402


@dataclass
class ImportCheck:
    module: str
    available: bool


@dataclass
class SmokeResult:
    candidate_id: str
    name: str
    repo_id: str
    priority: int
    metadata_ok: bool
    import_checks: list[ImportCheck]
    cache_present: bool
    status: str
    elapsed_s: float
    details: dict[str, Any]


def _check_imports(candidate: Candidate) -> list[ImportCheck]:
    checks = []
    modules = list(candidate.expected_imports)
    if candidate.id == "dots_tts_soar":
        modules.append("pynini")
    for module in modules:
        checks.append(ImportCheck(module=module, available=importlib.util.find_spec(module) is not None))
    return checks


def _check_hf_metadata(candidate: Candidate) -> tuple[bool, dict[str, Any]]:
    api = HfApi()
    try:
        info = api.model_info(candidate.repo_id)
    except HfHubHTTPError as exc:
        return False, {"error": str(exc)}

    return True, {
        "model_id": info.modelId,
        "sha": info.sha,
        "last_modified": info.last_modified.isoformat() if info.last_modified else None,
        "downloads": getattr(info, "downloads", None),
        "likes": getattr(info, "likes", None),
        "tags": list(getattr(info, "tags", []) or []),
    }


def _check_cache(candidate: Candidate) -> bool:
    if candidate.id == "indextts2":
        model_root = Path(_CACHE_ENV["AI_SPEECH_MODEL_ROOT"])
        return (model_root / "indextts2-runtime" / "checkpoints" / "config.yaml").exists()

    try:
        cache = scan_cache_dir()
    except Exception:
        return False

    normalized = f"models--{candidate.repo_id.replace('/', '--')}"
    for repo in cache.repos:
        if repo.repo_id == candidate.repo_id or repo.repo_path.name == normalized:
            return True
    return False


def run_metadata_smoke(candidate: Candidate) -> SmokeResult:
    start = time.perf_counter()
    metadata_ok, metadata = _check_hf_metadata(candidate)
    imports = _check_imports(candidate)
    cache_present = _check_cache(candidate)
    missing_imports = [check.module for check in imports if not check.available]

    if not metadata_ok:
        status = "metadata_failed"
    elif missing_imports:
        status = "needs_install"
    elif not cache_present:
        status = "needs_model_download"
    else:
        status = "ready_for_synthesis_smoke"

    return SmokeResult(
        candidate_id=candidate.id,
        name=candidate.name,
        repo_id=candidate.repo_id,
        priority=candidate.priority,
        metadata_ok=metadata_ok,
        import_checks=imports,
        cache_present=cache_present,
        status=status,
        elapsed_s=round(time.perf_counter() - start, 3),
        details={"metadata": metadata, "capabilities": candidate.capabilities, "notes": candidate.notes},
    )


def write_results(results: list[SmokeResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at_unix": time.time(),
        "results": [
            {
                **asdict(result),
                "import_checks": [asdict(check) for check in result.import_checks],
            }
            for result in results
        ],
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def print_table(results: list[SmokeResult]) -> None:
    print(f"{'id':30} {'prio':>4} {'metadata':>8} {'imports':>9} {'cache':>7} status")
    print("-" * 86)
    for result in results:
        imports_ok = all(check.available for check in result.import_checks)
        print(
            f"{result.candidate_id:30} "
            f"{result.priority:>4} "
            f"{str(result.metadata_ok):>8} "
            f"{str(imports_ok):>9} "
            f"{str(result.cache_present):>7} "
            f"{result.status}"
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AISpeechApp local TTS smoke checks.")
    parser.add_argument("--list", action="store_true", help="List configured candidates and exit.")
    parser.add_argument("--all", action="store_true", help="Smoke all configured candidates.")
    parser.add_argument("--candidate", action="append", default=[], help="Candidate id to smoke.")
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Run metadata/import/cache checks only. This is currently the implemented smoke level.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "reports" / "smoke_metadata.json",
        help="JSON output path.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    candidates = load_candidates()

    if args.list:
        for candidate in candidates:
            print(f"{candidate.id:30} priority={candidate.priority} repo={candidate.repo_id}")
        return 0

    if not args.metadata_only:
        print("Only metadata/import/cache smoke checks are implemented in this scaffold.", file=sys.stderr)
        return 2

    if args.all:
        selected = candidates
    elif args.candidate:
        selected = [get_candidate(candidate_id) for candidate_id in args.candidate]
    else:
        selected = [candidate for candidate in candidates if candidate.priority == 1]

    results = [run_metadata_smoke(candidate) for candidate in selected]
    write_results(results, args.output)
    print_table(results)
    print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
