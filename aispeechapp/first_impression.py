from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import os

from aispeechapp.cache_paths import configure_model_caches
from aispeechapp.candidates import PROJECT_ROOT, Candidate, get_candidate, load_candidates


PROMPTS_PATH = PROJECT_ROOT / "configs" / "first_impression_prompts.json"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "first_impression"
REPORT_PATH = PROJECT_ROOT / "reports" / "first_impression.json"


@dataclass(frozen=True)
class PromptCase:
    language_code: str
    label: str
    language_hint: str
    text: str


@dataclass
class FirstImpressionResult:
    candidate_id: str
    model_name: str
    language_code: str
    text: str
    output_path: str
    status: str
    elapsed_s: float
    details: dict[str, Any]


def load_prompt_cases(candidate: Candidate) -> list[PromptCase]:
    raw = json.loads(PROMPTS_PATH.read_text(encoding="utf-8"))
    cases = []
    for language_code, item in raw["languages"].items():
        cases.append(
            PromptCase(
                language_code=language_code,
                label=item["label"],
                language_hint=item["language_hint"],
                text=item["template"].format(model_name=candidate.name),
            )
        )
    return cases


def _backend_command(candidate: Candidate, case: PromptCase, output_path: Path) -> list[str]:
    script = PROJECT_ROOT / "scripts" / "synthesize_backend.py"
    return [
        sys.executable,
        str(script),
        "--candidate",
        candidate.id,
        "--text",
        case.text,
        "--language-code",
        case.language_code,
        "--language-hint",
        case.language_hint,
        "--output",
        str(output_path),
    ]


def run_case(candidate: Candidate, case: PromptCase) -> FirstImpressionResult:
    output_path = OUTPUT_ROOT / candidate.id / f"{case.language_code}.wav"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    cache_env = configure_model_caches()
    env = os.environ.copy()
    env.update(cache_env)
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUTF8", "1")
    completed = subprocess.run(
        _backend_command(candidate, case, output_path),
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )
    elapsed = round(time.perf_counter() - start, 3)
    status = "ok" if completed.returncode == 0 and output_path.exists() else "failed"
    return FirstImpressionResult(
        candidate_id=candidate.id,
        model_name=candidate.name,
        language_code=case.language_code,
        text=case.text,
        output_path=str(output_path),
        status=status,
        elapsed_s=elapsed,
        details={
            "returncode": completed.returncode,
            "stdout": (completed.stdout or "")[-4000:],
            "stderr": (completed.stderr or "")[-4000:],
        },
    )


def write_report(results: list[FirstImpressionResult], path: Path = REPORT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at_unix": time.time(),
        "results": [asdict(result) for result in results],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate first-impression TTS samples.")
    parser.add_argument("--all-priority-one", action="store_true")
    parser.add_argument("--candidate", action="append", default=[])
    parser.add_argument("--language-code", action="append", default=[])
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.all_priority_one:
        candidates = [candidate for candidate in load_candidates() if candidate.priority == 1]
    elif args.candidate:
        candidates = [get_candidate(candidate_id) for candidate_id in args.candidate]
    else:
        raise SystemExit("Pass --all-priority-one or at least one --candidate.")

    results = []
    language_filter = set(args.language_code)
    for candidate in candidates:
        for case in load_prompt_cases(candidate):
            if language_filter and case.language_code not in language_filter:
                continue
            result = run_case(candidate, case)
            results.append(result)
            print(f"{candidate.id} {case.language_code}: {result.status} -> {result.output_path}")

    write_report(results)
    print(f"Wrote {REPORT_PATH}")
    return 0 if all(result.status == "ok" for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
