from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANDIDATES_PATH = PROJECT_ROOT / "configs" / "candidates.json"


@dataclass(frozen=True)
class Candidate:
    id: str
    name: str
    priority: int
    repo_id: str
    backend: str
    expected_imports: tuple[str, ...]
    capabilities: tuple[str, ...]
    notes: str = ""
    generation_parameters: tuple[dict[str, Any], ...] = ()


def load_candidates(path: Path = DEFAULT_CANDIDATES_PATH) -> list[Candidate]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    candidates = []
    for item in raw["candidates"]:
        candidates.append(
            Candidate(
                id=item["id"],
                name=item["name"],
                priority=int(item["priority"]),
                repo_id=item["repo_id"],
                backend=item["backend"],
                expected_imports=tuple(item.get("expected_imports", [])),
                capabilities=tuple(item.get("capabilities", [])),
                notes=item.get("notes", ""),
                generation_parameters=tuple(item.get("generation_parameters", [])),
            )
        )
    return sorted(candidates, key=lambda c: (c.priority, c.name.lower()))


def get_candidate(candidate_id: str) -> Candidate:
    for candidate in load_candidates():
        if candidate.id == candidate_id:
            return candidate
    raise KeyError(f"Unknown candidate: {candidate_id}")
