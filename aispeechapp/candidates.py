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
    model_scale: str = ""
    notes: str = ""
    generation_parameters: tuple[dict[str, Any], ...] = ()
    text_controls: tuple[dict[str, Any], ...] = ()


def load_candidates(path: Path = DEFAULT_CANDIDATES_PATH) -> list[Candidate]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw_by_id = {item["id"]: item for item in raw["candidates"]}

    def inherited_tuple(item: dict[str, Any], field: str) -> tuple[dict[str, Any], ...]:
        inherited_from = item.get(f"inherits_{field}")
        inherited: list[dict[str, Any]] = []
        if isinstance(inherited_from, str):
            inherited = list(raw_by_id[inherited_from].get(field, []))
        return tuple(inherited + list(item.get(field, [])))

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
                model_scale=item.get("model_scale", ""),
                notes=item.get("notes", ""),
                generation_parameters=inherited_tuple(item, "generation_parameters"),
                text_controls=inherited_tuple(item, "text_controls"),
            )
        )
    return sorted(candidates, key=lambda c: (c.priority, c.name.lower()))


def get_candidate(candidate_id: str) -> Candidate:
    for candidate in load_candidates():
        if candidate.id == candidate_id:
            return candidate
    raise KeyError(f"Unknown candidate: {candidate_id}")
