from aispeechapp.candidates import Candidate
from aispeechapp.smoke import run_metadata_smoke


def test_metadata_smoke_handles_missing_repo():
    candidate = Candidate(
        id="missing",
        name="Missing",
        priority=99,
        repo_id="not-a-real-org/not-a-real-model-for-aispeechapp",
        backend="missing",
        expected_imports=("definitely_missing_module_for_aispeechapp",),
        capabilities=(),
    )
    result = run_metadata_smoke(candidate)
    assert result.candidate_id == "missing"
    assert result.status in {"metadata_failed", "needs_install"}

