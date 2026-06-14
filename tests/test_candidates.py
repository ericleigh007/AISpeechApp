from aispeechapp.candidates import load_candidates


def test_candidates_have_unique_ids():
    candidates = load_candidates()
    ids = [candidate.id for candidate in candidates]
    assert len(ids) == len(set(ids))


def test_priority_one_candidates_exist():
    candidates = load_candidates()
    priority_one = [candidate for candidate in candidates if candidate.priority == 1]
    assert {candidate.id for candidate in priority_one} >= {
        "qwen3_tts_17b_customvoice",
        "voxcpm2",
        "indextts2",
    }

