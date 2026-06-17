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


def test_generation_parameters_are_gui_renderable():
    valid_types = {"bool", "choice", "float", "int"}
    for candidate in load_candidates():
        parameter_ids = [parameter["id"] for parameter in candidate.generation_parameters]
        assert len(parameter_ids) == len(set(parameter_ids)), candidate.id
        for parameter in candidate.generation_parameters:
            assert parameter["type"] in valid_types
            assert "label" in parameter
            assert "default" in parameter
            if parameter["type"] == "choice":
                assert parameter.get("choices")
                assert parameter["default"] in parameter["choices"]


def test_voxcpm2_declares_variability_controls():
    voxcpm2 = next(candidate for candidate in load_candidates() if candidate.id == "voxcpm2")
    parameter_ids = {parameter["id"] for parameter in voxcpm2.generation_parameters}
    assert {
        "cfg_value",
        "inference_timesteps",
        "normalize",
        "denoise",
        "retry_badcase",
    } <= parameter_ids
