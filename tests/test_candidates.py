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
        "voxcpm2_quantized",
        "voxcpm2_torchao_int8",
        "voxcpm2_bnb_int8",
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
            assert len(parameter.get("description", "")) >= 40, (candidate.id, parameter["id"])
            if parameter["type"] == "choice":
                assert parameter.get("choices")
                assert parameter["default"] in parameter["choices"]


def test_text_controls_are_gui_renderable():
    valid_modes = {"parenthesized_prefix", "generation_option"}
    for candidate in load_candidates():
        control_ids = [control["id"] for control in candidate.text_controls]
        assert len(control_ids) == len(set(control_ids)), candidate.id
        for control in candidate.text_controls:
            assert control["mode"] in valid_modes
            assert "label" in control
            assert "default" in control
            assert len(control.get("description", "")) >= 40, (candidate.id, control["id"])


def test_generation_parameter_help_mentions_tradeoffs_for_key_controls():
    candidates = {candidate.id: candidate for candidate in load_candidates()}
    voxcpm2_parameters = {
        parameter["id"]: parameter["description"].lower()
        for parameter in candidates["voxcpm2"].generation_parameters
    }
    vibevoice_parameters = {
        parameter["id"]: parameter["description"].lower()
        for parameter in candidates["microsoft_vibevoice_15b"].generation_parameters
    }

    assert "latency" in voxcpm2_parameters["inference_timesteps"]
    assert "pacing" in voxcpm2_parameters["seed"]
    assert "ringing" in voxcpm2_parameters["cfg_value"]
    assert "windows" in voxcpm2_parameters["torch_compile"]
    assert "generation time" in vibevoice_parameters["ddpm_steps"]


def test_voxcpm2_declares_control_instruction_text_control():
    voxcpm2 = next(candidate for candidate in load_candidates() if candidate.id == "voxcpm2")
    controls = {control["id"]: control for control in voxcpm2.text_controls}

    assert controls["control_instruction"]["mode"] == "parenthesized_prefix"
    assert "pace" in controls["control_instruction"]["description"].lower()


def test_voxcpm2_declares_variability_controls():
    voxcpm2 = next(candidate for candidate in load_candidates() if candidate.id == "voxcpm2")
    parameter_ids = {parameter["id"] for parameter in voxcpm2.generation_parameters}
    assert {
        "cfg_value",
        "inference_timesteps",
        "seed",
        "normalize",
        "denoise",
        "retry_badcase",
        "torch_compile",
        "torch_compile_mode",
    } <= parameter_ids


def test_voxcpm2_quantized_inherits_baseline_controls_and_adds_quantization():
    quantized = next(candidate for candidate in load_candidates() if candidate.id == "voxcpm2_quantized")
    parameter_ids = {parameter["id"] for parameter in quantized.generation_parameters}
    text_control_ids = {control["id"] for control in quantized.text_controls}

    assert "control_instruction" in text_control_ids
    assert {
        "cfg_value",
        "inference_timesteps",
        "seed",
        "quantization_method",
        "quantization_targets",
    } <= parameter_ids
    assert "quantized" in quantized.capabilities


def test_voxcpm2_torchao_int8_declares_real_cuda_quantization_candidate():
    candidate = next(candidate for candidate in load_candidates() if candidate.id == "voxcpm2_torchao_int8")
    parameter_ids = {parameter["id"] for parameter in candidate.generation_parameters}
    quantization_method = next(
        parameter for parameter in candidate.generation_parameters if parameter["id"] == "quantization_method"
    )

    assert "torchao" in candidate.expected_imports
    assert "cuda_quantization" in candidate.capabilities
    assert {"cfg_value", "inference_timesteps", "seed", "quantization_method"} <= parameter_ids
    assert quantization_method["default"] == "torchao_int8_weight_only"


def test_voxcpm2_bnb_int8_declares_real_cuda_quantization_candidate():
    candidate = next(candidate for candidate in load_candidates() if candidate.id == "voxcpm2_bnb_int8")
    parameter_ids = {parameter["id"] for parameter in candidate.generation_parameters}
    quantization_method = next(
        parameter for parameter in candidate.generation_parameters if parameter["id"] == "quantization_method"
    )

    assert "bitsandbytes" in candidate.expected_imports
    assert "cuda_quantization" in candidate.capabilities
    assert {"cfg_value", "inference_timesteps", "seed", "quantization_method"} <= parameter_ids
    assert quantization_method["default"] == "bnb_int8_linear"


def test_priority_one_candidates_include_known_scale_metadata():
    scales = {
        candidate.id: candidate.model_scale
        for candidate in load_candidates()
        if candidate.id in {
            "qwen3_tts_17b_customvoice",
            "voxcpm2",
            "voxcpm2_quantized",
            "voxcpm2_torchao_int8",
            "voxcpm2_bnb_int8",
            "dots_tts_soar",
        }
    }

    assert scales == {
        "dots_tts_soar": "2B",
        "qwen3_tts_17b_customvoice": "1.7B",
        "voxcpm2": "2B",
        "voxcpm2_quantized": "2B",
        "voxcpm2_torchao_int8": "2B",
        "voxcpm2_bnb_int8": "2B",
    }
