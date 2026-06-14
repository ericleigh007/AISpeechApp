from __future__ import annotations

import os
from pathlib import Path


DEFAULT_MODEL_ROOT = Path("D:/AI/Models")


def configure_model_caches(model_root: Path = DEFAULT_MODEL_ROOT) -> dict[str, str]:
    """Route heavyweight model caches away from the system drive."""
    model_root = model_root.resolve()
    paths = {
        "AI_SPEECH_MODEL_ROOT": model_root,
        "HF_HOME": model_root / "huggingface",
        "HUGGINGFACE_HUB_CACHE": model_root / "huggingface" / "hub",
        "TRANSFORMERS_CACHE": model_root / "huggingface" / "hub",
        "HF_XET_CACHE": model_root / "huggingface" / "xet",
        "MODELSCOPE_CACHE": model_root / "modelscope",
        "TORCH_HOME": model_root / "torch",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)

    env = {key: str(path) for key, path in paths.items()}
    os.environ.update(env)
    return env
