"""Single-image pipeline. Spec §10.3."""
from __future__ import annotations
import hashlib
import time
from pathlib import Path

from minicpm_v_local.client import VLMClient

DEFAULT_PROMPT = "Describe the image in detail. List any visible objects and text."


def caption_image(client: VLMClient, image_path: Path, *,
                  model: str, prompt: str = DEFAULT_PROMPT) -> dict:
    t0 = time.monotonic()
    sha = hashlib.sha256(image_path.read_bytes()).hexdigest()
    text = client.caption(image_path, prompt=prompt, model=model)
    dt = int((time.monotonic() - t0) * 1000)
    return {
        "version": 1,
        "input": {"path": str(image_path), "sha256": sha},
        "model": model,
        "result": {"caption": text, "objects": [], "ocr_text": None},
        "timing_ms": {"load": 0, "infer": dt},
    }
