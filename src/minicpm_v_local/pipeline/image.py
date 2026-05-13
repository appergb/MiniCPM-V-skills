"""Single-image pipeline. Spec §10.3."""
from __future__ import annotations
import hashlib
import time
from pathlib import Path

from minicpm_v_local.client import VLMClient

DEFAULT_PROMPT = "Describe the image in detail. List any visible objects and text."


def caption_image(client: VLMClient, image_path: Path, *,
                  model: str, prompt: str = DEFAULT_PROMPT,
                  served_model: str | None = None) -> dict:
    """`model` = identifier reported in the JSON envelope (e.g. HF repo ID).
    `served_model` = exact name to send in the HTTP `model` field; must match
    what the server preloaded. If None, falls back to `model`."""
    t0 = time.monotonic()
    sha = hashlib.sha256(image_path.read_bytes()).hexdigest()
    text = client.caption(image_path, prompt=prompt, model=served_model or model)
    dt = int((time.monotonic() - t0) * 1000)
    return {
        "version": 1,
        "input": {"path": str(image_path), "sha256": sha},
        "model": model,
        "result": {"caption": text, "objects": [], "ocr_text": None},
        "timing_ms": {"load": 0, "infer": dt},
    }
