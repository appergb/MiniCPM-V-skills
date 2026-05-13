"""OpenAI-compatible HTTP client. Spec §10."""
from __future__ import annotations
import base64
from pathlib import Path
from typing import Optional

import httpx


class VLMClient:
    def __init__(self, base_url: str, timeout: float = 180.0):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(timeout=timeout)

    @staticmethod
    def _encode_image(path: Path) -> str:
        suffix = path.suffix.lstrip(".").lower() or "jpeg"
        if suffix == "jpg":
            suffix = "jpeg"
        b64 = base64.b64encode(path.read_bytes()).decode()
        return f"data:image/{suffix};base64,{b64}"

    def caption(self, image: Path, prompt: str, *, model: str = "minicpm-v") -> str:
        payload = {
            "model": model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": self._encode_image(image)}},
                    {"type": "text", "text": prompt},
                ],
            }],
            "max_tokens": 512,
        }
        r = self._client.post(f"{self.base_url}/v1/chat/completions", json=payload)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    def close(self) -> None:
        self._client.close()
