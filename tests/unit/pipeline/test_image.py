"""Tests for pipeline.image. SYNTHESIZED from plan bullets."""
from __future__ import annotations
import hashlib
from pathlib import Path
from unittest.mock import MagicMock

from minicpm_v_local.client import VLMClient
from minicpm_v_local.pipeline.image import caption_image, DEFAULT_PROMPT


def test_caption_image_returns_spec_shape(tmp_path: Path):
    # Arrange
    img = tmp_path / "foo.jpg"
    img.write_bytes(b"\xff\xd8\xff\xe0fake-jpeg-bytes")
    expected_sha = hashlib.sha256(img.read_bytes()).hexdigest()

    client = MagicMock(spec=VLMClient)
    client.caption.return_value = "A red bicycle."

    # Act
    out = caption_image(client, img, model="mlx-community/MiniCPM-V-4.6-4bit")

    # Assert
    assert out["version"] == 1
    assert out["input"] == {"path": str(img), "sha256": expected_sha}
    assert out["model"] == "mlx-community/MiniCPM-V-4.6-4bit"
    assert out["result"]["caption"] == "A red bicycle."
    assert out["result"]["objects"] == []
    assert out["result"]["ocr_text"] is None
    assert "load" in out["timing_ms"] and "infer" in out["timing_ms"]
    assert isinstance(out["timing_ms"]["infer"], int)
    client.caption.assert_called_once_with(img, prompt=DEFAULT_PROMPT, model="mlx-community/MiniCPM-V-4.6-4bit")


def test_caption_image_custom_prompt(tmp_path: Path):
    img = tmp_path / "x.png"
    img.write_bytes(b"data")
    client = MagicMock(spec=VLMClient)
    client.caption.return_value = "OCR text: hi"

    out = caption_image(client, img, model="m", prompt="OCR this")

    client.caption.assert_called_once_with(img, prompt="OCR this", model="m")
    assert out["result"]["caption"] == "OCR text: hi"


def test_caption_image_served_model_overrides_wire_model(tmp_path):
    img = tmp_path / "y.jpg"
    img.write_bytes(b"x")
    client = MagicMock(spec=VLMClient)
    client.caption.return_value = "a thing"

    out = caption_image(client, img,
                        model="repo/published-name",
                        served_model="/local/path/to/model")

    # JSON envelope reports the public model id
    assert out["model"] == "repo/published-name"
    # but the wire call uses the local path
    client.caption.assert_called_once_with(img, prompt=DEFAULT_PROMPT,
                                            model="/local/path/to/model")


def test_default_prompt_constant_exported():
    assert isinstance(DEFAULT_PROMPT, str) and len(DEFAULT_PROMPT) > 0


def test_resolve_prompt_default():
    from minicpm_v_local.pipeline.image import resolve_prompt, DEFAULT_PROMPT
    assert resolve_prompt(None, None) == DEFAULT_PROMPT


def test_resolve_prompt_explicit_overrides_preset():
    from minicpm_v_local.pipeline.image import resolve_prompt
    out = resolve_prompt("just describe it", "ui")
    assert out == "just describe it"


def test_resolve_prompt_preset_lookup():
    from minicpm_v_local.pipeline.image import resolve_prompt, PROMPT_PRESETS
    for name in ("ui", "photo", "doc", "chart"):
        assert resolve_prompt(None, name) == PROMPT_PRESETS[name]


def test_resolve_prompt_unknown_preset_raises():
    from minicpm_v_local.pipeline.image import resolve_prompt
    import pytest as _pytest
    with _pytest.raises(ValueError, match="unknown prompt preset"):
        resolve_prompt(None, "nonexistent")


def test_default_prompt_is_chinese_structured():
    from minicpm_v_local.pipeline.image import DEFAULT_PROMPT
    # v0.1.4: Chinese, structured (4-step numbered list)
    assert "请详细描述" in DEFAULT_PROMPT
    assert "1." in DEFAULT_PROMPT and "4." in DEFAULT_PROMPT
