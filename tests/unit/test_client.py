"""Tests for client.py — Phase 4 Task 4.2."""
from __future__ import annotations
import json
from pathlib import Path

import httpx
import pytest

from minicpm_v_local.client import VLMClient


def _make_mock_response(captured: dict):
    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content.decode())
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "a red bicycle"}}]
            },
        )
    return handler


def test_caption_posts_openai_payload_and_returns_content(tmp_path):
    img = tmp_path / "pic.jpg"
    img.write_bytes(b"\xff\xd8\xff\xe0fakejpegbytes")
    captured: dict = {}
    transport = httpx.MockTransport(_make_mock_response(captured))

    client = VLMClient("http://127.0.0.1:8765")
    client._client = httpx.Client(transport=transport)

    out = client.caption(img, prompt="Describe.", model="minicpm-v")

    assert out == "a red bicycle"
    assert captured["url"].endswith("/v1/chat/completions")
    body = captured["body"]
    assert body["model"] == "minicpm-v"
    assert body["max_tokens"] == 512
    parts = body["messages"][0]["content"]
    assert parts[0]["type"] == "image_url"
    assert parts[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert parts[1] == {"type": "text", "text": "Describe."}


def test_encode_image_normalizes_jpg_to_jpeg(tmp_path):
    img = tmp_path / "x.jpg"
    img.write_bytes(b"data")
    url = VLMClient._encode_image(img)
    assert url.startswith("data:image/jpeg;base64,")


def test_encode_image_keeps_png(tmp_path):
    img = tmp_path / "x.png"
    img.write_bytes(b"data")
    url = VLMClient._encode_image(img)
    assert url.startswith("data:image/png;base64,")


def test_base_url_trailing_slash_stripped():
    c = VLMClient("http://127.0.0.1:8765/")
    assert c.base_url == "http://127.0.0.1:8765"
    c.close()


def test_caption_raises_on_http_error(tmp_path):
    img = tmp_path / "p.jpg"
    img.write_bytes(b"x")
    transport = httpx.MockTransport(lambda req: httpx.Response(500, json={"error": "boom"}))
    client = VLMClient("http://127.0.0.1:8765")
    client._client = httpx.Client(transport=transport)
    with pytest.raises(httpx.HTTPStatusError):
        client.caption(img, prompt="x")
