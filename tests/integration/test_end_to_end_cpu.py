"""End-to-end pipeline test using httpx.MockTransport (no real server).

Spec §15.2 (integration): image / video pipeline JSON shape via mocked client.
"""
from __future__ import annotations
import json
import shutil
from pathlib import Path

import httpx
import pytest

from minicpm_v_local.client import VLMClient
from minicpm_v_local.config import VideoConfig
from minicpm_v_local.pipeline.image import caption_image
from minicpm_v_local.pipeline.video import process_video


FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
SAMPLE_IMG = FIXTURES / "sample.jpg"
SAMPLE_VID = FIXTURES / "sample-5s.mp4"


def _fake_caption_response(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    text_parts = [
        c.get("text", "") for m in body["messages"]
        for c in m["content"] if c.get("type") == "text"
    ]
    suffix = text_parts[0][:20] if text_parts else "frame"
    return httpx.Response(
        200,
        json={
            "choices": [
                {"message": {"content": f"a red square :: {suffix}"}}
            ]
        },
    )


@pytest.fixture
def mock_client() -> VLMClient:
    transport = httpx.MockTransport(_fake_caption_response)
    client = VLMClient(base_url="http://127.0.0.1:9999")
    client._client = httpx.Client(transport=transport)
    return client


def test_image_end_to_end(mock_client):
    assert SAMPLE_IMG.exists(), "run fixture generator first"
    result = caption_image(mock_client, SAMPLE_IMG, model="fake/model")
    assert result["version"] == 1
    assert result["input"]["path"].endswith("sample.jpg")
    assert len(result["input"]["sha256"]) == 64
    assert "a red square" in result["result"]["caption"]
    assert "load" in result["timing_ms"] and "infer" in result["timing_ms"]


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe required for video pipeline",
)
def test_video_end_to_end(mock_client, tmp_path, monkeypatch):
    assert SAMPLE_VID.exists(), "run fixture generator first"
    from minicpm_v_local import paths
    monkeypatch.setattr(paths, "frames_tmp_dir",
                        lambda run_id: tmp_path / f"frames-{run_id}")
    cfg = VideoConfig()
    result = process_video(mock_client, SAMPLE_VID, model="fake/model", cfg=cfg)
    assert result["version"] == 1
    assert result["input"]["duration_s"] > 0
    assert len(result["frames"]) >= 1
    assert len(result["scenes"]) >= 1
    assert all("caption" in fr for fr in result["frames"])
