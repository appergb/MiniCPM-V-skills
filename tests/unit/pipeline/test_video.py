"""Tests for pipeline.video. SYNTHESIZED."""
from __future__ import annotations
import json
import subprocess as _subproc
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from minicpm_v_local.client import VLMClient
from minicpm_v_local.config import VideoConfig
from minicpm_v_local.pipeline import video as vid
from minicpm_v_local.pipeline.video import (
    Frame, DEFAULT_PROMPT, probe, extract_keyframes, merge_scenes, process_video,
)


def _cfg(**overrides):
    base = dict(scene_threshold=0.3, fallback_interval=10.0, max_frames=60,
                scene_merge_similarity=0.85)
    base.update(overrides)
    return VideoConfig(**base)


def test_default_prompt_exported():
    assert isinstance(DEFAULT_PROMPT, str) and DEFAULT_PROMPT


def test_frame_dataclass_defaults():
    f = Frame(t=1.0, path=Path("/x.jpg"))
    assert f.caption is None and f.error is None


def test_probe_parses_ffprobe_json(monkeypatch, tmp_path):
    payload = {"format": {"duration": "12.5"},
               "streams": [{"r_frame_rate": "30000/1001"}]}
    def fake_run(*a, **kw):
        return _subproc.CompletedProcess(a, 0, stdout=json.dumps(payload), stderr="")
    monkeypatch.setattr(vid.subprocess, "run", fake_run)
    out = probe(tmp_path / "v.mp4")
    assert out["duration"] == pytest.approx(12.5)
    assert out["fps"] == pytest.approx(30000 / 1001)


def test_probe_zero_denominator_falls_back_to_30(monkeypatch, tmp_path):
    payload = {"format": {"duration": "1.0"},
               "streams": [{"r_frame_rate": "0/0"}]}
    monkeypatch.setattr(vid.subprocess, "run",
        lambda *a, **kw: _subproc.CompletedProcess(a, 0, stdout=json.dumps(payload), stderr=""))
    assert probe(tmp_path / "v.mp4")["fps"] == 30.0


def test_extract_keyframes_parses_showinfo(monkeypatch, tmp_path):
    frames_dir = tmp_path / "frames"
    monkeypatch.setattr(vid.paths, "frames_tmp_dir", lambda rid: frames_dir)

    stderr = "frame: pts_time:1.5 ...\nframe: pts_time:5.2 ...\n"
    calls = []
    def fake_run(args, capture_output=False, text=False, check=False):
        calls.append(args)
        # First call = scene-detect; write fake outputs
        if "select=" in " ".join(args) or any("scene" in a for a in args):
            frames_dir.mkdir(parents=True, exist_ok=True)
            (frames_dir / "scene_0001.jpg").write_bytes(b"j")
            (frames_dir / "scene_0002.jpg").write_bytes(b"j")
            return _subproc.CompletedProcess(args, 0, stdout="", stderr=stderr)
        if args[0] == "ffprobe":
            payload = {"format": {"duration": "20.0"}, "streams": [{"r_frame_rate": "30/1"}]}
            return _subproc.CompletedProcess(args, 0, stdout=json.dumps(payload), stderr="")
        # uniform fallback writes
        out = Path(args[-1]); out.write_bytes(b"j")
        return _subproc.CompletedProcess(args, 0, stdout="", stderr="")
    monkeypatch.setattr(vid.subprocess, "run", fake_run)

    frames = extract_keyframes(tmp_path / "v.mp4", cfg=_cfg())
    assert len(frames) >= 2
    assert frames[0].t <= frames[-1].t  # sorted


def test_extract_keyframes_downsamples_to_max_frames(monkeypatch, tmp_path):
    frames_dir = tmp_path / "frames"
    monkeypatch.setattr(vid.paths, "frames_tmp_dir", lambda rid: frames_dir)
    stderr_lines = "\n".join(f"frame pts_time:{i}.0" for i in range(100))
    def fake_run(args, capture_output=False, text=False, check=False):
        if args[0] == "ffprobe":
            return _subproc.CompletedProcess(args, 0,
                stdout=json.dumps({"format": {"duration": "100.0"},
                                   "streams": [{"r_frame_rate": "30/1"}]}), stderr="")
        frames_dir.mkdir(parents=True, exist_ok=True)
        if any("select=" in a for a in args):
            for i in range(100):
                (frames_dir / f"scene_{i:04d}.jpg").write_bytes(b"j")
            return _subproc.CompletedProcess(args, 0, stdout="", stderr=stderr_lines)
        Path(args[-1]).write_bytes(b"j")
        return _subproc.CompletedProcess(args, 0, stdout="", stderr="")
    monkeypatch.setattr(vid.subprocess, "run", fake_run)

    frames = extract_keyframes(tmp_path / "v.mp4", cfg=_cfg(max_frames=10))
    assert len(frames) == 10


def test_extract_keyframes_short_video_gets_at_least_one_frame(monkeypatch, tmp_path):
    """Short clips (duration < fallback_interval) must still yield ≥1 frame.

    Regression for v0.1.2: target_count=int(5/10)=0 used to leave frames empty.
    """
    frames_dir = tmp_path / "frames"
    monkeypatch.setattr(vid.paths, "frames_tmp_dir", lambda rid: frames_dir)

    def fake_run(args, capture_output=False, text=False, check=False):
        if args[0] == "ffprobe":
            return _subproc.CompletedProcess(args, 0,
                stdout=json.dumps({"format": {"duration": "5.0"},
                                   "streams": [{"r_frame_rate": "30/1"}]}),
                stderr="")
        # Both scene-detect and the "guaranteed t=0 frame" code paths write here.
        frames_dir.mkdir(parents=True, exist_ok=True)
        if any("select=" in a for a in args):
            # scene-detect finds nothing — uniform testsrc
            return _subproc.CompletedProcess(args, 0, stdout="", stderr="")
        # Single-frame extract: write the requested output path
        Path(args[-1]).write_bytes(b"j")
        return _subproc.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(vid.subprocess, "run", fake_run)

    frames = extract_keyframes(tmp_path / "short.mp4",
                               cfg=_cfg(fallback_interval=10.0, max_frames=60))
    assert len(frames) == 1
    assert frames[0].t == 0.0


def test_merge_scenes_identical_captions_merge():
    frames = [Frame(t=i, path=Path(f"/{i}.jpg"), caption="a cat") for i in range(3)]
    scenes = merge_scenes(frames, threshold=0.85)
    assert len(scenes) == 1
    assert scenes[0]["frame_indices"] == [0, 1, 2]
    assert "_tokens" not in scenes[0]


def test_merge_scenes_disjoint_captions_split():
    frames = [
        Frame(t=0, path=Path("/0.jpg"), caption="a cat"),
        Frame(t=1, path=Path("/1.jpg"), caption="red bicycle wall"),
        Frame(t=2, path=Path("/2.jpg"), caption="a cat"),
    ]
    scenes = merge_scenes(frames, threshold=0.85)
    assert len(scenes) == 3


def test_merge_scenes_empty_returns_empty():
    assert merge_scenes([], threshold=0.85) == []


def test_process_video_frame_error_does_not_break(monkeypatch, tmp_path):
    video = tmp_path / "v.mp4"; video.write_bytes(b"fake-video")
    frames_dir = tmp_path / "frames"
    monkeypatch.setattr(vid.paths, "frames_tmp_dir", lambda rid: frames_dir)

    def fake_run(args, capture_output=False, text=False, check=False):
        if args[0] == "ffprobe":
            return _subproc.CompletedProcess(args, 0,
                stdout=json.dumps({"format": {"duration": "5.0"},
                                   "streams": [{"r_frame_rate": "30/1"}]}), stderr="")
        frames_dir.mkdir(parents=True, exist_ok=True)
        (frames_dir / "scene_0001.jpg").write_bytes(b"j")
        return _subproc.CompletedProcess(args, 0, stdout="",
            stderr="frame pts_time:1.0\n")
    monkeypatch.setattr(vid.subprocess, "run", fake_run)

    client = MagicMock(spec=VLMClient)
    client.caption.side_effect = [RuntimeError("boom"), "ok caption"]

    out = process_video(client, video, model="m", cfg=_cfg(max_frames=2))
    assert out["version"] == 1
    assert out["input"]["duration_s"] == 5.0
    assert any(f["error"] for f in out["frames"])
    assert "ffmpeg" in out["timing_ms"]
    assert "infer_total" in out["timing_ms"]


def test_process_video_on_frame_done_callback(monkeypatch, tmp_path):
    video = tmp_path / "v.mp4"; video.write_bytes(b"v")
    frames_dir = tmp_path / "frames"
    monkeypatch.setattr(vid.paths, "frames_tmp_dir", lambda rid: frames_dir)
    def fake_run(args, **kw):
        if args[0] == "ffprobe":
            return _subproc.CompletedProcess(args, 0,
                stdout=json.dumps({"format": {"duration": "1.0"},
                                   "streams": [{"r_frame_rate": "30/1"}]}), stderr="")
        frames_dir.mkdir(parents=True, exist_ok=True)
        (frames_dir / "scene_0001.jpg").write_bytes(b"j")
        return _subproc.CompletedProcess(args, 0, stdout="", stderr="frame pts_time:0.5\n")
    monkeypatch.setattr(vid.subprocess, "run", fake_run)
    client = MagicMock(spec=VLMClient); client.caption.return_value = "x"
    seen = []
    process_video(client, video, model="m", cfg=_cfg(max_frames=5),
                  on_frame_done=lambda f: seen.append(f))
    assert len(seen) >= 1


def test_process_video_served_model_overrides_wire_model(monkeypatch, tmp_path):
    video = tmp_path / "v.mp4"; video.write_bytes(b"v")
    frames_dir = tmp_path / "frames"
    monkeypatch.setattr(vid.paths, "frames_tmp_dir", lambda rid: frames_dir)
    def fake_run(args, **kw):
        if args[0] == "ffprobe":
            return _subproc.CompletedProcess(args, 0,
                stdout=json.dumps({"format": {"duration": "1.0"},
                                   "streams": [{"r_frame_rate": "30/1"}]}), stderr="")
        frames_dir.mkdir(parents=True, exist_ok=True)
        (frames_dir / "scene_0001.jpg").write_bytes(b"j")
        return _subproc.CompletedProcess(args, 0, stdout="", stderr="frame pts_time:0.5\n")
    monkeypatch.setattr(vid.subprocess, "run", fake_run)
    client = MagicMock(spec=VLMClient); client.caption.return_value = "x"

    out = process_video(client, video, model="repo/published-name",
                        served_model="/local/path/to/model",
                        cfg=_cfg(max_frames=2))

    # JSON envelope reports the public model id
    assert out["model"] == "repo/published-name"
    # but every wire call uses the local path
    for call in client.caption.call_args_list:
        assert call.kwargs.get("model") == "/local/path/to/model"
