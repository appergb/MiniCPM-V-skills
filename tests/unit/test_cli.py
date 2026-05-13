"""Tests for cli.py — SYNTHESIZED dispatch tests for Phase 7."""
from __future__ import annotations
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from minicpm_v_local import cli


def _fake_state(**overrides):
    s = MagicMock()
    s.port = overrides.get("port", 8765)
    s.alive = overrides.get("alive", True)
    s.to_dict.return_value = overrides.get("dict", {"alive": True, "port": 8765})
    return s


@pytest.fixture
def patched_runtime(tmp_path, monkeypatch):
    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text(
        'backend = "mlx"\nquant = "4bit"\nidle_timeout = 300\n'
        'max_lifetime = 1800\nisolation = false\nisolation_mode = "auto"\n'
    )
    monkeypatch.setattr(cli.paths, "config_file", lambda: cfg_file)
    monkeypatch.setattr(cli.paths, "state_file", lambda: tmp_path / "state.json")
    monkeypatch.setattr(cli.detect, "resolve", lambda b: "mlx")

    fake_backend = MagicMock()
    fake_backend.tag = "mlx"
    fake_backend.artifact_id.return_value = "mlx-community/MiniCPM-V-4.6-4bit"
    monkeypatch.setattr(cli, "get_backend", lambda tag, quant: fake_backend)

    monkeypatch.setattr(cli, "ensure_model", lambda repo, backend: tmp_path / "model")
    monkeypatch.setattr(cli.manager, "ensure_warm",
                        lambda *a, **kw: _fake_state(port=8765))
    fake_client = MagicMock()
    monkeypatch.setattr(cli, "VLMClient", lambda **kw: fake_client)
    return {"cfg_file": cfg_file, "fake_backend": fake_backend, "fake_client": fake_client}


def test_image_command_returns_pipeline_json(patched_runtime, monkeypatch, capsys, tmp_path):
    expected = {"version": 1, "result": {"caption": "a cat"}}
    monkeypatch.setattr(cli, "caption_image", lambda c, p, *, model, prompt: expected)
    img = tmp_path / "x.jpg"
    img.write_bytes(b"data")

    rc = cli.main(["image", str(img)])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out == expected


def test_video_command_returns_pipeline_json(patched_runtime, monkeypatch, capsys, tmp_path):
    expected = {"version": 1, "frames": [], "scenes": []}
    monkeypatch.setattr(cli, "process_video",
                        lambda c, p, *, model, cfg, prompt: expected)
    vid = tmp_path / "v.mp4"
    vid.write_bytes(b"data")

    rc = cli.main(["video", str(vid)])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out == expected


def test_ttl_zero_calls_stop_after_pipeline(patched_runtime, monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "caption_image", lambda *a, **kw: {"ok": True})
    stop_calls = []
    monkeypatch.setattr(cli.manager, "stop", lambda *a, **kw: stop_calls.append(kw))
    img = tmp_path / "x.jpg"
    img.write_bytes(b"d")

    cli.main(["image", str(img), "--ttl", "0"])
    assert len(stop_calls) == 1


def test_keep_flag_propagates_to_ensure_warm(patched_runtime, monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "caption_image", lambda *a, **kw: {"ok": True})
    received = {}
    def fake_warm(*a, **kw):
        received.update(kw)
        return _fake_state(port=8765)
    monkeypatch.setattr(cli.manager, "ensure_warm", fake_warm)
    img = tmp_path / "x.jpg"
    img.write_bytes(b"d")

    cli.main(["image", str(img), "--keep"])
    assert received["keep"] is True


def test_missing_config_runs_doctor(monkeypatch, tmp_path):
    monkeypatch.setattr(cli.paths, "config_file", lambda: tmp_path / "absent.toml")
    monkeypatch.setattr(cli.paths, "state_file", lambda: tmp_path / "state.json")
    doctor_calls = []
    monkeypatch.setattr(cli.doctor, "run", lambda: doctor_calls.append(1) or 0)
    img = tmp_path / "x.jpg"; img.write_bytes(b"d")

    rc = cli.main(["image", str(img)])
    assert rc == 0
    assert doctor_calls == [1]


def test_doctor_reset_unlinks_config(monkeypatch, tmp_path):
    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text("x = 1\n")
    monkeypatch.setattr(cli.paths, "config_file", lambda: cfg_file)
    monkeypatch.setattr(cli.doctor, "run", lambda: 0)

    rc = cli.main(["doctor", "--reset"])
    assert rc == 0
    assert not cfg_file.exists()


def test_status_prints_alive_false_when_no_state(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(cli.paths, "state_file", lambda: tmp_path / "absent.json")
    monkeypatch.setattr(cli, "read_state", lambda p: None)

    rc = cli.main(["status"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out == {"alive": False}


def test_stop_force_calls_manager_stop(monkeypatch, tmp_path):
    received = {}
    monkeypatch.setattr(cli.manager, "stop",
                        lambda force=False: received.update(force=force))
    rc = cli.main(["stop", "--force"])
    assert rc == 0
    assert received == {"force": True}
