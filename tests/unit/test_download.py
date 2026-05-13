"""Tests for download.py — Phase 4 Task 4.1."""
from __future__ import annotations
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from minicpm_v_local import download, paths


@pytest.fixture
def fake_paths(tmp_path, monkeypatch):
    lock = tmp_path / "run" / "download.lock"
    cache_root = tmp_path / "cache"
    monkeypatch.setattr(paths, "download_lock", lambda: lock)
    monkeypatch.setattr(paths, "cache_dir", lambda backend: cache_root / backend)
    return {"lock": lock, "cache_root": cache_root}


def test_ensure_model_returns_target_path_and_calls_snapshot(fake_paths):
    with patch.object(download, "snapshot_download") as mock_dl:
        result = download.ensure_model("openbmb/MiniCPM-V-4_6", "mlx")
    expected = fake_paths["cache_root"] / "mlx" / "openbmb__MiniCPM-V-4_6"
    assert result == expected
    assert expected.is_dir()
    mock_dl.assert_called_once()
    kwargs = mock_dl.call_args.kwargs
    assert kwargs["repo_id"] == "openbmb/MiniCPM-V-4_6"
    assert kwargs["local_dir"] == str(expected)
    assert kwargs["local_dir_use_symlinks"] is False


def test_ensure_model_passes_allow_patterns(fake_paths):
    with patch.object(download, "snapshot_download") as mock_dl:
        download.ensure_model("foo/bar", "cpu", allow_patterns=["*.gguf"])
    assert mock_dl.call_args.kwargs["allow_patterns"] == ["*.gguf"]


def test_lock_file_created(fake_paths):
    with patch.object(download, "snapshot_download"):
        download.ensure_model("a/b", "mlx")
    assert fake_paths["lock"].exists()


def test_concurrent_calls_serialized(fake_paths):
    """Two threads racing ensure_model — flock must serialize them."""
    in_critical = []
    max_concurrent = [0]
    lock_check = threading.Lock()

    def slow_snapshot(**kwargs):
        with lock_check:
            in_critical.append(1)
            max_concurrent[0] = max(max_concurrent[0], sum(in_critical))
        time.sleep(0.1)
        with lock_check:
            in_critical.pop()

    with patch.object(download, "snapshot_download", side_effect=slow_snapshot):
        t1 = threading.Thread(target=download.ensure_model, args=("x/y", "mlx"))
        t2 = threading.Thread(target=download.ensure_model, args=("x/y", "mlx"))
        t1.start(); t2.start()
        t1.join(); t2.join()

    assert max_concurrent[0] == 1, "flock LOCK_EX must serialize concurrent downloads"
