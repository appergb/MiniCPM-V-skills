# tests/unit/server/test_manager.py
# SYNTHESIZED from "测试要点".
from __future__ import annotations
import signal
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from minicpm_v_local.server import manager
from minicpm_v_local.server.state import State, read_state, write_state


class FakeProc:
    def __init__(self, pid: int) -> None:
        self.pid = pid
        self.terminated = False

    def terminate(self) -> None:
        self.terminated = True


class FakeBackend:
    tag = "mlx"

    def launch_cmd(self, model_dir: str, port: int) -> list[str]:
        return ["python", "-m", "mlx_vlm.server", "--model", model_dir, "--port", str(port)]

    def health_path(self) -> str:
        return "/health"

    def artifact_id(self) -> str:
        return "mlx-community/MiniCPM-V-4.6-4bit"


@pytest.fixture
def patched_paths(tmp_path, monkeypatch):
    state_path = tmp_path / "state.json"
    log_dir = tmp_path / "logs"
    monkeypatch.setattr(manager.paths, "state_file", lambda: state_path)
    monkeypatch.setattr(manager.paths, "log_dir", lambda: log_dir)
    return state_path, log_dir


@pytest.fixture
def patched_spawn(monkeypatch):
    monkeypatch.setattr(manager, "_free_port", lambda r: 8765)
    monkeypatch.setattr(manager, "_wait_health", lambda url, timeout: True)
    proc_seq = [FakeProc(12345), FakeProc(12346)]  # server, watchdog
    monkeypatch.setattr(manager.subprocess, "Popen", lambda *a, **kw: proc_seq.pop(0))
    return proc_seq


def test_ensure_warm_spawns_when_no_state(patched_paths, patched_spawn, monkeypatch):
    state_path, _ = patched_paths
    monkeypatch.setattr(manager, "_pid_alive", lambda pid: False)
    state = manager.ensure_warm(
        FakeBackend(), Path("/fake/model"),
        port_range=(8765, 8775), health_timeout=60,
        ttl_seconds=300, max_lifetime=1800, keep=False,
        isolation_mode=None,
    )
    assert state.server_pid == 12345
    assert state.watchdog_pid == 12346
    assert state.port == 8765
    on_disk = read_state(state_path)
    assert on_disk.server_pid == 12345
    assert on_disk.alive is True


def test_ensure_warm_idempotent_when_alive(patched_paths, monkeypatch):
    state_path, _ = patched_paths
    now = datetime.now(timezone.utc)
    pre = State(
        backend="mlx", model_repo="x", server_pid=999, port=8765,
        started_at=now, watchdog_pid=1000, last_used_at=now,
        expire_at=now + timedelta(seconds=300),
        ttl_seconds=300, max_lifetime_at=None, keep=False,
        alive=True, cleanup_failed=False,
    )
    write_state(state_path, pre)
    monkeypatch.setattr(manager, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(manager, "_wait_health", lambda url, timeout: True)
    popen_mock = MagicMock()
    monkeypatch.setattr(manager.subprocess, "Popen", popen_mock)

    state = manager.ensure_warm(
        FakeBackend(), Path("/fake/model"),
        port_range=(8765, 8775), health_timeout=60,
        ttl_seconds=600, max_lifetime=1800, keep=False,
        isolation_mode=None,
    )
    assert state.server_pid == 999
    popen_mock.assert_not_called()


def test_max_lifetime_ceiling(patched_paths, monkeypatch):
    state_path, _ = patched_paths
    now = datetime.now(timezone.utc)
    ceiling = now + timedelta(seconds=60)
    pre = State(
        backend="mlx", model_repo="x", server_pid=999, port=8765,
        started_at=now, watchdog_pid=1000, last_used_at=now,
        expire_at=now + timedelta(seconds=10),
        ttl_seconds=300, max_lifetime_at=ceiling, keep=False,
        alive=True, cleanup_failed=False,
    )
    write_state(state_path, pre)
    monkeypatch.setattr(manager, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(manager, "_wait_health", lambda url, timeout: True)

    state = manager.ensure_warm(
        FakeBackend(), Path("/fake/model"),
        port_range=(8765, 8775), health_timeout=60,
        ttl_seconds=99999, max_lifetime=1800, keep=False,
        isolation_mode=None,
    )
    assert state.expire_at <= ceiling


def test_stop_term_then_kill(patched_paths, monkeypatch):
    state_path, _ = patched_paths
    now = datetime.now(timezone.utc)
    pre = State(
        backend="mlx", model_repo="x", server_pid=7777, port=8765,
        started_at=now, watchdog_pid=7778, last_used_at=now,
        expire_at=now, ttl_seconds=300, max_lifetime_at=None, keep=False,
        alive=True, cleanup_failed=False,
    )
    write_state(state_path, pre)
    signals_sent: list[tuple[int, int]] = []
    monkeypatch.setattr(manager.os, "kill", lambda pid, sig: signals_sent.append((pid, sig)))
    alive_seq = iter([True] * 50 + [True, False])  # stays alive through TERM grace
    monkeypatch.setattr(manager, "_pid_alive", lambda pid: next(alive_seq, False))
    monkeypatch.setattr(manager.time, "sleep", lambda s: None)

    manager.stop()
    server_sigs = [sig for pid, sig in signals_sent if pid == 7777]
    assert signal.SIGTERM in server_sigs
    assert signal.SIGKILL in server_sigs
    assert read_state(state_path) is None


def test_nuke_calls_pkill_and_clears_state(patched_paths, monkeypatch):
    state_path, _ = patched_paths
    # write a fake state
    now = datetime.now(timezone.utc)
    pre = State(
        backend="mlx", model_repo="x", server_pid=99999, port=8765,
        started_at=now, watchdog_pid=99998, last_used_at=now,
        expire_at=now, ttl_seconds=300, max_lifetime_at=None, keep=False,
        alive=True, cleanup_failed=False,
    )
    write_state(state_path, pre)

    pkill_calls = []
    def fake_run(args, **kw):
        pkill_calls.append(args)
        return MagicMock(returncode=0)
    monkeypatch.setattr("subprocess.run", fake_run)

    manager.nuke()

    # Verify pkill invoked for each known backend
    pkill_patterns = [args for args in pkill_calls if args[0] == "pkill"]
    assert len(pkill_patterns) >= 1
    assert any("mlx_vlm.server" in args for args in pkill_patterns)
    # State cleared
    assert read_state(state_path) is None
