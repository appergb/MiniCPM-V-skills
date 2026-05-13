# tests/unit/server/test_watchdog.py
# SYNTHESIZED from "测试要点".
from __future__ import annotations
import signal
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from minicpm_v_local.server import watchdog
from minicpm_v_local.server.state import State, write_state, read_state


def _make_state(*, expire_in_s: int, keep: bool = False, alive: bool = True,
                max_lifetime_in_s: int | None = None) -> State:
    now = datetime.now(timezone.utc)
    return State(
        backend="mlx", model_repo="x", server_pid=4242, port=8765,
        started_at=now, watchdog_pid=0, last_used_at=now,
        expire_at=now + timedelta(seconds=expire_in_s),
        ttl_seconds=300,
        max_lifetime_at=(now + timedelta(seconds=max_lifetime_in_s)) if max_lifetime_in_s is not None else None,
        keep=keep, alive=alive, cleanup_failed=False,
    )


@pytest.fixture
def patched_state(tmp_path, monkeypatch):
    state_path = tmp_path / "state.json"
    monkeypatch.setattr(watchdog.paths, "state_file", lambda: state_path)
    monkeypatch.setattr(watchdog.time, "sleep", lambda s: None)
    return state_path


def test_expired_triggers_kill_and_exit(patched_state, monkeypatch):
    write_state(patched_state, _make_state(expire_in_s=-10))
    sentinel = MagicMock(return_value=0)
    monkeypatch.setattr(watchdog, "_kill_and_exit", sentinel)
    rc = watchdog.main()
    assert rc == 0
    sentinel.assert_called_once_with(4242)


def test_future_expire_continues_loop(patched_state, monkeypatch):
    state_path = patched_state
    future = _make_state(expire_in_s=600)
    write_state(state_path, future)
    call_count = {"n": 0}
    real_read = watchdog.read_state

    def fake_read(p):
        call_count["n"] += 1
        if call_count["n"] >= 2:
            s = real_read(p)
            s.alive = False
            return s
        return real_read(p)

    monkeypatch.setattr(watchdog, "read_state", fake_read)
    monkeypatch.setattr(watchdog, "_kill_and_exit", MagicMock(side_effect=AssertionError))
    rc = watchdog.main()
    assert rc == 0
    assert call_count["n"] >= 2


def test_keep_true_never_kills(patched_state, monkeypatch):
    write_state(patched_state, _make_state(expire_in_s=-10, keep=True))
    call_count = {"n": 0}
    real_read = watchdog.read_state

    def fake_read(p):
        call_count["n"] += 1
        if call_count["n"] >= 2:
            s = real_read(p)
            s.alive = False
            return s
        return real_read(p)

    monkeypatch.setattr(watchdog, "read_state", fake_read)
    sentinel = MagicMock(side_effect=AssertionError("should not kill"))
    monkeypatch.setattr(watchdog, "_kill_and_exit", sentinel)
    rc = watchdog.main()
    assert rc == 0
    sentinel.assert_not_called()


def test_cleanup_failed_written_when_pid_remains_alive(patched_state, monkeypatch):
    write_state(patched_state, _make_state(expire_in_s=-10))
    monkeypatch.setattr(watchdog, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(watchdog.os, "kill", lambda pid, sig: None)
    rc = watchdog._kill_and_exit(4242)
    assert rc == 0
    final = read_state(patched_state)
    assert final.alive is False
    assert final.cleanup_failed is True
