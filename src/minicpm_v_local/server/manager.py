"""Server lifecycle. Spec §8, §12.2."""
from __future__ import annotations
import os
import signal
import socket
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import httpx

from minicpm_v_local import paths
from minicpm_v_local.runtime.backend import Backend
from minicpm_v_local.server import isolation
from minicpm_v_local.server.state import State, read_state, write_state, clear_state


def _free_port(port_range: tuple[int, int]) -> int:
    for p in range(port_range[0], port_range[1] + 1):
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", p))
                return p
            except OSError:
                continue
    raise RuntimeError(f"no free port in {port_range}")


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def _wait_health(url: str, timeout: int) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = httpx.get(url, timeout=2.0)
            if r.status_code == 200:
                return True
        except httpx.RequestError:
            pass
        time.sleep(0.5)
    return False


def ensure_warm(
    backend: Backend, model_dir: Path, *,
    port_range: tuple[int, int], health_timeout: int,
    ttl_seconds: int, max_lifetime: int, keep: bool,
    isolation_mode: Optional[str],
) -> State:
    """Idempotent: return State of running server, spawning if needed."""
    state_path = paths.state_file()
    existing = read_state(state_path)
    if existing and existing.alive and _pid_alive(existing.server_pid):
        url = f"http://127.0.0.1:{existing.port}{backend.health_path()}"
        if _wait_health(url, timeout=2):
            return _bump(state_path, existing, ttl_seconds, keep)

    return _spawn(
        backend, model_dir,
        port_range=port_range, health_timeout=health_timeout,
        ttl_seconds=ttl_seconds, max_lifetime=max_lifetime,
        keep=keep, isolation_mode=isolation_mode,
    )


def _spawn(
    backend: Backend, model_dir: Path, *,
    port_range, health_timeout, ttl_seconds, max_lifetime, keep, isolation_mode,
) -> State:
    port = _free_port(port_range)
    cmd = backend.launch_cmd(str(model_dir), port)
    if isolation_mode and isolation_mode != "none":
        cmd = isolation.wrap(cmd, mode=isolation_mode)

    log_path = paths.log_dir() / f"server-{port}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_f = open(log_path, "a")
    proc = subprocess.Popen(
        cmd, stdout=log_f, stderr=subprocess.STDOUT,
        start_new_session=True,
    )

    url = f"http://127.0.0.1:{port}{backend.health_path()}"
    if not _wait_health(url, timeout=health_timeout):
        proc.terminate()
        raise RuntimeError(f"server health check failed within {health_timeout}s; see {log_path}")

    now = datetime.now(timezone.utc)
    state = State(
        backend=backend.tag,
        model_repo=backend.artifact_id(),
        server_pid=proc.pid, port=port, started_at=now,
        watchdog_pid=0,  # 由 watchdog spawn 后填
        last_used_at=now,
        expire_at=now + timedelta(seconds=ttl_seconds),
        ttl_seconds=ttl_seconds,
        max_lifetime_at=now + timedelta(seconds=max_lifetime) if max_lifetime > 0 else None,
        keep=keep, alive=True, cleanup_failed=False,
        isolation_mode=isolation_mode,
    )
    write_state(paths.state_file(), state)

    # spawn watchdog
    wd = subprocess.Popen(
        ["python", "-m", "minicpm_v_local.server.watchdog"],
        start_new_session=True,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    state.watchdog_pid = wd.pid
    write_state(paths.state_file(), state)
    return state


def _bump(state_path: Path, s: State, ttl_seconds: int, keep: bool) -> State:
    now = datetime.now(timezone.utc)
    s.last_used_at = now
    s.keep = keep
    if not keep:
        new_expire = now + timedelta(seconds=ttl_seconds)
        if s.max_lifetime_at and new_expire > s.max_lifetime_at:
            new_expire = s.max_lifetime_at
        s.expire_at = new_expire
        s.ttl_seconds = ttl_seconds
    write_state(state_path, s)
    return s


def stop(force: bool = False) -> None:
    """Manual stop. Spec §8.3."""
    s = read_state(paths.state_file())
    if not s or not s.alive:
        return
    try:
        os.kill(s.server_pid, signal.SIGTERM)
        for _ in range(50):
            if not _pid_alive(s.server_pid):
                break
            time.sleep(0.1)
        if _pid_alive(s.server_pid):
            os.kill(s.server_pid, signal.SIGKILL)
        if s.watchdog_pid and _pid_alive(s.watchdog_pid):
            os.kill(s.watchdog_pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    clear_state(paths.state_file())


def nuke() -> None:
    """Nuclear stop: pkill any known backend server processes + force-clear state.

    Useful when state.json is corrupted or normal stop() can't find/kill the
    server (e.g. process detached, watchdog crashed, port stuck).
    """
    import subprocess as _sp
    patterns = ["mlx_vlm.server", "vllm.entrypoints.openai", "llama-server"]
    for pattern in patterns:
        try:
            _sp.run(["pkill", "-9", "-f", pattern],
                    capture_output=True, check=False)
        except FileNotFoundError:
            pass  # no pkill on this system; best-effort
    clear_state(paths.state_file())
