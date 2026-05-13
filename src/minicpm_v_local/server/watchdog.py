"""Sidecar watchdog process. Spec §12.3."""
from __future__ import annotations
import os
import signal
import sys
import time
from datetime import datetime, timezone

from minicpm_v_local import paths
from minicpm_v_local.server.state import read_state, write_state, clear_state

CHECK_INTERVAL_S = 10
TERM_GRACE_S = 5


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def main() -> int:
    while True:
        s = read_state(paths.state_file())
        if not s or not s.alive:
            return 0
        now = datetime.now(timezone.utc)
        expired = (not s.keep) and now >= s.expire_at
        over_lifetime = s.max_lifetime_at is not None and now >= s.max_lifetime_at
        if expired or over_lifetime:
            return _kill_and_exit(s.server_pid)
        time.sleep(CHECK_INTERVAL_S)


def _kill_and_exit(server_pid: int) -> int:
    if _pid_alive(server_pid):
        try:
            os.kill(server_pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        deadline = time.time() + TERM_GRACE_S
        while time.time() < deadline and _pid_alive(server_pid):
            time.sleep(0.2)
        if _pid_alive(server_pid):
            try:
                os.kill(server_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    # cleanup_failed 检测（简化 v1：仅检测进程是否消失）
    failed = _pid_alive(server_pid)
    s = read_state(paths.state_file())
    if s:
        s.alive = False
        s.cleanup_failed = failed
        write_state(paths.state_file(), s)
    else:
        clear_state(paths.state_file())
    return 0


if __name__ == "__main__":
    sys.exit(main())
