# tests/unit/server/test_state.py
from datetime import datetime, timezone
from minicpm_v_local.server.state import State, read_state, write_state, clear_state

def test_write_read_roundtrip(tmp_path):
    path = tmp_path / "state.json"
    s = State(
        backend="mlx", model_repo="mlx-community/MiniCPM-V-4.6-4bit",
        server_pid=1234, port=8765, started_at=datetime.now(timezone.utc),
        watchdog_pid=5678,
        last_used_at=datetime.now(timezone.utc),
        expire_at=datetime.now(timezone.utc),
        ttl_seconds=300, max_lifetime_at=None, keep=False,
        alive=True, cleanup_failed=False,
    )
    write_state(path, s)
    s2 = read_state(path)
    assert s2.server_pid == 1234

def test_read_missing_returns_none(tmp_path):
    assert read_state(tmp_path / "nope.json") is None

def test_clear_state(tmp_path):
    p = tmp_path / "state.json"
    p.write_text('{"alive": true}')
    clear_state(p)
    s = read_state(p)
    assert s is None or s.alive is False

def test_atomic_write_handles_concurrent_read(tmp_path):
    # half-written file simulated
    p = tmp_path / "state.json"
    p.write_text('{ "broken')
    assert read_state(p) is None  # tolerates corrupt JSON
