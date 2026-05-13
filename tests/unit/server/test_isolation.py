# tests/unit/server/test_isolation.py
import sys
from minicpm_v_local.server import isolation

def test_no_isolation_returns_cmd_as_is():
    cmd = ["python", "-m", "mlx_vlm.server"]
    wrapped = isolation.wrap(cmd, mode="none")
    assert wrapped == cmd

def test_mac_sandbox_exec_wraps(monkeypatch):
    monkeypatch.setattr(isolation, "_platform", lambda: "Darwin")
    wrapped = isolation.wrap(["python", "x"], mode="auto")
    assert wrapped[0] == "sandbox-exec"

def test_linux_bwrap_wraps(monkeypatch):
    monkeypatch.setattr(isolation, "_platform", lambda: "Linux")
    wrapped = isolation.wrap(["python", "x"], mode="auto")
    assert wrapped[0] == "bwrap"

def test_unsupported_platform_falls_back(monkeypatch):
    monkeypatch.setattr(isolation, "_platform", lambda: "Windows")
    wrapped = isolation.wrap(["python", "x"], mode="auto")
    assert wrapped[0] == "python"  # 退化为无沙箱
