# tests/unit/runtime/test_detect.py
import pytest
from minicpm_v_local.runtime import detect

@pytest.fixture
def fake_uname(monkeypatch):
    def _f(system, machine):
        monkeypatch.setattr(detect, "_uname", lambda: (system, machine))
    return _f

def test_mac_arm64(fake_uname):
    fake_uname("Darwin", "arm64")
    assert detect.auto_detect() == "mlx"

def test_mac_x86_returns_cpu(fake_uname):
    fake_uname("Darwin", "x86_64")
    assert detect.auto_detect() == "cpu"

def test_linux_with_nvidia(fake_uname, monkeypatch):
    fake_uname("Linux", "x86_64")
    monkeypatch.setattr(detect, "_has_cuda", lambda: True)
    assert detect.auto_detect() == "cuda"

def test_linux_without_nvidia(fake_uname, monkeypatch):
    fake_uname("Linux", "x86_64")
    monkeypatch.setattr(detect, "_has_cuda", lambda: False)
    assert detect.auto_detect() == "cpu"

def test_windows(fake_uname, monkeypatch):
    fake_uname("Windows", "AMD64")
    monkeypatch.setattr(detect, "_has_cuda", lambda: False)
    assert detect.auto_detect() == "cpu"
