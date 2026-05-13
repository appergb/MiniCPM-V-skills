"""Sandbox wrappers. Spec §9."""
from __future__ import annotations
import platform
import tempfile
from pathlib import Path
from typing import Optional

# Minimal mac profile: allow default, restrict home writes outside cache.
_MAC_PROFILE = """(version 1)
(allow default)
(deny file-write*
  (subpath (string-append (param "HOME") "/Documents"))
  (subpath (string-append (param "HOME") "/Desktop")))
"""


def _platform() -> str:
    return platform.system()


def _mac_wrap(cmd: list[str]) -> list[str]:
    prof = Path(tempfile.gettempdir()) / "minicpm-v-mac.sb"
    if not prof.exists():
        prof.write_text(_MAC_PROFILE)
    return ["sandbox-exec", "-f", str(prof), *cmd]


def _linux_wrap(cmd: list[str]) -> list[str]:
    base = [
        "bwrap",
        "--unshare-all",
        "--share-net",
        "--bind", "/", "/",
        "--proc", "/proc",
        "--dev", "/dev",
    ]
    # GPU 设备透传（容错：不存在则忽略）
    for dev in ("/dev/nvidia0", "/dev/nvidiactl", "/dev/nvidia-uvm"):
        if Path(dev).exists():
            base += ["--dev-bind", dev, dev]
    return [*base, *cmd]


def wrap(cmd: list[str], mode: str) -> list[str]:
    """Wrap a command in a sandbox.

    mode: 'none' | 'auto' | 'sandbox-exec' | 'bwrap'
    """
    if mode == "none":
        return cmd
    sys = _platform()
    if mode in ("auto", "sandbox-exec") and sys == "Darwin":
        return _mac_wrap(cmd)
    if mode in ("auto", "bwrap") and sys == "Linux":
        return _linux_wrap(cmd)
    return cmd  # 无支持平台：退化


def available_mode() -> Optional[str]:
    sys = _platform()
    if sys == "Darwin":
        return "sandbox-exec"
    if sys == "Linux":
        from shutil import which
        return "bwrap" if which("bwrap") else None
    return None
