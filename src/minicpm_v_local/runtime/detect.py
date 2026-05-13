"""Platform detection. Spec §6.2."""
from __future__ import annotations
import platform
import subprocess
from typing import Literal

BackendTag = Literal["mlx", "cuda", "cpu"]


def _uname() -> tuple[str, str]:
    return platform.system(), platform.machine()


def _has_cuda() -> bool:
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        return r.returncode == 0 and bool(r.stdout.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def auto_detect() -> BackendTag:
    system, machine = _uname()
    if system == "Darwin" and machine in ("arm64", "aarch64"):
        return "mlx"
    if system == "Linux" and _has_cuda():
        return "cuda"
    return "cpu"


def resolve(requested: str) -> BackendTag:
    """`requested` from config: 'auto' or explicit tag."""
    if requested in ("mlx", "cuda", "cpu"):
        return requested  # type: ignore
    return auto_detect()
