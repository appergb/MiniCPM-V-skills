"""Centralized filesystem paths. Spec §14.2."""
from __future__ import annotations
import os
from pathlib import Path

_APP = "minicpm-v-local"


def _xdg(env: str, default: str) -> Path:
    base = os.environ.get(env)
    return Path(base) if base else Path.home() / default


def config_dir() -> Path:
    return _xdg("XDG_CONFIG_HOME", ".config") / _APP


def config_file() -> Path:
    return config_dir() / "config.toml"


def run_dir() -> Path:
    return Path.home() / ".run" / _APP


def state_file() -> Path:
    return run_dir() / "state.json"


def cache_dir(backend: str) -> Path:
    return _xdg("XDG_CACHE_HOME", ".cache") / _APP / backend


def log_dir() -> Path:
    return _xdg("XDG_STATE_HOME", ".local/state") / _APP / "logs"


def cli_lock() -> Path:
    return run_dir() / "cli.lock"


def download_lock() -> Path:
    return run_dir() / "download.lock"


def frames_tmp_dir(run_id: str) -> Path:
    tmp = Path(os.environ.get("TMPDIR", "/tmp"))
    return tmp / _APP / f"frames-{run_id}"


def ensure_runtime_dirs() -> None:
    for d in (config_dir(), run_dir(), log_dir()):
        d.mkdir(parents=True, exist_ok=True)
