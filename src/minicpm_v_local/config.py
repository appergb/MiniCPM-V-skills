"""Layered config: CLI > env > toml > default. Spec §14."""
from __future__ import annotations
import os
import tomllib
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

_ENV_MAP = {
    "MINICPM_BACKEND": "backend",
    "MINICPM_QUANT": "quant",
    "MINICPM_IDLE_TIMEOUT": ("idle_timeout", int),
    "MINICPM_MAX_LIFETIME": ("max_lifetime", int),
}


@dataclass
class VideoConfig:
    scene_threshold: float = 0.3
    fallback_interval: float = 10.0
    max_frames: int = 60
    scene_merge_similarity: float = 0.85


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port_range: tuple[int, int] = (8765, 8775)
    health_timeout: int = 60


@dataclass
class Config:
    backend: str = "auto"
    quant: str = "4bit"
    idle_timeout: int = 300
    max_lifetime: int = 1800
    isolation: bool = False
    isolation_mode: str = "auto"
    video: VideoConfig = field(default_factory=VideoConfig)
    server: ServerConfig = field(default_factory=ServerConfig)

    @classmethod
    def defaults(cls) -> "Config":
        return cls()


def _read_toml(p: Path) -> dict[str, Any]:
    if not p.exists():
        return {}
    return tomllib.loads(p.read_text())


def _apply_env(cfg: Config) -> None:
    for env_key, target in _ENV_MAP.items():
        val = os.environ.get(env_key)
        if val is None:
            continue
        if isinstance(target, tuple):
            attr, caster = target
            setattr(cfg, attr, caster(val))
        else:
            setattr(cfg, target, val)


def _apply_dict(cfg: Config, data: dict[str, Any]) -> None:
    for k, v in data.items():
        if k == "video" and isinstance(v, dict):
            for vk, vv in v.items():
                setattr(cfg.video, vk, vv)
        elif k == "server" and isinstance(v, dict):
            for sk, sv in v.items():
                if sk == "port_range":
                    setattr(cfg.server, sk, tuple(sv))
                else:
                    setattr(cfg.server, sk, sv)
        elif hasattr(cfg, k):
            setattr(cfg, k, v)


def load(toml_path: Path, cli_overrides: dict[str, Any]) -> Config:
    cfg = Config.defaults()
    _apply_dict(cfg, _read_toml(toml_path))
    _apply_env(cfg)
    _apply_dict(cfg, {k: v for k, v in cli_overrides.items() if v is not None})
    return cfg


def dump_toml(cfg: Config, p: Path) -> None:
    """Write minimal TOML; used by doctor after first setup."""
    lines = [
        f'backend = "{cfg.backend}"',
        f'quant = "{cfg.quant}"',
        f"idle_timeout = {cfg.idle_timeout}",
        f"max_lifetime = {cfg.max_lifetime}",
        f"isolation = {str(cfg.isolation).lower()}",
        f'isolation_mode = "{cfg.isolation_mode}"',
        "",
        "[video]",
        f"scene_threshold = {cfg.video.scene_threshold}",
        f"fallback_interval = {cfg.video.fallback_interval}",
        f"max_frames = {cfg.video.max_frames}",
        f"scene_merge_similarity = {cfg.video.scene_merge_similarity}",
    ]
    p.write_text("\n".join(lines) + "\n")
