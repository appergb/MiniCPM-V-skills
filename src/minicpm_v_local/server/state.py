"""state.json schema + atomic IO. Spec §8.5."""
from __future__ import annotations
import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

SCHEMA_VERSION = 1


@dataclass
class State:
    backend: str
    model_repo: str
    server_pid: int
    port: int
    started_at: datetime
    watchdog_pid: int
    last_used_at: datetime
    expire_at: datetime
    ttl_seconds: int
    max_lifetime_at: Optional[datetime]
    keep: bool
    alive: bool
    cleanup_failed: bool
    isolation_mode: Optional[str] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, datetime):
                d[k] = v.isoformat()
        d["schema_version"] = SCHEMA_VERSION
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "State":
        d = dict(d)
        d.pop("schema_version", None)
        for k in ("started_at", "last_used_at", "expire_at", "max_lifetime_at"):
            if d.get(k):
                d[k] = datetime.fromisoformat(d[k])
            else:
                d[k] = None
        return cls(**d)


def read_state(path: Path) -> Optional[State]:
    if not path.exists():
        return None
    try:
        return State.from_dict(json.loads(path.read_text()))
    except (json.JSONDecodeError, TypeError, KeyError):
        return None


def write_state(path: Path, s: State) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(s.to_dict(), indent=2))
    os.replace(tmp, path)


def clear_state(path: Path) -> None:
    if path.exists():
        path.unlink()
