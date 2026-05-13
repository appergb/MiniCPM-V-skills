"""Model download with lockfile. Spec §13."""
from __future__ import annotations
import fcntl
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from huggingface_hub import snapshot_download

from minicpm_v_local import paths


@contextmanager
def _download_lock() -> Iterator[None]:
    lock_path = paths.download_lock()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def ensure_model(repo_id: str, backend: str, *, allow_patterns: list[str] | None = None) -> Path:
    """Download (or verify cached) model. Returns local model dir."""
    target = paths.cache_dir(backend) / repo_id.replace("/", "__")
    target.mkdir(parents=True, exist_ok=True)
    with _download_lock():
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(target),
            allow_patterns=allow_patterns,
            local_dir_use_symlinks=False,
        )
    return target
