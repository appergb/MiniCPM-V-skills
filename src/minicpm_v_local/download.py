"""Model download with lockfile, retry, and hf_transfer support. Spec §13."""
from __future__ import annotations
import fcntl
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

# Enable hf_transfer (multi-connection Rust accelerator) when installed.
# Must run BEFORE importing huggingface_hub's downloader internals.
try:
    import hf_transfer  # noqa: F401
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
except ImportError:
    pass

from huggingface_hub import snapshot_download
from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError

from minicpm_v_local import paths

# Retry policy for transient network failures (proxy 5xx, ProxyError, etc.)
_RETRY_BACKOFFS_S = (5, 15, 45)
_RETRYABLE = (HfHubHTTPError, LocalEntryNotFoundError, OSError, ConnectionError)


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


def _cleanup_incomplete(target: Path) -> int:
    """Remove HF's partial-download marker files. Returns count removed."""
    if not target.exists():
        return 0
    removed = 0
    for p in target.rglob("*.incomplete"):
        try:
            p.unlink()
            removed += 1
        except OSError:
            pass
    return removed


def ensure_model(
    repo_id: str,
    backend: str,
    *,
    allow_patterns: list[str] | None = None,
    max_attempts: int = 3,
) -> Path:
    """Download (or verify cached) model. Returns local model dir.

    Resilient to transient network errors via exponential-backoff retry.
    Cleans leftover .incomplete files before each attempt.
    """
    target = paths.cache_dir(backend) / repo_id.replace("/", "__")
    target.mkdir(parents=True, exist_ok=True)

    with _download_lock():
        last_err: Exception | None = None
        for attempt in range(max_attempts):
            removed = _cleanup_incomplete(target)
            if removed:
                print(f"        cleaned {removed} incomplete file(s) from previous run")
            try:
                snapshot_download(
                    repo_id=repo_id,
                    local_dir=str(target),
                    allow_patterns=allow_patterns,
                )
                return target
            except _RETRYABLE as e:
                last_err = e
                if attempt + 1 < max_attempts:
                    backoff = _RETRY_BACKOFFS_S[min(attempt, len(_RETRY_BACKOFFS_S) - 1)]
                    print(f"        download attempt {attempt + 1}/{max_attempts} failed "
                          f"({type(e).__name__}); retrying in {backoff}s...")
                    time.sleep(backoff)
                    continue
                raise
        # unreachable but defensive
        if last_err:
            raise last_err
    return target
