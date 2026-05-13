"""Backend abstraction. Spec §5, §6."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Literal

BackendTag = Literal["mlx", "cuda", "cpu"]


class Backend(ABC):
    tag: BackendTag
    quant: str

    @abstractmethod
    def artifact_id(self) -> str: ...

    @abstractmethod
    def launch_cmd(self, model_dir: str, port: int) -> list[str]: ...

    @abstractmethod
    def install_check(self) -> tuple[bool, str]:
        """Return (ok, message). False if backend deps missing."""

    def health_path(self) -> str:
        return "/health"
