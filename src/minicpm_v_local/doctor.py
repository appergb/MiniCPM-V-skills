"""Doctor: 8-step first-run setup. Spec §12.1."""
from __future__ import annotations
import shutil
import sys
from dataclasses import replace
from pathlib import Path
from typing import Callable

from minicpm_v_local import paths
from minicpm_v_local.config import Config, dump_toml
from minicpm_v_local.runtime import detect
from minicpm_v_local.runtime.factory import get_backend
from minicpm_v_local.server import isolation
from minicpm_v_local.download import ensure_model


Prompter = Callable[[str, str], str]


def _default_prompt(question: str, default: str) -> str:
    ans = input(f"{question} [{default}]: ").strip()
    return ans or default


def run(prompter: Prompter = _default_prompt) -> int:
    print("Running minicpm-v doctor...")

    # 1. detect
    tag = detect.auto_detect()
    print(f"  [1/8] backend tag: {tag}")

    # 2. python deps
    backend = get_backend(tag, quant="4bit")
    ok, msg = backend.install_check()
    if not ok:
        print(f"  [2/8] missing deps for {tag}: {msg}")
        return 2
    print(f"  [2/8] python deps OK")

    # 3. ffmpeg
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        print("  [3/8] missing ffmpeg/ffprobe. Install via: brew install ffmpeg (mac) / apt install ffmpeg (linux)")
        return 2
    print("  [3/8] ffmpeg OK")

    # 4. quant
    if tag == "mlx":
        quant = prompter("Quantization (4bit/5bit/8bit/bf16)", "4bit")
    elif tag == "cuda":
        quant = "bf16"
    else:
        quant = "Q4_K_M"
    print(f"  [4/8] quant: {quant}")

    # 5. download
    backend = get_backend(tag, quant=quant)
    repo = backend.artifact_id()
    print(f"  [5/8] downloading {repo} ...")
    model_dir = ensure_model(repo, backend=tag)
    print(f"        → {model_dir}")

    # 6. isolation
    iso_ans = prompter("Enable sandbox isolation? (y/N)", "n")
    isolation_on = iso_ans.lower().startswith("y")
    iso_mode = isolation.available_mode() or "none" if isolation_on else "none"
    print(f"  [6/8] isolation: {isolation_on} ({iso_mode})")

    # 7. idle timeout
    idle_str = prompter("Default idle_timeout in seconds", "300")
    idle = int(idle_str)
    print(f"  [7/8] idle_timeout: {idle}")

    # 8. test launch — 留到首次推理时做（避免 doctor 太慢）
    cfg = Config.defaults()
    cfg.backend = tag
    cfg.quant = quant
    cfg.isolation = isolation_on
    cfg.isolation_mode = iso_mode
    cfg.idle_timeout = idle
    paths.ensure_runtime_dirs()
    paths.config_file().parent.mkdir(parents=True, exist_ok=True)
    dump_toml(cfg, paths.config_file())
    print(f"  [8/8] config written to {paths.config_file()}")
    print("Doctor done. Try: minicpm-v image <path>")
    return 0
