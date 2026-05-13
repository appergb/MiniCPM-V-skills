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
    """Fallback prompt — plain stdin input. Used when questionary missing."""
    ans = input(f"{question} [{default}]: ").strip()
    return ans or default


# Questionary-based interactive prompter — arrow-key selection, green highlight.
try:
    import questionary as _q  # type: ignore

    _QUANT_QUESTION = "Quantization (4bit/5bit/8bit/bf16)"

    def _interactive_prompt(question: str, default: str) -> str:
        """Smart prompter — uses arrow-key select for known choice sets, plain text otherwise."""
        if _QUANT_QUESTION in question:
            choice = _q.select(
                question,
                choices=["4bit", "5bit", "8bit", "bf16"],
                default=default,
            ).unsafe_ask()
            return choice or default
        if "Enable sandbox isolation" in question:
            yes = _q.confirm(
                "Enable sandbox isolation? (sandbox-exec / bwrap)",
                default=default.lower().startswith("y"),
            ).unsafe_ask()
            return "y" if yes else "n"
        # default: free-text with default value
        ans = _q.text(question, default=default).unsafe_ask()
        return (ans or default).strip()

    _default_prompt = _interactive_prompt  # use questionary when available
except ImportError:
    pass


def run(prompter: Prompter = _default_prompt,
        force_backend: str | None = None,
        force_quant: str | None = None,
        non_interactive: bool = False) -> int:
    print("Running minicpm-v doctor...")

    if non_interactive:
        prompter = lambda q, d: d  # noqa: E731 — short-circuit all prompts to defaults

    # 1. detect
    tag = force_backend or detect.auto_detect()
    print(f"  [1/8] backend tag: {tag}{' (forced)' if force_backend else ''}")

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
    if force_quant:
        quant = force_quant
    elif tag == "mlx":
        quant = prompter("Quantization (4bit/5bit/8bit/bf16)", "4bit")
    elif tag == "cuda":
        quant = "bf16"
    else:
        quant = "Q4_K_M"
    print(f"  [4/8] quant: {quant}{' (forced)' if force_quant else ''}")

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
