"""Tests for doctor.run(). Spec §12.1."""
from __future__ import annotations
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from minicpm_v_local import doctor


def _fake_backend(tag: str, quant: str):
    b = MagicMock()
    b.install_check.return_value = (True, "")
    b.artifact_id.return_value = f"mlx-community/MiniCPM-V-4.6-{quant}"
    return b


@pytest.fixture
def happy_mocks(tmp_path, monkeypatch):
    cfg_file = tmp_path / "config.toml"
    monkeypatch.setattr(doctor.paths, "config_file", lambda: cfg_file)
    monkeypatch.setattr(doctor.paths, "ensure_runtime_dirs", lambda: None)
    monkeypatch.setattr(doctor.detect, "auto_detect", lambda: "mlx")
    monkeypatch.setattr(doctor, "get_backend", _fake_backend)
    monkeypatch.setattr(doctor.shutil, "which", lambda x: f"/usr/bin/{x}")
    monkeypatch.setattr(doctor, "ensure_model",
                        lambda repo, backend: tmp_path / "models" / repo)
    dump_calls = []
    monkeypatch.setattr(doctor, "dump_toml",
                        lambda cfg, p: dump_calls.append((cfg, p)))
    monkeypatch.setattr(doctor.isolation, "available_mode",
                        lambda: "sandbox-exec")
    return {"cfg_file": cfg_file, "dump_calls": dump_calls}


def test_happy_path_mac_default_quant(happy_mocks):
    rc = doctor.run(prompter=lambda q, d: d)
    assert rc == 0
    cfg, path = happy_mocks["dump_calls"][-1]
    assert path == happy_mocks["cfg_file"]
    assert cfg.backend == "mlx"
    assert cfg.quant == "4bit"
    assert cfg.isolation is False
    assert cfg.isolation_mode == "none"
    assert cfg.idle_timeout == 300


def test_happy_path_mac_user_picks_bf16(happy_mocks):
    answers = iter(["bf16", "y", "120"])
    rc = doctor.run(prompter=lambda q, d: next(answers))
    assert rc == 0
    cfg, _ = happy_mocks["dump_calls"][-1]
    assert cfg.quant == "bf16"
    assert cfg.isolation is True
    assert cfg.isolation_mode == "sandbox-exec"
    assert cfg.idle_timeout == 120


def test_cuda_uses_bf16_no_quant_prompt(happy_mocks, monkeypatch):
    monkeypatch.setattr(doctor.detect, "auto_detect", lambda: "cuda")
    rc = doctor.run(prompter=lambda q, d: d)
    assert rc == 0
    cfg, _ = happy_mocks["dump_calls"][-1]
    assert cfg.backend == "cuda"
    assert cfg.quant == "bf16"


def test_cpu_uses_q4_k_m_default(happy_mocks, monkeypatch):
    monkeypatch.setattr(doctor.detect, "auto_detect", lambda: "cpu")
    rc = doctor.run(prompter=lambda q, d: d)
    assert rc == 0
    cfg, _ = happy_mocks["dump_calls"][-1]
    assert cfg.backend == "cpu"
    assert cfg.quant == "Q4_K_M"


def test_force_backend_overrides_autodetect(happy_mocks, monkeypatch):
    # auto_detect would say mlx, but caller forces cuda
    monkeypatch.setattr(doctor.detect, "auto_detect", lambda: "mlx")
    rc = doctor.run(prompter=lambda q, d: d, force_backend="cuda")
    assert rc == 0
    cfg, _ = happy_mocks["dump_calls"][-1]
    assert cfg.backend == "cuda"
    assert cfg.quant == "bf16"  # cuda default


def test_force_quant_skips_prompt(happy_mocks):
    # mlx normally prompts for quant; force_quant skips that
    calls = []
    rc = doctor.run(prompter=lambda q, d: calls.append(q) or d,
                    force_quant="bf16")
    assert rc == 0
    cfg, _ = happy_mocks["dump_calls"][-1]
    assert cfg.quant == "bf16"
    # the quant question should NOT appear in calls (we used force_quant)
    assert not any("Quantization" in q for q in calls)


def test_non_interactive_uses_all_defaults(happy_mocks):
    # Even on mlx (which normally asks for quant), non_interactive should pick the default 4bit
    rc = doctor.run(non_interactive=True)
    assert rc == 0
    cfg, _ = happy_mocks["dump_calls"][-1]
    assert cfg.backend == "mlx"
    assert cfg.quant == "4bit"
    assert cfg.isolation is False
    assert cfg.idle_timeout == 300


def test_combined_force_backend_quant_non_interactive(happy_mocks, monkeypatch):
    # Full headless override: explicit backend + quant + non-interactive
    monkeypatch.setattr(doctor.detect, "auto_detect", lambda: "mlx")
    rc = doctor.run(force_backend="cuda", force_quant="bf16", non_interactive=True)
    assert rc == 0
    cfg, _ = happy_mocks["dump_calls"][-1]
    assert cfg.backend == "cuda"
    assert cfg.quant == "bf16"


def test_missing_ffmpeg_exits_2(happy_mocks, monkeypatch):
    monkeypatch.setattr(doctor.shutil, "which", lambda x: None)
    rc = doctor.run(prompter=lambda q, d: d)
    assert rc == 2
    assert happy_mocks["dump_calls"] == []


def test_missing_python_deps_exits_2(happy_mocks, monkeypatch):
    def bad_backend(tag, quant):
        b = _fake_backend(tag, quant)
        b.install_check.return_value = (False, "mlx-vlm not installed")
        return b
    monkeypatch.setattr(doctor, "get_backend", bad_backend)
    rc = doctor.run(prompter=lambda q, d: d)
    assert rc == 2
    assert happy_mocks["dump_calls"] == []


def test_mlx_missing_4_6_module_exits_2(happy_mocks, monkeypatch):
    """mlx backend without minicpmv4_6 module should exit 2 with install hint."""
    monkeypatch.setattr(doctor.detect, "auto_detect", lambda: "mlx")

    real_import = __import__
    def fake_import(name, *a, **kw):
        if name == "mlx_vlm.models" or (name == "mlx_vlm.models.minicpmv4_6"):
            raise ImportError("no module")
        if "mlx_vlm.models" in name and "minicpmv4_6" in name:
            raise ImportError("no module")
        return real_import(name, *a, **kw)

    # Patch the actual import statement target. The code does:
    #   from mlx_vlm.models import minicpmv4_6
    # which translates to importing mlx_vlm.models then getattr minicpmv4_6.
    import sys
    # Simulate: mlx_vlm.models has no `minicpmv4_6` attr.
    fake_models_module = type(sys)('mlx_vlm.models')
    monkeypatch.setitem(sys.modules, "mlx_vlm.models", fake_models_module)
    # Remove any cached minicpmv4_6 submodule
    sys.modules.pop("mlx_vlm.models.minicpmv4_6", None)

    rc = doctor.run(prompter=lambda q, d: d)
    assert rc == 2


def test_proxy_env_warning_does_not_block(happy_mocks, monkeypatch, capsys):
    """HTTP_PROXY set should only emit a warning, not block doctor."""
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:7890")
    # Skip mlx 4.6 precheck by using cpu backend
    monkeypatch.setattr(doctor.detect, "auto_detect", lambda: "cpu")
    rc = doctor.run(prompter=lambda q, d: d)
    captured = capsys.readouterr()
    assert "proxy env vars set" in captured.out
    assert rc == 0
