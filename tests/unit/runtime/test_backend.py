# tests/unit/runtime/test_backend.py
import pytest
from minicpm_v_local.runtime.backend import Backend
from minicpm_v_local.runtime.factory import get_backend

def test_factory_returns_correct_backend():
    b = get_backend("mlx", quant="4bit")
    assert b.tag == "mlx"

def test_factory_unknown_raises():
    with pytest.raises(ValueError):
        get_backend("tpu", quant="4bit")

def test_backend_has_required_methods():
    b = get_backend("mlx", quant="4bit")
    assert callable(b.launch_cmd)
    assert isinstance(b.artifact_id(), str)
    assert "mlx-community/MiniCPM-V-4.6" in b.artifact_id()
