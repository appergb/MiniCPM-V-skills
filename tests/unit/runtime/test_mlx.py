from minicpm_v_local.runtime.mlx import MLXBackend

def test_mlx_launch_cmd():
    b = MLXBackend(quant="4bit")
    cmd = b.launch_cmd("/tmp/model", 8765)
    assert "mlx_vlm.server" in " ".join(cmd)
    assert "--port" in cmd and "8765" in cmd
    assert "--model" in cmd and "/tmp/model" in cmd

def test_mlx_artifact_per_quant():
    assert "4bit" in MLXBackend(quant="4bit").artifact_id()
    assert "bf16" in MLXBackend(quant="bf16").artifact_id()
