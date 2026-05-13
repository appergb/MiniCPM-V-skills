from minicpm_v_local.runtime.cuda import CUDABackend

def test_cuda_launch_cmd():
    b = CUDABackend(quant="bf16")
    cmd = b.launch_cmd("/tmp/model", 8765)
    assert "vllm" in cmd and "serve" in cmd
    assert "--port" in cmd and "8765" in cmd
    assert "/tmp/model" in cmd
    assert "--trust-remote-code" in cmd

def test_cuda_artifact_is_openbmb():
    assert CUDABackend(quant="bf16").artifact_id() == "openbmb/MiniCPM-V-4.6"
    assert CUDABackend(quant="4bit").artifact_id() == "openbmb/MiniCPM-V-4.6"
