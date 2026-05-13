from minicpm_v_local.runtime.cpu import CPUBackend

def test_cpu_launch_cmd():
    b = CPUBackend(quant="Q4_K_M")
    cmd = b.launch_cmd("/tmp/model", 8765)
    assert "llama-server" in cmd
    assert "--port" in cmd and "8765" in cmd
    assert "-m" in cmd
    assert "/tmp/model/ggml-model-Q4_K_M.gguf" in cmd
    assert "--mmproj" in cmd
    assert "/tmp/model/mmproj-model-f16.gguf" in cmd

def test_cpu_artifact_id():
    assert "gguf" in CPUBackend(quant="Q4_K_M").artifact_id().lower()
    assert "MiniCPM-V-4.6" in CPUBackend(quant="Q4_K_M").artifact_id()

def test_cpu_quant_defaults_when_empty():
    b = CPUBackend(quant="")
    assert b.quant == "Q4_K_M"
