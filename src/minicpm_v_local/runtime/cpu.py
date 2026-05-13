from shutil import which
from minicpm_v_local.runtime.backend import Backend


class CPUBackend(Backend):
    tag = "cpu"

    def __init__(self, quant: str):
        self.quant = quant or "Q4_K_M"

    def artifact_id(self) -> str:
        # GGUF repos for 4.6 may not yet exist as of 2026-05-13.
        # doctor 应该 verify 并给出 fallback。
        return "openbmb/MiniCPM-V-4.6-gguf"

    def launch_cmd(self, model_dir: str, port: int) -> list[str]:
        return [
            "llama-server",
            "-m", f"{model_dir}/ggml-model-Q4_K_M.gguf",
            "--port", str(port),
            "--host", "127.0.0.1",
            "--mmproj", f"{model_dir}/mmproj-model-f16.gguf",
        ]

    def install_check(self) -> tuple[bool, str]:
        if which("llama-server"):
            return True, ""
        return False, "install llama.cpp release: brew install llama.cpp (mac) or download from github"
