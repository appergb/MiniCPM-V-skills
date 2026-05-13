from minicpm_v_local.runtime.backend import Backend


class CUDABackend(Backend):
    tag = "cuda"

    def __init__(self, quant: str):
        self.quant = quant  # vLLM 不分量化档位，记录用

    def artifact_id(self) -> str:
        return "openbmb/MiniCPM-V-4.6"

    def launch_cmd(self, model_dir: str, port: int) -> list[str]:
        return [
            "vllm", "serve", model_dir,
            "--port", str(port),
            "--host", "127.0.0.1",
            "--trust-remote-code",
        ]

    def install_check(self) -> tuple[bool, str]:
        try:
            import vllm  # noqa
            return True, ""
        except ImportError:
            return False, "pip install vllm"
