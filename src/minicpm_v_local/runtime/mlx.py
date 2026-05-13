from minicpm_v_local.runtime.backend import Backend


class MLXBackend(Backend):
    tag = "mlx"
    _QUANT_MAP = {
        "4bit": "mlx-community/MiniCPM-V-4.6-4bit",
        "5bit": "mlx-community/MiniCPM-V-4.6-5bit",
        "8bit": "mlx-community/MiniCPM-V-4.6-8bit",
        "bf16": "mlx-community/MiniCPM-V-4.6-bf16",
    }

    def __init__(self, quant: str):
        if quant not in self._QUANT_MAP:
            raise ValueError(f"unsupported quant for mlx: {quant}")
        self.quant = quant

    def artifact_id(self) -> str:
        return self._QUANT_MAP[self.quant]

    def launch_cmd(self, model_dir: str, port: int) -> list[str]:
        return [
            "python", "-m", "mlx_vlm.server",
            "--model", model_dir,
            "--port", str(port),
            "--host", "127.0.0.1",
        ]

    def install_check(self) -> tuple[bool, str]:
        try:
            import mlx_vlm  # noqa
            return True, ""
        except ImportError:
            return False, "pip install -U mlx-vlm"
