from minicpm_v_local.runtime.backend import Backend
from minicpm_v_local.runtime.mlx import MLXBackend
from minicpm_v_local.runtime.cuda import CUDABackend
from minicpm_v_local.runtime.cpu import CPUBackend


def get_backend(tag: str, quant: str) -> Backend:
    mapping = {"mlx": MLXBackend, "cuda": CUDABackend, "cpu": CPUBackend}
    if tag not in mapping:
        raise ValueError(f"unknown backend: {tag}")
    return mapping[tag](quant=quant)
