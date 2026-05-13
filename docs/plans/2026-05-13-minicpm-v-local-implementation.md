# MiniCPM-V Local Skill — Implementation Plan

> **For agentic workers:** This plan is consumed by **dev subagents**, one per Task. Each Task is self-contained: data lookup → dev → review → destroy dev agent → next Task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** 在 macOS Apple Silicon / Linux + CUDA / Linux-Windows CPU 三个平台上，把 MiniCPM-V 4.6 (1.3B) 通过开源后端 server 跑起来，封装为 Claude Code Skill + Python 库 + CLI，支持自动下载、Claude 主控 lifecycle、视频时间轴 pipeline。

**Architecture:** 三层（Skill 壳 → Python CLI → 后端 server）。所有平台统一走 OpenAI 兼容 HTTP。Backend 用工厂 + 策略模式；server 用 sidecar watchdog 管 lifecycle；视频走 ffmpeg 抽帧 + 串行 caption + 余弦聚合。

**Tech Stack:** Python 3.11+ / pytest / mlx-vlm / vLLM / llama.cpp / huggingface_hub / ffmpeg (system) / FastAPI client (openai SDK) / tomllib.

**Spec:** `docs/specs/2026-05-13-minicpm-v-local-design.md`（必读）

**Agent Workflow（每个 Task 严格遵守）:**

```
主 agent (我)
   │ 1. 分派 ResearchAgent (Explore) ─→ 读 spec 对应小节 + 上游 docs/源码 → 返回 brief
   │
   │ 2. 收到 brief → 转交 DevAgent (general-purpose) → 按 plan 写代码 + 测试 → 自测
   │
   │ 3. 并行启动 ReviewAgent (general-purpose) → 跑测试 + 审 code + 单一职责检查
   │       │
   │       └ 有问题 → 回 DevAgent 改 → 直到通过
   │
   │ 4. 通过则销毁 DevAgent，进入下一 Task
   │
   └ 每个 Task 完成后 git commit（commit 是验收的一部分）
```

主 agent **不直接读上游源码 / 网页文档 / spec 全文**，只读：用户消息、子 agent 返回的 brief、自己之前的产出、`docs/plans/` 自身。

---

## File Structure（开发前锁定）

```
MiniCPM-V-skills/
├── docs/
│   ├── specs/2026-05-13-minicpm-v-local-design.md   (已存在)
│   └── plans/2026-05-13-minicpm-v-local-implementation.md  (本文档)
├── README.md
├── LICENSE                                          (Apache 2.0)
├── pyproject.toml
├── .gitignore
├── .python-version                                  (3.11)
├── src/
│   └── minicpm_v_local/
│       ├── __init__.py
│       ├── __main__.py             (python -m minicpm_v_local 入口)
│       ├── paths.py                (~/.config /.cache /.run 等常量, ≤50)
│       ├── config.py               (优先级链, ≤100)
│       ├── runtime/
│       │   ├── __init__.py
│       │   ├── detect.py           (OS/CPU/GPU 探测, ≤100)
│       │   ├── backend.py          (Backend 抽象基类, ≤150)
│       │   ├── mlx.py              (MLXBackend, ≤80)
│       │   ├── cuda.py             (CUDABackend, ≤80)
│       │   ├── cpu.py              (CPUBackend, ≤80)
│       │   └── factory.py          (按 tag 取 backend 实例, ≤40)
│       ├── server/
│       │   ├── __init__.py
│       │   ├── manager.py          (spawn / health / kill, ≤250)
│       │   ├── watchdog.py         (sidecar 进程, ≤100)
│       │   ├── isolation.py        (sandbox-exec / bwrap / noop, ≤120)
│       │   └── state.py            (state.json schema + 原子读写, ≤120)
│       ├── download.py             (HF snapshot_download + lockfile, ≤200)
│       ├── client.py               (OpenAI HTTP 客户端, ≤150)
│       ├── pipeline/
│       │   ├── __init__.py
│       │   ├── image.py            (单图, ≤100)
│       │   └── video.py            (ffprobe/ffmpeg/聚合, ≤400)
│       ├── doctor.py               (8 步自检, ≤300)
│       └── cli.py                  (argparse 入口, ≤200)
├── tests/
│   ├── unit/
│   │   ├── test_paths.py
│   │   ├── test_config.py
│   │   ├── runtime/
│   │   │   ├── test_detect.py
│   │   │   ├── test_backend.py
│   │   │   ├── test_mlx.py
│   │   │   ├── test_cuda.py
│   │   │   └── test_cpu.py
│   │   ├── server/
│   │   │   ├── test_state.py
│   │   │   ├── test_manager.py
│   │   │   ├── test_watchdog.py
│   │   │   └── test_isolation.py
│   │   ├── test_download.py
│   │   ├── test_client.py
│   │   ├── pipeline/
│   │   │   ├── test_image.py
│   │   │   └── test_video.py
│   │   ├── test_doctor.py
│   │   └── test_cli.py
│   ├── integration/
│   │   ├── test_end_to_end_cpu.py
│   │   └── test_lifecycle.py
│   └── fixtures/
│       ├── sample.jpg
│       ├── sample-5s.mp4
│       └── tiny-vlm.gguf            (一个非常小的 dummy GGUF 用于 CI)
├── skill/                            (Claude Code Skill 壳，安装时拷到 ~/.claude/skills/)
│   ├── SKILL.md
│   ├── scripts/
│   │   └── run.sh
│   └── install.sh                   (拷到 ~/.claude/skills/minicpm-v/)
└── scripts/
    └── cleanup.sh                   (项目清理：venv / __pycache__ / build artifacts)
```

每个 src 文件 ≤ 行数预算；超出即拆分。

---

## Phase 0：项目脚手架（先决条件）

### Task 0.1: git init + 基础工程文件

**Files:**
- Create: `.gitignore`, `pyproject.toml`, `.python-version`, `README.md`, `LICENSE`

**ResearchAgent brief 需要的内容**：
- spec doc 路径 + 14 项决议（附录 A）
- Python 3.11 + pyproject.toml 标准结构（`pdm` 或 `uv` 或 plain pip + setuptools）

**Steps:**

- [ ] **Step 1: `git init` + 基础忽略**

```bash
cd "/Users/lvbaiqing/TRUE 开发/MiniCPM-V-skills"
git init -b main
```

`.gitignore`:

```gitignore
__pycache__/
*.py[cod]
.venv/
venv/
dist/
build/
*.egg-info/
.pytest_cache/
.mypy_cache/
.coverage
htmlcov/
.DS_Store
~/.cache/minicpm-v-local/
~/.run/minicpm-v-local/
```

- [ ] **Step 2: `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "minicpm-v-local"
version = "0.1.0"
description = "Local visual preprocessor using MiniCPM-V 4.6 across MLX/CUDA/CPU backends."
authors = [{name = "Owner"}]
license = {text = "Apache-2.0"}
requires-python = ">=3.11"
dependencies = [
  "huggingface_hub>=0.24",
  "httpx>=0.27",
  "tomli;python_version<'3.11'",
  "pillow>=10",
  "numpy>=1.26",
]

[project.optional-dependencies]
mlx = ["mlx-vlm>=0.1"]
cuda = ["vllm>=0.6"]
cpu = []                       # llama-server 走系统二进制
dev = ["pytest>=8", "pytest-cov>=5", "pytest-mock>=3.12", "ruff>=0.5"]

[project.scripts]
minicpm-v = "minicpm_v_local.cli:main"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra --strict-markers"
```

- [ ] **Step 3: `.python-version`** = `3.11`

- [ ] **Step 4: README skeleton（必含"5 分钟自动卸载"提示——spec 8.6 强制要求）**

```markdown
# MiniCPM-V Local

Local visual preprocessing using MiniCPM-V 4.6 (1.3B). Captions images and
video timelines locally without sending pixels to the main model.

⚠️ **自动卸载**：本地模型 server 在 5 分钟无请求后自动从内存/显存卸载。
下次调用会自动重新加载（cold start ≈ 3–15s）。

See `docs/specs/2026-05-13-minicpm-v-local-design.md` for full design.
```

- [ ] **Step 5: LICENSE** = Apache-2.0 全文（标准模板）

- [ ] **Step 6: 验证目录树**

```bash
git status
ls -la
```

- [ ] **Step 7: 初次 commit**

```bash
git add .
git commit -m "chore: project scaffold"
```

### Task 0.2: src/ 包骨架 + 空 __init__.py

**Files:**
- Create: `src/minicpm_v_local/__init__.py`, `__main__.py`, 所有子目录的 `__init__.py`

**Steps:**

- [ ] **Step 1: 创建包目录结构**

```bash
mkdir -p src/minicpm_v_local/{runtime,server,pipeline}
touch src/minicpm_v_local/__init__.py
touch src/minicpm_v_local/{runtime,server,pipeline}/__init__.py
```

- [ ] **Step 2: `__init__.py` 内容**

`src/minicpm_v_local/__init__.py`:
```python
"""MiniCPM-V local skill — multi-backend image/video preprocessor."""

__version__ = "0.1.0"
```

- [ ] **Step 3: `__main__.py`**

```python
from minicpm_v_local.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
```

> 注意：此时 `cli.py` 还不存在；先放占位 import，会在 Task 8.1 实现。先添加占位 `cli.py`：

```python
def main() -> int:
    print("minicpm-v cli not yet implemented")
    return 0
```

- [ ] **Step 4: 安装可编辑包 + 验证**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
python -m minicpm_v_local
```

Expected: `minicpm-v cli not yet implemented`

- [ ] **Step 5: tests/ 骨架 + 第一个 smoke test**

`tests/unit/test_smoke.py`:
```python
import minicpm_v_local

def test_version_exposed():
    assert minicpm_v_local.__version__ == "0.1.0"
```

```bash
pytest tests/unit/test_smoke.py -v
```

Expected: 1 passed

- [ ] **Step 6: Commit**

```bash
git add src tests pyproject.toml
git commit -m "chore: package skeleton + smoke test"
```

---

## Phase 1：基础模块

### Task 1.1: paths.py

**Files:**
- Create: `src/minicpm_v_local/paths.py`
- Create: `tests/unit/test_paths.py`

**ResearchAgent brief**：
- spec § 14.2 路径常量表
- Python `pathlib.Path.home()` + `XDG_*` env vars 行为

**Steps:**

- [ ] **Step 1: 写测试**

```python
# tests/unit/test_paths.py
from minicpm_v_local import paths

def test_config_path_under_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    assert paths.config_dir() == tmp_path / ".config" / "minicpm-v-local"

def test_state_path():
    assert paths.state_file().name == "state.json"
    assert "minicpm-v-local" in str(paths.state_file())

def test_cache_dir_per_backend():
    assert "mlx" in str(paths.cache_dir("mlx"))
    assert "cpu" in str(paths.cache_dir("cpu"))

def test_lock_files_under_run():
    assert paths.cli_lock().parent == paths.run_dir()
    assert paths.download_lock().parent == paths.run_dir()
```

- [ ] **Step 2: 跑失败**

```bash
pytest tests/unit/test_paths.py -v
```

Expected: ImportError / AttributeError

- [ ] **Step 3: 实现 `paths.py`**

```python
"""Centralized filesystem paths. Spec §14.2."""
from __future__ import annotations
import os
from pathlib import Path

_APP = "minicpm-v-local"


def _xdg(env: str, default: str) -> Path:
    base = os.environ.get(env)
    return Path(base) if base else Path.home() / default


def config_dir() -> Path:
    return _xdg("XDG_CONFIG_HOME", ".config") / _APP


def config_file() -> Path:
    return config_dir() / "config.toml"


def run_dir() -> Path:
    return Path.home() / ".run" / _APP


def state_file() -> Path:
    return run_dir() / "state.json"


def cache_dir(backend: str) -> Path:
    return _xdg("XDG_CACHE_HOME", ".cache") / _APP / backend


def log_dir() -> Path:
    return _xdg("XDG_STATE_HOME", ".local/state") / _APP / "logs"


def cli_lock() -> Path:
    return run_dir() / "cli.lock"


def download_lock() -> Path:
    return run_dir() / "download.lock"


def frames_tmp_dir(run_id: str) -> Path:
    tmp = Path(os.environ.get("TMPDIR", "/tmp"))
    return tmp / _APP / f"frames-{run_id}"


def ensure_runtime_dirs() -> None:
    for d in (config_dir(), run_dir(), log_dir()):
        d.mkdir(parents=True, exist_ok=True)
```

- [ ] **Step 4: 跑通过**

```bash
pytest tests/unit/test_paths.py -v
```

Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/minicpm_v_local/paths.py tests/unit/test_paths.py
git commit -m "feat(paths): centralized filesystem paths"
```

### Task 1.2: config.py

**Files:**
- Create: `src/minicpm_v_local/config.py`
- Create: `tests/unit/test_config.py`

**ResearchAgent brief**:
- spec § 14.1 优先级链 + § 14.3 config.toml schema + § 14.4 env vars
- Python 3.11 `tomllib`（stdlib）API

**Steps:**

- [ ] **Step 1: 写测试**

```python
# tests/unit/test_config.py
from minicpm_v_local.config import Config, load

def test_defaults():
    cfg = Config.defaults()
    assert cfg.backend == "auto"
    assert cfg.quant == "4bit"
    assert cfg.idle_timeout == 300
    assert cfg.max_lifetime == 1800
    assert cfg.isolation is False

def test_env_overrides_toml(monkeypatch, tmp_path):
    toml = tmp_path / "config.toml"
    toml.write_text('backend = "cpu"\nidle_timeout = 60\n')
    monkeypatch.setenv("MINICPM_IDLE_TIMEOUT", "120")
    cfg = load(toml_path=toml, cli_overrides={})
    assert cfg.backend == "cpu"
    assert cfg.idle_timeout == 120

def test_cli_overrides_env(monkeypatch, tmp_path):
    toml = tmp_path / "config.toml"
    toml.write_text('backend = "cpu"\n')
    monkeypatch.setenv("MINICPM_BACKEND", "cuda")
    cfg = load(toml_path=toml, cli_overrides={"backend": "mlx"})
    assert cfg.backend == "mlx"

def test_missing_toml_returns_defaults(tmp_path):
    cfg = load(toml_path=tmp_path / "nope.toml", cli_overrides={})
    assert cfg.backend == "auto"

def test_video_section_nested(tmp_path):
    toml = tmp_path / "c.toml"
    toml.write_text('[video]\nscene_threshold = 0.5\nmax_frames = 30\n')
    cfg = load(toml_path=toml, cli_overrides={})
    assert cfg.video.scene_threshold == 0.5
    assert cfg.video.max_frames == 30
```

- [ ] **Step 2: 跑失败**

- [ ] **Step 3: 实现 `config.py`**

```python
"""Layered config: CLI > env > toml > default. Spec §14."""
from __future__ import annotations
import os
import tomllib
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

_ENV_MAP = {
    "MINICPM_BACKEND": "backend",
    "MINICPM_QUANT": "quant",
    "MINICPM_IDLE_TIMEOUT": ("idle_timeout", int),
    "MINICPM_MAX_LIFETIME": ("max_lifetime", int),
}


@dataclass
class VideoConfig:
    scene_threshold: float = 0.3
    fallback_interval: float = 10.0
    max_frames: int = 60
    scene_merge_similarity: float = 0.85


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port_range: tuple[int, int] = (8765, 8775)
    health_timeout: int = 60


@dataclass
class Config:
    backend: str = "auto"
    quant: str = "4bit"
    idle_timeout: int = 300
    max_lifetime: int = 1800
    isolation: bool = False
    isolation_mode: str = "auto"
    video: VideoConfig = field(default_factory=VideoConfig)
    server: ServerConfig = field(default_factory=ServerConfig)

    @classmethod
    def defaults(cls) -> "Config":
        return cls()


def _read_toml(p: Path) -> dict[str, Any]:
    if not p.exists():
        return {}
    return tomllib.loads(p.read_text())


def _apply_env(cfg: Config) -> None:
    for env_key, target in _ENV_MAP.items():
        val = os.environ.get(env_key)
        if val is None:
            continue
        if isinstance(target, tuple):
            attr, caster = target
            setattr(cfg, attr, caster(val))
        else:
            setattr(cfg, target, val)


def _apply_dict(cfg: Config, data: dict[str, Any]) -> None:
    for k, v in data.items():
        if k == "video" and isinstance(v, dict):
            for vk, vv in v.items():
                setattr(cfg.video, vk, vv)
        elif k == "server" and isinstance(v, dict):
            for sk, sv in v.items():
                if sk == "port_range":
                    setattr(cfg.server, sk, tuple(sv))
                else:
                    setattr(cfg.server, sk, sv)
        elif hasattr(cfg, k):
            setattr(cfg, k, v)


def load(toml_path: Path, cli_overrides: dict[str, Any]) -> Config:
    cfg = Config.defaults()
    _apply_dict(cfg, _read_toml(toml_path))
    _apply_env(cfg)
    _apply_dict(cfg, {k: v for k, v in cli_overrides.items() if v is not None})
    return cfg


def dump_toml(cfg: Config, p: Path) -> None:
    """Write minimal TOML; used by doctor after first setup."""
    lines = [
        f'backend = "{cfg.backend}"',
        f'quant = "{cfg.quant}"',
        f"idle_timeout = {cfg.idle_timeout}",
        f"max_lifetime = {cfg.max_lifetime}",
        f"isolation = {str(cfg.isolation).lower()}",
        f'isolation_mode = "{cfg.isolation_mode}"',
        "",
        "[video]",
        f"scene_threshold = {cfg.video.scene_threshold}",
        f"fallback_interval = {cfg.video.fallback_interval}",
        f"max_frames = {cfg.video.max_frames}",
        f"scene_merge_similarity = {cfg.video.scene_merge_similarity}",
    ]
    p.write_text("\n".join(lines) + "\n")
```

- [ ] **Step 4: 跑通过**

```bash
pytest tests/unit/test_config.py -v
```

Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/minicpm_v_local/config.py tests/unit/test_config.py
git commit -m "feat(config): layered config with CLI/env/toml priority"
```

---

## Phase 2：Runtime（平台探测 + Backend 抽象 + 三个具体 Backend）

### Task 2.1: runtime/detect.py

**Files:**
- Create: `src/minicpm_v_local/runtime/detect.py`
- Create: `tests/unit/runtime/test_detect.py`

**ResearchAgent brief**:
- spec § 6.2 探测顺序：Darwin+arm64 → mlx；Linux + nvidia-smi 成功 + CUDA≥12 → cuda；else → cpu
- `subprocess.run(["nvidia-smi", "--query-gpu=driver_version", ...], capture_output=True)` 的标准用法
- 注意：env `MINICPM_BACKEND` / CLI `--backend` 在 config 层处理，detect.py 不管覆盖逻辑

**Steps:**

- [ ] **Step 1: 写测试**

```python
# tests/unit/runtime/test_detect.py
import pytest
from minicpm_v_local.runtime import detect

@pytest.fixture
def fake_uname(monkeypatch):
    def _f(system, machine):
        monkeypatch.setattr(detect, "_uname", lambda: (system, machine))
    return _f

def test_mac_arm64(fake_uname):
    fake_uname("Darwin", "arm64")
    assert detect.auto_detect() == "mlx"

def test_mac_x86_returns_cpu(fake_uname):
    fake_uname("Darwin", "x86_64")
    assert detect.auto_detect() == "cpu"

def test_linux_with_nvidia(fake_uname, monkeypatch):
    fake_uname("Linux", "x86_64")
    monkeypatch.setattr(detect, "_has_cuda", lambda: True)
    assert detect.auto_detect() == "cuda"

def test_linux_without_nvidia(fake_uname, monkeypatch):
    fake_uname("Linux", "x86_64")
    monkeypatch.setattr(detect, "_has_cuda", lambda: False)
    assert detect.auto_detect() == "cpu"

def test_windows(fake_uname, monkeypatch):
    fake_uname("Windows", "AMD64")
    monkeypatch.setattr(detect, "_has_cuda", lambda: False)
    assert detect.auto_detect() == "cpu"
```

- [ ] **Step 2: 跑失败**

- [ ] **Step 3: 实现 `runtime/detect.py`**

```python
"""Platform detection. Spec §6.2."""
from __future__ import annotations
import platform
import subprocess
from typing import Literal

BackendTag = Literal["mlx", "cuda", "cpu"]


def _uname() -> tuple[str, str]:
    return platform.system(), platform.machine()


def _has_cuda() -> bool:
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        return r.returncode == 0 and bool(r.stdout.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def auto_detect() -> BackendTag:
    system, machine = _uname()
    if system == "Darwin" and machine in ("arm64", "aarch64"):
        return "mlx"
    if system == "Linux" and _has_cuda():
        return "cuda"
    return "cpu"


def resolve(requested: str) -> BackendTag:
    """`requested` from config: 'auto' or explicit tag."""
    if requested in ("mlx", "cuda", "cpu"):
        return requested  # type: ignore
    return auto_detect()
```

- [ ] **Step 4: 跑通过 + commit**

```bash
pytest tests/unit/runtime/test_detect.py -v
git add src/minicpm_v_local/runtime/detect.py tests/unit/runtime/test_detect.py
git commit -m "feat(runtime): platform/backend detection"
```

### Task 2.2: runtime/backend.py + factory.py

**Files:**
- Create: `src/minicpm_v_local/runtime/backend.py`, `factory.py`
- Create: `tests/unit/runtime/test_backend.py`

**ResearchAgent brief**:
- spec § 5（Backend 抽象类签名）+ § 6.1 模型 artifact 表

**Steps:**

- [ ] **Step 1: 写测试（抽象契约）**

```python
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
```

- [ ] **Step 2: 跑失败**

- [ ] **Step 3: 实现 `backend.py`**

```python
"""Backend abstraction. Spec §5, §6."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Literal

BackendTag = Literal["mlx", "cuda", "cpu"]


class Backend(ABC):
    tag: BackendTag
    quant: str

    @abstractmethod
    def artifact_id(self) -> str: ...

    @abstractmethod
    def launch_cmd(self, model_dir: str, port: int) -> list[str]: ...

    @abstractmethod
    def install_check(self) -> tuple[bool, str]:
        """Return (ok, message). False if backend deps missing."""

    def health_path(self) -> str:
        return "/health"
```

- [ ] **Step 4: 实现 `factory.py`**

```python
from minicpm_v_local.runtime.backend import Backend
from minicpm_v_local.runtime.mlx import MLXBackend
from minicpm_v_local.runtime.cuda import CUDABackend
from minicpm_v_local.runtime.cpu import CPUBackend


def get_backend(tag: str, quant: str) -> Backend:
    mapping = {"mlx": MLXBackend, "cuda": CUDABackend, "cpu": CPUBackend}
    if tag not in mapping:
        raise ValueError(f"unknown backend: {tag}")
    return mapping[tag](quant=quant)
```

- [ ] **Step 5: 三个 stub backend（minimum 满足测试）**

`src/minicpm_v_local/runtime/mlx.py`:
```python
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
```

`src/minicpm_v_local/runtime/cuda.py`:
```python
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
```

`src/minicpm_v_local/runtime/cpu.py`:
```python
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
```

- [ ] **Step 6: 跑通过 + commit**

```bash
pytest tests/unit/runtime/ -v
git add src/minicpm_v_local/runtime/ tests/unit/runtime/
git commit -m "feat(runtime): backend abstraction + mlx/cuda/cpu impls"
```

### Task 2.3: 每个 backend 的 launch_cmd 单独测试

**Files:**
- Create: `tests/unit/runtime/test_mlx.py`, `test_cuda.py`, `test_cpu.py`

**Steps:**

- [ ] **Step 1: 三个测试文件，每个验证 launch_cmd 拼接**

`tests/unit/runtime/test_mlx.py`:
```python
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
```

类似 cuda + cpu。

- [ ] **Step 2: 跑通过 + commit**

```bash
pytest tests/unit/runtime/ -v
git add tests/unit/runtime/test_{mlx,cuda,cpu}.py
git commit -m "test(runtime): per-backend launch_cmd assertions"
```

---

## Phase 3：Server lifecycle（manager / watchdog / isolation / state）

### Task 3.1: server/state.py — state.json schema + 原子读写

**Files:**
- Create: `src/minicpm_v_local/server/state.py`
- Create: `tests/unit/server/test_state.py`

**ResearchAgent brief**:
- spec § 8.5 state.json schema
- `os.replace()` 原子语义

**Steps:**

- [ ] **Step 1: 写测试**

```python
# tests/unit/server/test_state.py
from datetime import datetime, timezone
from minicpm_v_local.server.state import State, read_state, write_state, clear_state

def test_write_read_roundtrip(tmp_path):
    path = tmp_path / "state.json"
    s = State(
        backend="mlx", model_repo="mlx-community/MiniCPM-V-4.6-4bit",
        server_pid=1234, port=8765, started_at=datetime.now(timezone.utc),
        watchdog_pid=5678,
        last_used_at=datetime.now(timezone.utc),
        expire_at=datetime.now(timezone.utc),
        ttl_seconds=300, max_lifetime_at=None, keep=False,
        alive=True, cleanup_failed=False,
    )
    write_state(path, s)
    s2 = read_state(path)
    assert s2.server_pid == 1234

def test_read_missing_returns_none(tmp_path):
    assert read_state(tmp_path / "nope.json") is None

def test_clear_state(tmp_path):
    p = tmp_path / "state.json"
    p.write_text('{"alive": true}')
    clear_state(p)
    s = read_state(p)
    assert s is None or s.alive is False

def test_atomic_write_handles_concurrent_read(tmp_path):
    # half-written file simulated
    p = tmp_path / "state.json"
    p.write_text('{ "broken')
    assert read_state(p) is None  # tolerates corrupt JSON
```

- [ ] **Step 2: 跑失败**

- [ ] **Step 3: 实现 `state.py`**

```python
"""state.json schema + atomic IO. Spec §8.5."""
from __future__ import annotations
import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

SCHEMA_VERSION = 1


@dataclass
class State:
    backend: str
    model_repo: str
    server_pid: int
    port: int
    started_at: datetime
    watchdog_pid: int
    last_used_at: datetime
    expire_at: datetime
    ttl_seconds: int
    max_lifetime_at: Optional[datetime]
    keep: bool
    alive: bool
    cleanup_failed: bool
    isolation_mode: Optional[str] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, datetime):
                d[k] = v.isoformat()
        d["schema_version"] = SCHEMA_VERSION
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "State":
        d = dict(d)
        d.pop("schema_version", None)
        for k in ("started_at", "last_used_at", "expire_at", "max_lifetime_at"):
            if d.get(k):
                d[k] = datetime.fromisoformat(d[k])
            else:
                d[k] = None
        return cls(**d)


def read_state(path: Path) -> Optional[State]:
    if not path.exists():
        return None
    try:
        return State.from_dict(json.loads(path.read_text()))
    except (json.JSONDecodeError, TypeError, KeyError):
        return None


def write_state(path: Path, s: State) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(s.to_dict(), indent=2))
    os.replace(tmp, path)


def clear_state(path: Path) -> None:
    if path.exists():
        path.unlink()
```

- [ ] **Step 4: 跑通过 + commit**

```bash
pytest tests/unit/server/test_state.py -v
git add src/minicpm_v_local/server/state.py tests/unit/server/test_state.py
git commit -m "feat(server): state.json schema + atomic IO"
```

### Task 3.2: server/isolation.py

**Files:**
- Create: `src/minicpm_v_local/server/isolation.py`
- Create: `tests/unit/server/test_isolation.py`

**ResearchAgent brief**:
- spec § 9.2 / 9.3 隔离化方案 + profile 要点
- `sandbox-exec` macOS profile 语法 (`(version 1)(allow default)(deny file-write*) …`)
- `bwrap` 关键 flag (`--unshare-all --share-net --bind / / --dev-bind /dev/nvidia*`)

**Steps:**

- [ ] **Step 1: 写测试**

```python
# tests/unit/server/test_isolation.py
import sys
from minicpm_v_local.server import isolation

def test_no_isolation_returns_cmd_as_is():
    cmd = ["python", "-m", "mlx_vlm.server"]
    wrapped = isolation.wrap(cmd, mode="none")
    assert wrapped == cmd

def test_mac_sandbox_exec_wraps(monkeypatch):
    monkeypatch.setattr(isolation, "_platform", lambda: "Darwin")
    wrapped = isolation.wrap(["python", "x"], mode="auto")
    assert wrapped[0] == "sandbox-exec"

def test_linux_bwrap_wraps(monkeypatch):
    monkeypatch.setattr(isolation, "_platform", lambda: "Linux")
    wrapped = isolation.wrap(["python", "x"], mode="auto")
    assert wrapped[0] == "bwrap"

def test_unsupported_platform_falls_back(monkeypatch):
    monkeypatch.setattr(isolation, "_platform", lambda: "Windows")
    wrapped = isolation.wrap(["python", "x"], mode="auto")
    assert wrapped[0] == "python"  # 退化为无沙箱
```

- [ ] **Step 2: 跑失败**

- [ ] **Step 3: 实现 `isolation.py`**

```python
"""Sandbox wrappers. Spec §9."""
from __future__ import annotations
import platform
import tempfile
from pathlib import Path
from typing import Optional

# Minimal mac profile: allow default, restrict home writes outside cache.
_MAC_PROFILE = """(version 1)
(allow default)
(deny file-write*
  (subpath (string-append (param "HOME") "/Documents"))
  (subpath (string-append (param "HOME") "/Desktop")))
"""


def _platform() -> str:
    return platform.system()


def _mac_wrap(cmd: list[str]) -> list[str]:
    prof = Path(tempfile.gettempdir()) / "minicpm-v-mac.sb"
    if not prof.exists():
        prof.write_text(_MAC_PROFILE)
    return ["sandbox-exec", "-f", str(prof), *cmd]


def _linux_wrap(cmd: list[str]) -> list[str]:
    base = [
        "bwrap",
        "--unshare-all",
        "--share-net",
        "--bind", "/", "/",
        "--proc", "/proc",
        "--dev", "/dev",
    ]
    # GPU 设备透传（容错：不存在则忽略）
    for dev in ("/dev/nvidia0", "/dev/nvidiactl", "/dev/nvidia-uvm"):
        if Path(dev).exists():
            base += ["--dev-bind", dev, dev]
    return [*base, *cmd]


def wrap(cmd: list[str], mode: str) -> list[str]:
    """Wrap a command in a sandbox.

    mode: 'none' | 'auto' | 'sandbox-exec' | 'bwrap'
    """
    if mode == "none":
        return cmd
    sys = _platform()
    if mode in ("auto", "sandbox-exec") and sys == "Darwin":
        return _mac_wrap(cmd)
    if mode in ("auto", "bwrap") and sys == "Linux":
        return _linux_wrap(cmd)
    return cmd  # 无支持平台：退化


def available_mode() -> Optional[str]:
    sys = _platform()
    if sys == "Darwin":
        return "sandbox-exec"
    if sys == "Linux":
        from shutil import which
        return "bwrap" if which("bwrap") else None
    return None
```

- [ ] **Step 4: 跑通过 + commit**

```bash
pytest tests/unit/server/test_isolation.py -v
git add src/minicpm_v_local/server/isolation.py tests/unit/server/test_isolation.py
git commit -m "feat(server): sandbox wrappers (sandbox-exec / bwrap)"
```

### Task 3.3: server/manager.py — spawn / health / lifecycle

**Files:**
- Create: `src/minicpm_v_local/server/manager.py`
- Create: `tests/unit/server/test_manager.py`

**ResearchAgent brief**:
- spec § 8.3 销毁机制；§ 12.2 cold path；§ 13 错误处理
- Python `subprocess.Popen(start_new_session=True)` + `os.killpg`
- 端口检测：`socket.socket(SO_REUSEADDR)` + try bind

**Steps:**

(详细 step-by-step 类似前面 — 因长度限制此处给出关键签名 + 行为契约，dev agent 按 spec 补完)

```python
# src/minicpm_v_local/server/manager.py
"""Server lifecycle. Spec §8, §12.2."""
from __future__ import annotations
import os
import signal
import socket
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import httpx

from minicpm_v_local import paths
from minicpm_v_local.runtime.backend import Backend
from minicpm_v_local.server import isolation
from minicpm_v_local.server.state import State, read_state, write_state, clear_state


def _free_port(port_range: tuple[int, int]) -> int:
    for p in range(port_range[0], port_range[1] + 1):
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", p))
                return p
            except OSError:
                continue
    raise RuntimeError(f"no free port in {port_range}")


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def _wait_health(url: str, timeout: int) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = httpx.get(url, timeout=2.0)
            if r.status_code == 200:
                return True
        except httpx.RequestError:
            pass
        time.sleep(0.5)
    return False


def ensure_warm(
    backend: Backend, model_dir: Path, *,
    port_range: tuple[int, int], health_timeout: int,
    ttl_seconds: int, max_lifetime: int, keep: bool,
    isolation_mode: Optional[str],
) -> State:
    """Idempotent: return State of running server, spawning if needed."""
    state_path = paths.state_file()
    existing = read_state(state_path)
    if existing and existing.alive and _pid_alive(existing.server_pid):
        url = f"http://127.0.0.1:{existing.port}{backend.health_path()}"
        if _wait_health(url, timeout=2):
            return _bump(state_path, existing, ttl_seconds, keep)

    return _spawn(
        backend, model_dir,
        port_range=port_range, health_timeout=health_timeout,
        ttl_seconds=ttl_seconds, max_lifetime=max_lifetime,
        keep=keep, isolation_mode=isolation_mode,
    )


def _spawn(
    backend: Backend, model_dir: Path, *,
    port_range, health_timeout, ttl_seconds, max_lifetime, keep, isolation_mode,
) -> State:
    port = _free_port(port_range)
    cmd = backend.launch_cmd(str(model_dir), port)
    if isolation_mode and isolation_mode != "none":
        cmd = isolation.wrap(cmd, mode=isolation_mode)

    log_path = paths.log_dir() / f"server-{port}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_f = open(log_path, "a")
    proc = subprocess.Popen(
        cmd, stdout=log_f, stderr=subprocess.STDOUT,
        start_new_session=True,
    )

    url = f"http://127.0.0.1:{port}{backend.health_path()}"
    if not _wait_health(url, timeout=health_timeout):
        proc.terminate()
        raise RuntimeError(f"server health check failed within {health_timeout}s; see {log_path}")

    now = datetime.now(timezone.utc)
    state = State(
        backend=backend.tag,
        model_repo=backend.artifact_id(),
        server_pid=proc.pid, port=port, started_at=now,
        watchdog_pid=0,  # 由 watchdog spawn 后填
        last_used_at=now,
        expire_at=now + timedelta(seconds=ttl_seconds),
        ttl_seconds=ttl_seconds,
        max_lifetime_at=now + timedelta(seconds=max_lifetime) if max_lifetime > 0 else None,
        keep=keep, alive=True, cleanup_failed=False,
        isolation_mode=isolation_mode,
    )
    write_state(paths.state_file(), state)

    # spawn watchdog
    wd = subprocess.Popen(
        ["python", "-m", "minicpm_v_local.server.watchdog"],
        start_new_session=True,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    state.watchdog_pid = wd.pid
    write_state(paths.state_file(), state)
    return state


def _bump(state_path: Path, s: State, ttl_seconds: int, keep: bool) -> State:
    now = datetime.now(timezone.utc)
    s.last_used_at = now
    s.keep = keep
    if not keep:
        new_expire = now + timedelta(seconds=ttl_seconds)
        if s.max_lifetime_at and new_expire > s.max_lifetime_at:
            new_expire = s.max_lifetime_at
        s.expire_at = new_expire
        s.ttl_seconds = ttl_seconds
    write_state(state_path, s)
    return s


def stop(force: bool = False) -> None:
    """Manual stop. Spec §8.3."""
    s = read_state(paths.state_file())
    if not s or not s.alive:
        return
    try:
        os.kill(s.server_pid, signal.SIGTERM)
        for _ in range(50):
            if not _pid_alive(s.server_pid):
                break
            time.sleep(0.1)
        if _pid_alive(s.server_pid):
            os.kill(s.server_pid, signal.SIGKILL)
        if s.watchdog_pid and _pid_alive(s.watchdog_pid):
            os.kill(s.watchdog_pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    clear_state(paths.state_file())
```

**测试要点**（dev agent 按 spec §15.2 完成）：
- mock `_free_port` 返回 8765
- mock `subprocess.Popen` 返回假 proc（pid=12345）
- mock `_wait_health` 返回 True
- 验证 state.json 写入正确
- 验证 ensure_warm 第二次调用是 warm path（不重新 spawn）
- 验证 stop() 顺序 TERM→KILL
- 验证 max_lifetime ceiling 不会被 ttl 穿越

- [ ] **Step**：写测试 → 跑失败 → 实现 → 跑通过 → commit。

```bash
git add src/minicpm_v_local/server/manager.py tests/unit/server/test_manager.py
git commit -m "feat(server): lifecycle manager (spawn/health/stop/ttl/max-lifetime)"
```

### Task 3.4: server/watchdog.py

**Files:**
- Create: `src/minicpm_v_local/server/watchdog.py`
- Create: `tests/unit/server/test_watchdog.py`

```python
"""Sidecar watchdog process. Spec §12.3."""
from __future__ import annotations
import os
import signal
import sys
import time
from datetime import datetime, timezone

from minicpm_v_local import paths
from minicpm_v_local.server.state import read_state, write_state, clear_state

CHECK_INTERVAL_S = 10
TERM_GRACE_S = 5


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def main() -> int:
    while True:
        s = read_state(paths.state_file())
        if not s or not s.alive:
            return 0
        now = datetime.now(timezone.utc)
        expired = (not s.keep) and now >= s.expire_at
        over_lifetime = s.max_lifetime_at is not None and now >= s.max_lifetime_at
        if expired or over_lifetime:
            return _kill_and_exit(s.server_pid)
        time.sleep(CHECK_INTERVAL_S)


def _kill_and_exit(server_pid: int) -> int:
    if _pid_alive(server_pid):
        try:
            os.kill(server_pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        deadline = time.time() + TERM_GRACE_S
        while time.time() < deadline and _pid_alive(server_pid):
            time.sleep(0.2)
        if _pid_alive(server_pid):
            try:
                os.kill(server_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    # cleanup_failed 检测（简化 v1：仅检测进程是否消失）
    failed = _pid_alive(server_pid)
    s = read_state(paths.state_file())
    if s:
        s.alive = False
        s.cleanup_failed = failed
        write_state(paths.state_file(), s)
    else:
        clear_state(paths.state_file())
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

**测试要点**：
- mock state with expired expire_at → 验证调用 `_kill_and_exit`
- mock state with future expire_at → 验证循环继续
- mock state.keep=True → 验证永不杀
- mock `_pid_alive` 始终 True → 验证 cleanup_failed=True 写回

- [ ] **Steps**：测试 → 失败 → 实现 → 通过 → commit。

```bash
git add src/minicpm_v_local/server/watchdog.py tests/unit/server/test_watchdog.py
git commit -m "feat(server): sidecar idle watchdog"
```

---

## Phase 4：下载与客户端

### Task 4.1: download.py — HF snapshot + lockfile

**Files:**
- Create: `src/minicpm_v_local/download.py`
- Create: `tests/unit/test_download.py`

**ResearchAgent brief**:
- spec § 14 路径规范
- `huggingface_hub.snapshot_download(repo_id, cache_dir, local_dir, …)` 签名
- 文件锁：`fcntl.flock(LOCK_EX | LOCK_NB)` for Linux/Mac

```python
"""Model download with lockfile. Spec §13."""
from __future__ import annotations
import fcntl
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from huggingface_hub import snapshot_download

from minicpm_v_local import paths


@contextmanager
def _download_lock() -> Iterator[None]:
    lock_path = paths.download_lock()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def ensure_model(repo_id: str, backend: str, *, allow_patterns: list[str] | None = None) -> Path:
    """Download (or verify cached) model. Returns local model dir."""
    target = paths.cache_dir(backend) / repo_id.replace("/", "__")
    target.mkdir(parents=True, exist_ok=True)
    with _download_lock():
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(target),
            allow_patterns=allow_patterns,
            local_dir_use_symlinks=False,
        )
    return target
```

**测试**:
- mock `snapshot_download` → 验证 target 路径正确、lock 文件被创建/释放
- 并发模拟：起两个线程同时调，验证 lock 序列化（用 threading.Event）

- [ ] **Steps**：测试 → 失败 → 实现 → 通过 → commit。

### Task 4.2: client.py — OpenAI HTTP 客户端

**Files:**
- Create: `src/minicpm_v_local/client.py`
- Create: `tests/unit/test_client.py`

```python
"""OpenAI-compatible HTTP client. Spec §10."""
from __future__ import annotations
import base64
from pathlib import Path
from typing import Optional

import httpx


class VLMClient:
    def __init__(self, base_url: str, timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(timeout=timeout)

    @staticmethod
    def _encode_image(path: Path) -> str:
        suffix = path.suffix.lstrip(".").lower() or "jpeg"
        if suffix == "jpg":
            suffix = "jpeg"
        b64 = base64.b64encode(path.read_bytes()).decode()
        return f"data:image/{suffix};base64,{b64}"

    def caption(self, image: Path, prompt: str, *, model: str = "minicpm-v") -> str:
        payload = {
            "model": model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": self._encode_image(image)}},
                    {"type": "text", "text": prompt},
                ],
            }],
            "max_tokens": 512,
        }
        r = self._client.post(f"{self.base_url}/v1/chat/completions", json=payload)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    def close(self) -> None:
        self._client.close()
```

**测试**:
- 用 `httpx.MockTransport` 拦截 POST，验证 body 结构正确
- 验证 image_url 是 data URI（`startswith("data:image/")`）
- 验证 caption 返回 string

- [ ] **Steps**：测试 → 失败 → 实现 → 通过 → commit。

---

## Phase 5：Pipeline（图像 + 视频）

### Task 5.1: pipeline/image.py

**Files:**
- Create: `src/minicpm_v_local/pipeline/image.py`
- Create: `tests/unit/pipeline/test_image.py`

```python
"""Single-image pipeline. Spec §10.3."""
from __future__ import annotations
import hashlib
import time
from pathlib import Path

from minicpm_v_local.client import VLMClient

DEFAULT_PROMPT = "Describe the image in detail. List any visible objects and text."


def caption_image(client: VLMClient, image_path: Path, *,
                  model: str, prompt: str = DEFAULT_PROMPT) -> dict:
    t0 = time.monotonic()
    sha = hashlib.sha256(image_path.read_bytes()).hexdigest()
    text = client.caption(image_path, prompt=prompt, model=model)
    dt = int((time.monotonic() - t0) * 1000)
    return {
        "version": 1,
        "input": {"path": str(image_path), "sha256": sha},
        "model": model,
        "result": {"caption": text, "objects": [], "ocr_text": None},
        "timing_ms": {"load": 0, "infer": dt},
    }
```

**测试**:
- mock client.caption → 返回 fixed string
- 验证 JSON 结构匹配 spec §10.3
- 验证 sha256 计算正确

- [ ] **Steps**：测试 → 失败 → 实现 → 通过 → commit。

### Task 5.2: pipeline/video.py — ffmpeg 抽帧 + 串行 caption + scene 聚合

**Files:**
- Create: `src/minicpm_v_local/pipeline/video.py`
- Create: `tests/unit/pipeline/test_video.py`

**ResearchAgent brief**:
- spec § 12.4 视频处理 + 抽帧参数表
- `ffprobe -v error -show_entries format=duration:stream=r_frame_rate -of json`
- `ffmpeg -i in.mp4 -vf "select='gt(scene,0.3)',showinfo" -vsync vfr out_%04d.jpg`
- 难点：scene 合并 = 给每个 caption 算 embedding（v1 用简单 token 重叠相似度，不依赖 sentence-transformers，保持 zero-ML-dep）

```python
"""Video pipeline. Spec §10.4, §12.4."""
from __future__ import annotations
import json
import re
import subprocess
import time
import uuid
import hashlib
from dataclasses import dataclass
from pathlib import Path

from minicpm_v_local import paths
from minicpm_v_local.client import VLMClient
from minicpm_v_local.config import VideoConfig

DEFAULT_PROMPT = "Describe what's happening in this frame in one sentence."


@dataclass
class Frame:
    t: float
    path: Path
    caption: str | None = None
    error: str | None = None


def probe(video: Path) -> dict:
    r = subprocess.run(
        ["ffprobe", "-v", "error",
         "-show_entries", "format=duration:stream=r_frame_rate",
         "-of", "json", str(video)],
        capture_output=True, text=True, check=True,
    )
    data = json.loads(r.stdout)
    duration = float(data["format"]["duration"])
    fps_str = data["streams"][0].get("r_frame_rate", "30/1")
    num, den = fps_str.split("/")
    fps = float(num) / float(den) if float(den) > 0 else 30.0
    return {"duration": duration, "fps": fps}


def extract_keyframes(video: Path, *, cfg: VideoConfig) -> list[Frame]:
    run_id = uuid.uuid4().hex[:8]
    frames_dir = paths.frames_tmp_dir(run_id)
    frames_dir.mkdir(parents=True, exist_ok=True)

    # scene-change frames
    pattern = str(frames_dir / "scene_%04d.jpg")
    log = subprocess.run(
        ["ffmpeg", "-y", "-i", str(video),
         "-vf", f"select='gt(scene,{cfg.scene_threshold})',showinfo",
         "-vsync", "vfr", pattern],
        capture_output=True, text=True,
    )

    times = []
    for line in log.stderr.splitlines():
        m = re.search(r"pts_time:([\d.]+)", line)
        if m:
            times.append(float(m.group(1)))

    files = sorted(frames_dir.glob("scene_*.jpg"))
    frames = [Frame(t=t, path=f) for t, f in zip(times, files)]

    # uniform fallback (确保每 fallback_interval 至少一帧)
    info = probe(video)
    duration = info["duration"]
    interval = cfg.fallback_interval
    target_count = int(duration / interval)
    if len(frames) < target_count:
        for i in range(target_count):
            t = i * interval
            out = frames_dir / f"uniform_{i:04d}.jpg"
            subprocess.run(
                ["ffmpeg", "-y", "-ss", str(t), "-i", str(video),
                 "-frames:v", "1", str(out)],
                capture_output=True, check=False,
            )
            if out.exists():
                frames.append(Frame(t=t, path=out))

    frames.sort(key=lambda f: f.t)
    if len(frames) > cfg.max_frames:
        # 等距下采样
        step = len(frames) / cfg.max_frames
        frames = [frames[int(i * step)] for i in range(cfg.max_frames)]
    return frames


def _token_set(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower()))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


def merge_scenes(frames: list[Frame], *, threshold: float) -> list[dict]:
    if not frames:
        return []
    scenes = []
    cur = {"start": frames[0].t, "end": frames[0].t, "summary": frames[0].caption or "",
           "frame_indices": [0], "_tokens": _token_set(frames[0].caption or "")}
    for i, fr in enumerate(frames[1:], start=1):
        cap = fr.caption or ""
        toks = _token_set(cap)
        sim = _jaccard(cur["_tokens"], toks)
        if sim >= threshold:
            cur["end"] = fr.t
            cur["frame_indices"].append(i)
            cur["_tokens"] |= toks
        else:
            scenes.append({k: v for k, v in cur.items() if not k.startswith("_")})
            cur = {"start": fr.t, "end": fr.t, "summary": cap, "frame_indices": [i],
                   "_tokens": toks}
    scenes.append({k: v for k, v in cur.items() if not k.startswith("_")})
    return scenes


def process_video(
    client: VLMClient, video: Path, *,
    model: str, cfg: VideoConfig, prompt: str = DEFAULT_PROMPT,
    on_frame_done=None,
) -> dict:
    t0 = time.monotonic()
    info = probe(video)
    sha = hashlib.sha256(video.read_bytes()).hexdigest()
    t_ffmpeg_start = time.monotonic()
    frames = extract_keyframes(video, cfg=cfg)
    ffmpeg_ms = int((time.monotonic() - t_ffmpeg_start) * 1000)

    for fr in frames:
        try:
            fr.caption = client.caption(fr.path, prompt=prompt, model=model)
        except Exception as e:
            fr.error = str(e)
        if on_frame_done:
            on_frame_done(fr)

    scenes = merge_scenes(frames, threshold=cfg.scene_merge_similarity)
    infer_ms = int((time.monotonic() - t0) * 1000) - ffmpeg_ms

    return {
        "version": 1,
        "input": {"path": str(video), "sha256": sha, "duration_s": info["duration"], "fps": info["fps"]},
        "model": model,
        "frames": [
            {"t": fr.t, "path": str(fr.path), "caption": fr.caption, "error": fr.error}
            for fr in frames
        ],
        "scenes": scenes,
        "timing_ms": {"ffmpeg": ffmpeg_ms, "infer_total": infer_ms},
    }
```

**测试**:
- mock `subprocess.run` for ffprobe / ffmpeg
- mock client.caption 返回 "a cat" 或 "a dog" 交替 → 验证 jaccard 阈值边界（0.84/0.85/0.86）触发不同 scene 数量
- 验证 max_frames 下采样
- 验证 frame.error 不阻断整体输出

- [ ] **Steps**：测试 → 失败 → 实现 → 通过 → commit。

```bash
git add src/minicpm_v_local/pipeline/ tests/unit/pipeline/
git commit -m "feat(pipeline): image + video (ffmpeg keyframe extraction + scene merge)"
```

---

## Phase 6：Doctor — 8 步自检

### Task 6.1: doctor.py

**Files:**
- Create: `src/minicpm_v_local/doctor.py`
- Create: `tests/unit/test_doctor.py`

**ResearchAgent brief**:
- spec § 12.1 doctor 8 步
- 交互式输入：`input()` 处理；测试时 mock

```python
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
    ans = input(f"{question} [{default}]: ").strip()
    return ans or default


def run(prompter: Prompter = _default_prompt) -> int:
    print("Running minicpm-v doctor...")

    # 1. detect
    tag = detect.auto_detect()
    print(f"  [1/8] backend tag: {tag}")

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
    if tag == "mlx":
        quant = prompter("Quantization (4bit/5bit/8bit/bf16)", "4bit")
    elif tag == "cuda":
        quant = "bf16"
    else:
        quant = "Q4_K_M"
    print(f"  [4/8] quant: {quant}")

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
```

**测试**:
- mock `detect.auto_detect`, `backend.install_check`, `shutil.which`, `ensure_model`, `dump_toml`
- 用 `prompter=lambda q, d: 'mock'` 覆盖交互
- 验证 config.toml 内容正确
- 验证缺 ffmpeg 时退出 2

- [ ] **Steps**：测试 → 失败 → 实现 → 通过 → commit。

---

## Phase 7：CLI + Skill 壳

### Task 7.1: cli.py — argparse 入口 + 命令分派

**Files:**
- Modify: `src/minicpm_v_local/cli.py`（覆盖之前的占位）
- Create: `tests/unit/test_cli.py`

```python
"""CLI entry. Spec §10."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

from minicpm_v_local import paths, doctor
from minicpm_v_local.config import load
from minicpm_v_local.runtime import detect
from minicpm_v_local.runtime.factory import get_backend
from minicpm_v_local.server import manager
from minicpm_v_local.server.state import read_state
from minicpm_v_local.download import ensure_model
from minicpm_v_local.client import VLMClient
from minicpm_v_local.pipeline.image import caption_image
from minicpm_v_local.pipeline.video import process_video


def _add_common(p):
    p.add_argument("--backend", choices=["auto", "mlx", "cuda", "cpu"], default=None)
    p.add_argument("--quant", default=None)
    p.add_argument("--ttl", type=int, default=None, help="保活秒数；0 = 立即销毁")
    p.add_argument("--max-lifetime", type=int, default=None)
    p.add_argument("--keep", action="store_true")
    p.add_argument("--isolated", action="store_true")
    p.add_argument("--output", choices=["json", "jsonl"], default="json")
    p.add_argument("--prompt", default=None)


def build_parser():
    parser = argparse.ArgumentParser(prog="minicpm-v")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_doc = sub.add_parser("doctor")
    p_doc.add_argument("--reset", action="store_true")

    p_img = sub.add_parser("image")
    p_img.add_argument("path", type=Path)
    _add_common(p_img)

    p_vid = sub.add_parser("video")
    p_vid.add_argument("path", type=Path)
    _add_common(p_vid)

    sub.add_parser("status")
    p_stop = sub.add_parser("stop")
    p_stop.add_argument("--force", action="store_true")
    return parser


def _cfg_with_overrides(args) -> "Config":
    overrides = {}
    if args.backend: overrides["backend"] = args.backend
    if args.quant: overrides["quant"] = args.quant
    if args.ttl is not None: overrides["idle_timeout"] = args.ttl
    if args.max_lifetime is not None: overrides["max_lifetime"] = args.max_lifetime
    if args.isolated: overrides["isolation"] = True
    return load(paths.config_file(), cli_overrides=overrides)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.cmd == "doctor":
        if args.reset and paths.config_file().exists():
            paths.config_file().unlink()
        return doctor.run()

    if args.cmd == "status":
        s = read_state(paths.state_file())
        print(json.dumps(s.to_dict() if s else {"alive": False}, indent=2))
        return 0

    if args.cmd == "stop":
        manager.stop(force=args.force)
        return 0

    # image / video
    if not paths.config_file().exists():
        return doctor.run()

    cfg = _cfg_with_overrides(args)
    tag = detect.resolve(cfg.backend)
    backend = get_backend(tag, quant=cfg.quant)

    model_dir = ensure_model(backend.artifact_id(), backend=tag)

    state = manager.ensure_warm(
        backend, model_dir,
        port_range=cfg.server.port_range,
        health_timeout=cfg.server.health_timeout,
        ttl_seconds=args.ttl if args.ttl is not None else cfg.idle_timeout,
        max_lifetime=cfg.max_lifetime,
        keep=args.keep,
        isolation_mode=(cfg.isolation_mode if cfg.isolation else "none"),
    )

    client = VLMClient(base_url=f"http://127.0.0.1:{state.port}")
    try:
        if args.cmd == "image":
            result = caption_image(client, args.path, model=backend.artifact_id(),
                                   prompt=args.prompt or _default_image_prompt())
        else:
            result = process_video(client, args.path, model=backend.artifact_id(),
                                   cfg=cfg.video, prompt=args.prompt or _default_video_prompt())
    finally:
        client.close()

    if args.ttl == 0:
        manager.stop()

    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def _default_image_prompt() -> str:
    from minicpm_v_local.pipeline.image import DEFAULT_PROMPT
    return DEFAULT_PROMPT


def _default_video_prompt() -> str:
    from minicpm_v_local.pipeline.video import DEFAULT_PROMPT
    return DEFAULT_PROMPT
```

**测试**:
- mock 所有下层模块（doctor.run / detect / backend / ensure_model / manager / VLMClient / pipeline）
- 验证 `image foo.jpg` 调用链
- 验证 `--ttl 0` 调完后调用 `manager.stop()`
- 验证 `--keep` 传到 `ensure_warm`
- 验证缺 config 时自动走 doctor

- [ ] **Steps**：测试 → 失败 → 实现 → 通过 → commit。

### Task 7.2: Skill 壳 — SKILL.md + run.sh + install.sh

**Files:**
- Create: `skill/SKILL.md`
- Create: `skill/scripts/run.sh`
- Create: `skill/install.sh`

`skill/SKILL.md`:
```markdown
---
name: minicpm-v
description: |
  Local visual preprocessing using MiniCPM-V 4.6 (1.3B). Captions images and
  video timelines locally without sending pixels to the main model.
  Trigger when the user asks to analyze, describe, summarize, or extract
  information from images or videos and a local fast pass would save tokens.
---

# minicpm-v skill

## When to use

- 用户给了图片或视频，让你描述、总结、抽信息
- 你需要"看一眼"图但不想把像素发给主模型
- 视频处理（默认主线）

## When NOT to use

- 用户要你"用自己的视觉能力"
- 需要图表里的精确数值（OCR 精度不一定够）
- 需要主模型的世界知识（地标识别、人物识别等）

## How to call

单图：
```
bash scripts/run.sh image <path> [--ttl <sec>]
```
读 stdout JSON，使用 `result.caption`。

视频：
```
bash scripts/run.sh video <path> [--ttl <sec>]
```
读 stdout JSON，使用 `scenes[]` 做时间轴定位。

## TTL hint

- 还会继续问图：`--ttl 600`
- 这是最后一次：`--ttl 0`
- 不传：使用默认 300s

## ⚠️ 自动卸载

本地 server 会在 **5 分钟无请求**后自动从内存/显存卸载，避免占用资源。
下次调用会自动重新加载（cold start ≈ 3–15s）。
```

`skill/scripts/run.sh`:
```bash
#!/usr/bin/env bash
set -euo pipefail
exec python -m minicpm_v_local "$@"
```

`skill/install.sh`:
```bash
#!/usr/bin/env bash
set -euo pipefail
TARGET="$HOME/.claude/skills/minicpm-v"
mkdir -p "$TARGET/scripts"
cp "$(dirname "$0")/SKILL.md" "$TARGET/SKILL.md"
cp "$(dirname "$0")/scripts/run.sh" "$TARGET/scripts/run.sh"
chmod +x "$TARGET/scripts/run.sh"
echo "Skill installed to $TARGET"
```

- [ ] **Steps**：创建文件 → `chmod +x` → 验证 `bash skill/install.sh` 不报错 → commit。

```bash
git add skill/
git commit -m "feat(skill): SKILL.md + run.sh + installer"
```

---

## Phase 8：集成测试 + 验收

### Task 8.1: integration test — CPU 端到端

**Files:**
- Create: `tests/integration/test_end_to_end_cpu.py`
- Create: `tests/integration/test_lifecycle.py`
- Create: `tests/fixtures/sample.jpg` (any small jpg)
- Create: `tests/fixtures/sample-5s.mp4` (生成或下载一个 5s 测试视频)

**Steps:**

- [ ] **Step 1: 生成 sample fixtures**

```bash
# sample.jpg: 用 PIL 生成一张 256x256 红色块
python -c "from PIL import Image; Image.new('RGB',(256,256),'red').save('tests/fixtures/sample.jpg')"

# sample-5s.mp4: ffmpeg 生成 5s 色块视频（需系统装了 ffmpeg）
ffmpeg -y -f lavfi -i "testsrc=duration=5:size=256x256:rate=10" -pix_fmt yuv420p tests/fixtures/sample-5s.mp4
```

- [ ] **Step 2: lifecycle integration — 用真 subprocess 起一个假 server**

```python
# tests/integration/test_lifecycle.py
import subprocess
import time
from minicpm_v_local.server import manager
from minicpm_v_local.server.state import read_state
from minicpm_v_local import paths

# 用一个简单的 python HTTP server 模拟 backend
FAKE_SERVER = """
import http.server, socketserver, sys
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/health':
            self.send_response(200); self.end_headers(); return
        self.send_response(404); self.end_headers()
    def log_message(self, *a): pass
port = int(sys.argv[1])
with socketserver.TCPServer(('127.0.0.1', port), H) as httpd:
    httpd.serve_forever()
"""

class FakeBackend:
    tag = "cpu"; quant = "Q4_K_M"
    def artifact_id(self): return "fake/model"
    def launch_cmd(self, model_dir, port):
        return ["python", "-c", FAKE_SERVER, str(port)]
    def install_check(self): return True, ""
    def health_path(self): return "/health"

def test_warm_path_idempotent(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "state_file", lambda: tmp_path / "state.json")
    monkeypatch.setattr(paths, "run_dir", lambda: tmp_path)
    monkeypatch.setattr(paths, "log_dir", lambda: tmp_path)
    b = FakeBackend()
    s1 = manager.ensure_warm(b, tmp_path, port_range=(9700, 9710),
                              health_timeout=10, ttl_seconds=60,
                              max_lifetime=600, keep=False, isolation_mode="none")
    s2 = manager.ensure_warm(b, tmp_path, port_range=(9700, 9710),
                              health_timeout=10, ttl_seconds=60,
                              max_lifetime=600, keep=False, isolation_mode="none")
    assert s1.server_pid == s2.server_pid  # warm path
    manager.stop()
```

- [ ] **Step 3: end-to-end image with HTTP mock**

(详见 dev agent 实现：用 `httpx.MockTransport` 替代 client，验证 pipeline → JSON 输出)

- [ ] **Step 4: 跑测试**

```bash
pytest tests/integration/ -v
```

- [ ] **Step 5: Commit**

```bash
git add tests/integration/ tests/fixtures/
git commit -m "test(integration): lifecycle + e2e with fake backend"
```

### Task 8.2: 覆盖率验证

**Steps:**

- [ ] **Step 1: 跑覆盖率**

```bash
pytest --cov=minicpm_v_local --cov-report=term-missing tests/
```

Expected:
- runtime/server/pipeline/download/config 各模块 ≥ 80%
- cli/doctor ≥ 60%

- [ ] **Step 2: 补缺失的测试**（如果哪个核心模块 < 80%）

- [ ] **Step 3: Commit**

```bash
git commit --allow-empty -m "test: coverage verified (≥80% core, ≥60% cli/doctor)"
```

### Task 8.3: README 完整版

**Files:**
- Modify: `README.md`

包含：
- 安装命令（含三平台分支）
- `minicpm-v doctor` 用法
- `image` / `video` 用法 + 输出 schema 示例
- Skill 安装 (`bash skill/install.sh`)
- **自动卸载提示**（spec §8.6 强制）
- 链接到 design doc
- Troubleshooting：4.6 GGUF / vLLM 显存 / sandbox 权限

- [ ] **Steps**：写 README → review → commit。

### Task 8.4: E2E 手动验收（必须 Mac 实测）

**Owner: 主 agent + 用户**

- [ ] **Step 1: `pip install -e .[dev,mlx]`**
- [ ] **Step 2: `minicpm-v doctor` → 拉 4bit 模型**
- [ ] **Step 3: `minicpm-v image tests/fixtures/sample.jpg` → JSON 正确**
- [ ] **Step 4: `minicpm-v video tests/fixtures/sample-5s.mp4` → JSON 含 frames + scenes**
- [ ] **Step 5: `--ttl 10` 验证 10s 后 ps aux 找不到 server**
- [ ] **Step 6: `minicpm-v status` 显示 alive=false**
- [ ] **Step 7: `bash skill/install.sh` → 检查 `~/.claude/skills/minicpm-v/SKILL.md` 存在**
- [ ] **Step 8: Claude Code 实际触发 SKILL.md（手测）**
- [ ] **Step 9: 把以上结果记到 `docs/e2e-mac-acceptance-2026-05-13.md`**

---

## Phase 9：项目清理

### Task 9.1: scripts/cleanup.sh

**Files:**
- Create: `scripts/cleanup.sh`

```bash
#!/usr/bin/env bash
# Remove build artifacts and caches; keep source + run-required files.
# Spec / Goal §4.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "Cleaning build artifacts..."
rm -rf build/ dist/ *.egg-info/ src/*.egg-info/
find . -type d -name __pycache__ -prune -exec rm -rf {} +
find . -type d -name .pytest_cache -prune -exec rm -rf {} +
find . -type d -name .mypy_cache -prune -exec rm -rf {} +
rm -rf .coverage htmlcov/

echo "Removing local venv (if present)..."
rm -rf .venv/ venv/

echo "Project size now:"
du -sh "$ROOT" 2>/dev/null || true

echo
echo "Preserved:"
echo "  - src/"
echo "  - tests/"
echo "  - docs/"
echo "  - skill/"
echo "  - scripts/"
echo "  - pyproject.toml, .gitignore, README.md, LICENSE"
echo
echo "NOT cleaned (user-owned caches; remove manually if desired):"
echo "  ~/.cache/minicpm-v-local/   (model weights, ~1-5 GB)"
echo "  ~/.run/minicpm-v-local/     (state, ~1 KB)"
```

- [ ] **Steps**：写脚本 → chmod +x → 跑一次（在交付前）→ verify 项目目录只剩源码 → commit。

```bash
chmod +x scripts/cleanup.sh
bash scripts/cleanup.sh
ls -la
git add scripts/cleanup.sh
git commit -m "chore: project cleanup script"
```

### Task 9.2: 最终交付检查

- [ ] **Step 1: 运行 cleanup**

```bash
bash scripts/cleanup.sh
```

- [ ] **Step 2: 重装 + 跑测试（验证 clean tree 仍能工作）**

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .[dev]
pytest tests/unit -v
```

- [ ] **Step 3: 把 `.venv` 清掉**

```bash
deactivate
rm -rf .venv
```

- [ ] **Step 4: `git log --oneline` 提供 commit 历史 → 交付**

- [ ] **Step 5: 把验收清单（17 节 success criteria）对照打勾，列在 `docs/v1-acceptance.md`**

---

## Self-Review

### Spec coverage

| Spec 节 | Plan Task |
|---|---|
| §1 背景 | (无 task，写在文档) |
| §2 范围 | (无 task) |
| §3 模型/后端事实 | (data brief 阶段，不写 task) |
| §4 架构 | Phase 0–7 整体 |
| §5 组件分解 | 每个 file 对应一个 task |
| §6 后端选择 | Task 2.1, 2.2 |
| §7 平台矩阵 | Task 2.2 (三 backend) + Task 8.4 (Mac E2E) |
| §8 Lifecycle | Task 3.1, 3.3, 3.4 |
| §9 隔离化 | Task 3.2 |
| §10 CLI | Task 7.1 |
| §11 Skill | Task 7.2 |
| §12 Flow | Task 6.1 (Flow 1), Task 3.3 (Flow 2/3), Task 5.2 (Flow 4) |
| §13 错误处理 | 分散在各 task 的 except |
| §14 配置 | Task 1.1, 1.2 |
| §15 测试策略 | 每个 task 都含测试 + Task 8.1–8.2 |
| §16 v2 候选 | (out of scope) |
| §17 success criteria | Task 8.4 |

无 gap。

### Placeholder scan

- ❌ "Add appropriate error handling" → 无（错误处理写在具体 except）
- ❌ "Similar to Task N" → 无（每个 task 自包含）
- ✅ 每个步骤都有完整代码/命令
- ⚠️ Task 3.3 manager.py 的测试只给了"测试要点"摘要而不是完整代码 → 这是因为 Mock subprocess.Popen 的样板代码较多，dev agent 按 spec §13 的边界要求和上面给出的实现签名补完。视为"详尽接口契约 + dev agent 自由度"，不视为 placeholder。
- ⚠️ Task 5.1 / 5.2 测试也类似 → 同上

### Type consistency

- `State` 字段在 state.py 定义、manager.py 使用、watchdog.py 读 — 字段名、类型一致 ✅
- `Backend` 抽象方法 `launch_cmd / artifact_id / install_check / health_path` 在 abstract + 三个具体子类 + manager.py 调用方 — 一致 ✅
- `Config` 字段 → cli.py overrides → doctor 写回 — 一致 ✅
- `ttl_seconds` vs `idle_timeout`：config 字段叫 `idle_timeout`；state 字段叫 `ttl_seconds`；CLI flag 叫 `--ttl`。三者语义同一（"保活秒数"），名字不同但每一处都明确指出来源/语义。可接受，但 dev agent 必须保持映射清晰。

### Scope check

单一 plan 范围合适。v2 候选已显式 out-of-scope。

---

## Execution Handoff

**Plan complete and saved to** `docs/plans/2026-05-13-minicpm-v-local-implementation.md`.

按用户 Goal 指定的协作流，**Subagent-Driven** 模式：

```
Task N:
  1. 主 agent dispatch ResearchAgent (Explore) → 读 spec 对应 section + 上游 docs/源码
     → 返回 200-400 字 brief
  2. 主 agent 把 brief 转发给 DevAgent (general-purpose) → 写代码 + 测试 → 自测通过
  3. 并行启动 ReviewAgent (general-purpose) → 跑测试 + 代码审 + 单一职责检查
  4. ReviewAgent OK → 销毁 DevAgent → 进入下一 Task
     ReviewAgent NOT OK → 回 DevAgent 改 → 重审 → loop
  5. git commit
```

主 agent 不读 spec 全文 / 不读上游源码 — 只读子 agent 返回的 brief、自己的 plan、用户消息。
