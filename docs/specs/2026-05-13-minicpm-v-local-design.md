# MiniCPM-V Local Skill — 设计文档

| 字段 | 值 |
|---|---|
| 日期 | 2026-05-13 |
| 状态 | Draft（待用户审批） |
| 目标版本 | v1 |
| 工作目录 | `/Users/lvbaiqing/TRUE 开发/MiniCPM-V-skills/` |
| 关联 | 用户原始三条需求（跨平台后端 / 本地预处理 / 视频为核心） |

---

## 1. 背景与目标

### 1.1 用户原始需求

1. **跨平台后端适配**
   - (a) 不同平台采用不同的后端运行
   - (b) 封装为一个技能，实现全自动下载与使用
2. **本地预处理与视觉处理**
   - (a) 模型对视频和图片进行提前预处理
   - (b) 简单图片场景下本地直接识别，结果发给主模型
   - (c) 做一个本地独立的视觉图像与语音处理模型
3. **主要处理方向**：以视频处理为核心

补充需求（已在 brainstorming 中确认）：

4. **按需下载与初始化**：首次使用时校验环境并补齐配置，后续直调
5. **服务销毁策略**：任务完成后默认 5 分钟内销毁；可由调用方（Claude）自定义；最大销毁时间作硬上限
6. **隔离化作为安全选项**：默认关闭，doctor 询问开启，进程销毁失败时主动提示

### 1.2 设计目标（按优先级）

- **P0**：在 Apple Silicon / Linux+CUDA / Linux-Windows CPU 三类平台上，通过统一 CLI 接口把 MiniCPM-V 4.6 (1.3B) 跑起来；Claude 通过 Skill 调用，获取结构化 JSON。
- **P0**：模型自动下载 + 首次自检；后续调用 ≤ 1s warm path（不重复加载权重）。
- **P0**：闲置 5 分钟自动卸载，避免占内存/显存；Claude 可主动控制 TTL。
- **P1**：视频处理是核心 — 关键帧抽取 + 每帧 caption + 时间轴聚合。
- **P2**：隔离化沙箱（备用）— 进程销毁失败时的兜底。

### 1.3 非目标（v1 out of scope）

- ❌ ASR / 语音转写（用户确认放 v2，预留接口）
- ❌ 移动端（iOS / Android / HarmonyOS）—— 官方有边缘代码可未来引入
- ❌ Intel Mac（不在用户平台列表）
- ❌ Docker / Podman 容器化（用户明令"太重"）
- ❌ Web UI / GUI（CLI-only）
- ❌ 模型微调 / LoRA

---

## 2. 范围划分（v1 in-scope）

| 子系统 | 范围 |
|---|---|
| Skill 壳 | `~/.claude/skills/minicpm-v/` 一份 SKILL.md + 极薄 shell entrypoint |
| Python 库 | `minicpm_v_local`，封装 backend / lifecycle / pipeline |
| CLI | `minicpm-v {doctor, image, video, status, stop}` |
| Backend | mlx-vlm (macOS) / vLLM (Linux+CUDA) / llama-server (CPU) — 全部为开源 server，OpenAI 兼容 |
| 模型下载 | `huggingface_hub.snapshot_download`，按 backend 拉对应 artifact |
| 视频 pipeline | ffmpeg 抽关键帧 + 串行 caption + 时间轴聚合 |
| Lifecycle | sidecar watchdog，state.json 驱动，SIGTERM → SIGKILL → 检测残留 |
| 隔离化 | macOS sandbox-exec / Linux bubblewrap，默认 off |

---

## 3. 模型与后端核对（事实基线）

> 本节信息由 brainstorming 阶段实际查询确认（2026-05-13）。

### 3.1 MiniCPM-V 4.6

| 项 | 值 |
|---|---|
| 模型 ID | `openbmb/MiniCPM-V-4.6`（另有 `MiniCPM-V-4.6-Thinking`） |
| 参数 | 1.3B（400M SigLIP2 视觉 + 800M Qwen3.5 语言） |
| 发布 | 2026-05-11 |
| 输入 | 文本 + 图像 + 视频，上下文 262K |
| License | Apache 2.0 |
| 官方后端 | Transformers / llama.cpp(GGUF) / vLLM / SGLang / Ollama |

### 3.2 MLX 量化版（mlx-community 维护，2026-05-12 更新）

- `mlx-community/MiniCPM-V-4.6-4bit`
- `mlx-community/MiniCPM-V-4.6-5bit`
- `mlx-community/MiniCPM-V-4.6-8bit`
- `mlx-community/MiniCPM-V-4.6-bf16`
- `mlx-community/MiniCPM-V-4.6-mxfp4`
- `mlx-community/MiniCPM-V-4.6-mxfp8`
- `mlx-community/MiniCPM-V-4.6-nvfp4`

### 3.3 后端服务化方案

- **mlx-vlm** (`pip install -U mlx-vlm`) 原生支持 MiniCPM-V 4.6，自带 FastAPI OpenAI 兼容 server (`mlx_vlm.server --model … --port …`)
- **vLLM** (`vllm serve openbmb/MiniCPM-V-4.6 --port …`) 官方 README 列出支持
- **llama.cpp** (`llama-server -m model.gguf --port …`) GGUF 支持需上游确认（4.6 刚出，**v1 doctor 中检测；若未就绪给清晰错误，引导用户回退 4.5 GGUF 或等待**）

---

## 4. 整体架构

```
┌─ Claude Code (主模型) ────────────────────────────────┐
│ 1. 触发 Skill (image / video / doctor)               │
└────────────────┬────────────────────────────────────┘
                 │
                 ▼
┌─ Layer 1: Claude Code Skill ────────────────────────┐
│ ~/.claude/skills/minicpm-v/                          │
│   SKILL.md          触发词、用法、Claude 怎么解读 JSON │
│   scripts/run.sh    → python -m minicpm_v_local …   │
└────────────────┬────────────────────────────────────┘
                 │ subprocess
                 ▼
┌─ Layer 2: Python CLI / Orchestrator ────────────────┐
│ minicpm-v doctor       首次校验 + 补齐配置            │
│ minicpm-v image <p>    单图 → JSON                   │
│ minicpm-v video <p>    视频 → 时间轴 JSON             │
│ minicpm-v status       打印 state.json               │
│ minicpm-v stop         手动销毁                      │
└────────┬─────────────────────┬───────────────────────┘
         │ HTTP localhost      │ spawn/kill
         ▼                     ▼
┌─ Backend Server (开源) ┐   ┌─ Idle Watchdog ────────┐
│ Mac:  mlx_vlm.server  │   │ sidecar 进程            │
│ CUDA: vllm serve      │   │ 闲置 N 秒 → SIGTERM    │
│ CPU:  llama-server    │   │ N 默认 300s            │
└────────────────────────┘   └────────────────────────┘
```

### 4.1 为什么三层

- **Skill 壳**：让 Claude 知道"这里有个工具"，且 Skill 是 Claude Code 原生的可发现机制。
- **Python 库 + CLI**：业务逻辑承载层；脱离 Claude 也能跑（pytest / 手测）。
- **后端 server**：完全使用开源现成方案，我们不写 GPU/Metal kernel；切换后端 = 换 binary 启动命令。

### 4.2 为什么所有平台都走 HTTP（OpenAI 兼容）

| 候选 | 评估 | 结论 |
|---|---|---|
| 直接 Python import 各 backend | 三套不同 API，抽象层难写 | ❌ |
| 统一 HTTP 客户端 | 三个 backend 都原生提供 OpenAI 兼容 server | ✅ |
| 自研 gRPC | 工作量大，无收益 | ❌ |

HTTP 走 loopback，开销 < 1ms，相对单帧推理 100ms~3s 可以忽略。

---

## 5. 组件分解

每个文件 < 800 行，按职责拆分。

| 模块 | 职责 | 行数预算 |
|---|---|---|
| `runtime/detect.py` | OS/CPU/GPU 探测 → `BackendTag` (`mlx`/`cuda`/`cpu`) | < 100 |
| `runtime/backend.py` | `Backend` 抽象类：`launch_cmd()` / `health_url()` / `artifact_id()` / `install_check()` | < 150 |
| `runtime/mlx.py` | `MLXBackend` — 调 `mlx_vlm.server` | < 80 |
| `runtime/cuda.py` | `CUDABackend` — 调 `vllm serve` | < 80 |
| `runtime/cpu.py` | `CPUBackend` — 调 `llama-server` | < 80 |
| `server/manager.py` | spawn / health / SIGTERM；state.json 原子读写 | < 250 |
| `server/watchdog.py` | sidecar 进程；定时检查 state.expire_at | < 100 |
| `server/isolation.py` | 沙箱启动器：`sandbox-exec` / `bwrap` / `noop` | < 120 |
| `download.py` | `snapshot_download` + lockfile + 进度 + 校验 | < 200 |
| `client.py` | OpenAI HTTP 客户端（chat.completions + image_url） | < 150 |
| `pipeline/image.py` | 单图：encode → call → JSON | < 100 |
| `pipeline/video.py` | ffprobe → 抽帧 → 串行 caption → 聚合 | < 400 |
| `pipeline/router.py` | （v2 占位）简单/复杂图片分流；v1 默认全本地 | < 50 |
| `doctor.py` | 8 步自检 + 引导 | < 300 |
| `config.py` | 优先级链 CLI > env > toml > default | < 100 |
| `cli.py` | argparse 入口 | < 200 |
| `paths.py` | 集中所有路径常量（cache / state / config / log） | < 50 |

**Skill 壳**（Layer 1）：

| 文件 | 内容 |
|---|---|
| `~/.claude/skills/minicpm-v/SKILL.md` | YAML frontmatter + 触发场景 + 用法 + 输出 schema + 自动卸载告知 |
| `~/.claude/skills/minicpm-v/scripts/run.sh` | 极薄：`exec python -m minicpm_v_local "$@"` |

---

## 6. 后端选择决策

### 6.1 平台 → backend 映射

| 平台探测结果 | BackendTag | Server | 模型 artifact |
|---|---|---|---|
| `Darwin` + `arm64` | `mlx` | `mlx_vlm.server` | `mlx-community/MiniCPM-V-4.6-{4bit\|8bit\|bf16}` |
| `Linux` + NVIDIA GPU (CUDA ≥ 12) | `cuda` | `vllm serve` | `openbmb/MiniCPM-V-4.6`（原始 safetensors） |
| 其他 / `--force-cpu` | `cpu` | `llama-server` | GGUF（4.6 GGUF 待上游；fallback 错误提示） |

### 6.2 探测顺序

```
detect():
  if darwin and arm64: return "mlx"
  if linux and nvidia-smi succeeds and CUDA ≥ 12: return "cuda"
  return "cpu"
```

用户可通过 `--backend mlx|cuda|cpu` 或 env `MINICPM_BACKEND` 强制覆盖。

### 6.3 量化档位选择（Mac）

doctor 阶段询问，默认 `4bit`（最小 ~900MB，速度最佳）；专业用户可选 `bf16` (~2.6GB) 拿最高质量。

---

## 7. 平台适配矩阵

| 维度 | macOS arm64 | Linux + CUDA | Linux/Win CPU |
|---|---|---|---|
| 推理 backend | mlx-vlm | vLLM | llama.cpp |
| Server binary | `mlx_vlm.server` | `vllm serve` | `llama-server` |
| 安装方式 | `pip install mlx-vlm` | `pip install vllm` | 系统包或 release tarball |
| 模型大小（默认） | 4bit ≈ 0.9 GB | bf16 ≈ 2.6 GB | Q4_K_M ≈ 0.9 GB |
| 推理速度（单图 caption 预期） | 200–500ms | 100–300ms | 1–3s |
| 启动时间（cold） | 3–8s | 10–30s | 5–15s |
| 隔离化方案 | `sandbox-exec` | `bubblewrap` (bwrap) | Windows: 无；Linux: bwrap |
| 已知风险 | mlx-vlm 对 4.6 支持已 verify (2026-05-13)；量化档位影响 OCR 质量 | vLLM 显存占用大，需 ≥ 8GB VRAM | 4.6 GGUF 上游支持需 doctor 检测；可能需要 fallback |

---

## 8. Lifecycle 与销毁策略

### 8.1 销毁主控（用户已确认方案 A）

> **Claude 主控 + server 默认 idle timer 兑底。**

- Claude 通过 CLI flag 表达意图（继续用 / 立即结束）
- watchdog 读 state.json，到期自动 kill
- 如果 Claude 没传任何 hint，fallback 到默认 `idle_timeout = 300s`

### 8.2 销毁意图表达（CLI flag）

| Claude 意图 | CLI 形态 | `state.expire_at` 行为 |
|---|---|---|
| 继续用，估计还要 N 秒 | `--ttl <N>` | `now + N` |
| 这次是最后一次 | `--ttl 0` | 调用结束立即销毁 |
| 永久保活（调试） | `--keep` | 不写 expire_at，watchdog 跳过 |
| 不传任何 flag | (无) | `now + idle_default` (默认 300s) |
| 任意调用 | `--max-lifetime <N>` | server 启动时设硬上限，无论后续 ttl 怎么续，都不能穿越；默认 30 min |

**`--ttl 0` 与 `--done` 合并**：v1 不引入 `--done`，统一用 `--ttl 0` 表达"用完即销"。理由：减少 CLI surface，语义同源。

### 8.3 销毁机制（kill 进程，不做 unload）

| 步骤 | 行为 | 超时 |
|---|---|---|
| 1 | `kill -TERM <server.pid>`（让进程优雅退出） | 等 5s |
| 2 | 若还活着，`kill -KILL <server.pid>` | 立即 |
| 3 | 抽样检测 GPU 显存 / RSS 是否回收 | 2s 内 |
| 4 | 若未回收 → 写 `cleanup_failed = true` | — |
| 5 | 清空 state.json（`alive = false`） | — |

`cleanup_failed = true` 会在下一次 `doctor` 或 `status` 时显示，并建议启用隔离化。

### 8.4 视频任务期间不会被杀

每帧调用都更新 `last_used_at` 和 `expire_at`，timer 自然续期。**如果**整段视频预估超过 `max-lifetime`，Claude 应该用 `--max-lifetime <N>` 主动扩大上限；否则到达 max-lifetime 时 server 会被强杀，剩余帧失败（标记 frame.error）。

### 8.5 state.json schema

路径：`~/.run/minicpm-v-local/state.json`（运行时状态，与 `~/.config/` 的持久配置分离）。

```json
{
  "schema_version": 1,
  "backend": "mlx",
  "model_repo": "mlx-community/MiniCPM-V-4.6-4bit",
  "server": {
    "pid": 41203,
    "port": 8765,
    "started_at": "2026-05-13T14:02:11Z",
    "health_url": "http://127.0.0.1:8765/health",
    "isolation_mode": null
  },
  "watchdog": {
    "pid": 41204
  },
  "lifecycle": {
    "last_used_at": "2026-05-13T14:08:47Z",
    "expire_at":    "2026-05-13T14:13:47Z",
    "ttl_seconds":  300,
    "max_lifetime_at": "2026-05-13T14:32:11Z",
    "keep":         false
  },
  "cleanup_failed": false,
  "alive": true
}
```

写入用 `os.replace()` 做原子替换。watchdog 读到半截 JSON 时 → ignore + retry。

### 8.6 必须在用户文档中明示

`README.md` 和 `SKILL.md` 都需要包含以下提示（用户需求 5(b) 明确要求）：

> ⚠️ **自动卸载提示**：本地模型 server 会在 **5 分钟无请求**后自动从内存/显存卸载，以避免长期占用资源。下次调用会自动重新加载（cold start ≈ 3–15s）。配置项：`idle_timeout`（秒），可通过 `config.toml`、env `MINICPM_IDLE_TIMEOUT`、CLI `--ttl` 控制。

---

## 9. 隔离化沙箱（备用安全选项）

### 9.1 启用时机

| 场景 | 行为 |
|---|---|
| Doctor 初次自检 | 询问用户是否启用，默认否，写入 config.toml |
| `cleanup_failed = true` | `doctor` / `status` 主动提示"建议启用隔离化" |
| 用户显式 `--isolated` | 当次强制启用 |

不主动每次启用 / 不静默启用。

### 9.2 平台映射

| 平台 | 沙箱机制 | 命令样例 | v1 状态 |
|---|---|---|---|
| macOS | `sandbox-exec`（系统自带） | `sandbox-exec -f profile.sb python -m mlx_vlm.server …` | ✅ |
| Linux | `bubblewrap` (`bwrap`) | `bwrap --unshare-all --share-net --bind / / python -m vllm.entrypoints.openai.api_server …` | ✅ |
| Windows | 无原生轻量方案 | — | ❌ doctor 显式告知不支持 |

**显式不采用**：Docker / Podman（用户禁止"太重"）、firejail（依赖 SUID，新发行版限制多）、chroot（不够强且不易访问 GPU）。

### 9.3 沙箱 profile 要点

- 允许：模型 cache 目录读、临时帧目录读写、loopback 网络
- 禁止：除 cache 目录外的家目录读写、外网（除非显式开 HF 下载窗口）
- GPU 设备：Linux 必须 `--dev-bind /dev/nvidia* /dev/nvidia*`；Mac Metal 在 sandbox-exec 下需要额外 entitlement——v1 提供模板 profile，复杂场景由用户调整。

---

## 10. CLI 接口规范

### 10.1 子命令一览

```
minicpm-v doctor [--reset]            首次自检 / 重置配置
minicpm-v image <path> [opts]         单图 caption / 描述
minicpm-v video <path> [opts]         视频时间轴 + 关键帧 caption
minicpm-v status                       打印 state.json (人类可读)
minicpm-v stop [--force]               主动销毁 server
```

### 10.2 共用 opts（所有推理命令）

| flag | 含义 | 默认 |
|---|---|---|
| `--backend {mlx,cuda,cpu}` | 强制 backend | auto-detect |
| `--quant {4bit,8bit,bf16,…}` | 强制量化档位（Mac） | config.toml |
| `--ttl <sec>` | 本次调用结束后保活时间；`0` = 立即销毁 | `idle_timeout` |
| `--max-lifetime <sec>` | server 总生命硬上限 | 1800 |
| `--keep` | 永久保活（调试） | false |
| `--isolated` | 启用沙箱 | config.toml |
| `--output {json,jsonl}` | 输出格式 | json |
| `--prompt <text>` | 覆盖默认 prompt | 内置 |

### 10.3 `image` 输出 schema

```json
{
  "version": 1,
  "input": { "path": "foo.jpg", "sha256": "…" },
  "model": "mlx-community/MiniCPM-V-4.6-4bit",
  "backend": "mlx",
  "result": {
    "caption": "A red bicycle leaning against a brick wall …",
    "objects": ["bicycle", "brick wall"],
    "ocr_text": null
  },
  "timing_ms": { "load": 0, "infer": 312 }
}
```

`ocr_text` 在 v1 是 null 占位；prompt 可被覆盖触发 OCR，但默认 pipeline 不做。

### 10.4 `video` 输出 schema

```json
{
  "version": 1,
  "input": { "path": "v.mp4", "sha256": "…", "duration_s": 142.3, "fps": 30 },
  "model": "mlx-community/MiniCPM-V-4.6-4bit",
  "backend": "mlx",
  "frames": [
    { "t": 3.142, "path": ".../frame_001.jpg", "caption": "…", "error": null }
  ],
  "scenes": [
    { "start": 0.0, "end": 12.4, "summary": "…", "frame_indices": [0,1,2] }
  ],
  "timing_ms": { "ffmpeg": 1200, "infer_total": 8400 }
}
```

`frame.error` 非 null 时表示这一帧推理失败，不阻断整体输出。

### 10.5 退出码

| Code | 含义 |
|---|---|
| 0 | 成功 |
| 1 | 通用错误 |
| 2 | 配置缺失 / doctor 未跑过 |
| 3 | server 启动失败 |
| 4 | 推理失败 |
| 5 | 输入文件非法 |
| 10 | 模型下载失败 |

---

## 11. Skill 接口规范

### 11.1 SKILL.md 结构

```yaml
---
name: minicpm-v
description: |
  Local visual preprocessing using MiniCPM-V 4.6 (1.3B). Captions images and
  video timelines locally without sending pixels to the main model.
  Trigger when the user asks to analyze, describe, summarize, or extract
  information from images or videos and a local fast pass would save tokens.
---
```

正文要点：

1. **何时触发**：用户上传图/视频且非"高难度推理"型问题
2. **不要触发**：用户明确要求"用你的视觉能力"、需要图表精确数值、需要主模型的世界知识
3. **怎么调**：
   - `bash scripts/run.sh image <path>` → 读 stdout JSON
   - `bash scripts/run.sh video <path>` → 读 stdout JSON
4. **TTL 用法**：
   - 还要继续问图：传 `--ttl 600`
   - 这是最后一次：传 `--ttl 0`
5. **输出怎么用**：把 `result.caption` 当作图像内容传递给后续推理；视频用 `scenes[]` 做时间轴定位
6. **自动卸载告知**（用户需求 5(b)）：服务闲置 5 分钟会自动卸载，下次调用会重新加载

### 11.2 scripts/run.sh

```bash
#!/usr/bin/env bash
# Thin entrypoint. Real logic lives in the Python package.
set -euo pipefail
exec python -m minicpm_v_local "$@"
```

---

## 12. 关键 Flow

### 12.1 首次使用（cold start，触发 doctor 自检）

```
Claude → Skill image foo.jpg
   ▼
run.sh → python -m minicpm_v_local image foo.jpg
   ▼
cli.py 读 ~/.config/minicpm-v-local/config.toml
   │   不存在 → 自动转 doctor
   ▼
doctor.py 8 步:
   1. OS/CPU/GPU 探测 → backend tag
   2. 检查 Python 依赖（mlx-vlm / vllm / llama-cpp-python）
      缺则 pip install（询问用户确认）
   3. 检查 ffmpeg/ffprobe；缺则提示安装命令（不自动装系统包）
   4. 选量化档位（Mac: 询问；CUDA/CPU 各有默认）
   5. 下载模型 weights → ~/.cache/minicpm-v-local/<backend>/<repo>/
   6. 询问"是否启用隔离化沙箱"（默认否）→ 写 config
   7. 询问"默认 idle_timeout"（默认 300s）→ 写 config
   8. 试启动 server + /health → 跑一张测试图 → 写 config.toml
   ▼
回到 image foo.jpg → 走 12.2
```

### 12.2 常态调用（warm path）

```
cli image foo.jpg --ttl 600
   ▼
server/manager.py 读 state.json
   ├─ alive && health OK   → warm，直调
   └─ 否则                  → cold:
        spawn backend server (Popen, setsid)
        spawn watchdog (Popen, setsid, detached)
        轮询 /health 最多 60s
        写 state.json
   ▼
client.py POST /v1/chat/completions (image base64)
   ▼
更新 state.json: last_used_at = now, expire_at = now + ttl
   ▼
print JSON → stdout
```

### 12.3 闲置销毁

```
watchdog (sidecar, setsid)
   每 10s loop:
     读 state.json
     if now >= expire_at:
        kill -TERM <server.pid>
        sleep 5
        if 还活着: kill -KILL <server.pid>
        if GPU/RSS 未回收 (2s 内抽样):
           写 state: cleanup_failed=true
        清空 state.json (alive=false)
        exit
```

### 12.4 视频处理 pipeline

```
input: video.mp4
   ▼
ffprobe → duration, fps, resolution
   ▼
ffmpeg -vf "select='gt(scene,0.3)',showinfo" -vsync vfr
   + 兜底均匀采样（每 10s 至少 1 帧）
   → frames/00:00:03.142.jpg, …
   ▼
ensure server warm (12.2 cold path)
   ▼
for each frame:
   client.chat_completion(image=frame, prompt="Describe what's happening")
   更新 state (续期)
   ▼
后处理：相邻帧 caption 余弦相似度 > 阈值 → 合并成 scene
   ▼
输出 JSON (12.4 / 10.4)
```

抽帧配置项（写进 config.toml）：

| 项 | 默认 | 含义 |
|---|---|---|
| `video.scene_threshold` | 0.3 | ffmpeg scene detect 阈值 |
| `video.fallback_interval` | 10.0 | 兜底均匀采样间隔（秒） |
| `video.max_frames` | 60 | 单视频最多抽多少帧（防爆） |
| `video.scene_merge_similarity` | 0.85 | scene 合并的余弦阈值 |

---

## 13. 错误处理与边界

按层处理，不在内部代码塞防御性 try/except。

| 边界 | 处理 |
|---|---|
| doctor 任一步失败 | 输出"哪一步、什么原因、建议命令"，exit 2 |
| server spawn 失败 | 3 次重试（端口冲突时换端口），失败建议跑 doctor，exit 3 |
| Health check 60s 不过 | kill 残留 + 报错，exit 3 |
| HTTP 推理错误 | 单图：1 次重试 → 失败 exit 4；视频：单帧失败标记 `frame.error`，不阻断整体 |
| 模型下载断网 | `huggingface_hub` 自带断点续传；最终失败给清晰错误 + 手动下载步骤，exit 10 |
| 输入文件非法 | 入口校验 mime + ffprobe，明确报"非视频"/"非图片"，exit 5 |
| state.json 半截 / 损坏 | watchdog 忽略 + retry；manager 重建 |
| 多个 CLI 同时调 | 用 lockfile（fcntl）排队；后到者等 first server up |
| CLI 子进程被 SIGINT（如 Ctrl-C） | CLI 进程退出，**不杀 server**（server 是长寿的；下次调用还能 warm 接上） |
| 端口冲突 | 范围 8765–8775 内重试 |

**只在边界 validate**：CLI 入口 / HTTP 响应解析 / 文件读入。内部模块之间假设契约。

---

## 14. 配置与路径

### 14.1 优先级链

```
CLI flag  >  env var (MINICPM_*)  >  config.toml  >  built-in default
```

### 14.2 路径常量

| 用途 | 路径 |
|---|---|
| 持久配置 | `~/.config/minicpm-v-local/config.toml` |
| 运行时状态 | `~/.run/minicpm-v-local/state.json` |
| 模型缓存 | `~/.cache/minicpm-v-local/<backend>/<repo>/` |
| 日志 | `~/.local/state/minicpm-v-local/logs/` |
| 锁文件 | `~/.run/minicpm-v-local/cli.lock`, `…/download.lock` |
| 视频帧临时目录 | `$TMPDIR/minicpm-v-local/frames-<run_id>/`（任务结束清理） |

### 14.3 config.toml 示例

```toml
backend = "auto"          # auto | mlx | cuda | cpu
quant = "4bit"
idle_timeout = 300        # seconds
max_lifetime = 1800       # hard ceiling
isolation = false
isolation_mode = "auto"   # auto | sandbox-exec | bwrap | none

[server]
host = "127.0.0.1"
port_range = [8765, 8775]
health_timeout = 60

[video]
scene_threshold = 0.3
fallback_interval = 10.0
max_frames = 60
scene_merge_similarity = 0.85

[download]
source = "huggingface"    # v1 only; v2 may add mirrors
```

### 14.4 环境变量

| Env | 等价 config 项 |
|---|---|
| `MINICPM_BACKEND` | `backend` |
| `MINICPM_QUANT` | `quant` |
| `MINICPM_IDLE_TIMEOUT` | `idle_timeout` |
| `MINICPM_MAX_LIFETIME` | `max_lifetime` |
| `MINICPM_PORT` | `server.port_range[0]` |
| `MINICPM_LOG_LEVEL` | logging |

---

## 15. 测试策略

### 15.1 测试金字塔

| 层 | 工具 | 覆盖率目标 |
|---|---|---|
| Unit | pytest + monkeypatch | 核心模块 ≥ 80% (runtime / server / pipeline / download / config) |
| Integration | pytest，跑一个 tiny GGUF on CPU | 关键路径覆盖 |
| E2E | 手工 + 脚本 | Mac 实测 + idle 销毁实测 + Skill 触发实测 |

CLI 入口和 doctor 因有大量交互 IO，目标 60%（per testing.md，coverage 针对真实风险）。

### 15.2 关键测试用例

**Unit**

- `detect.py`：mock `platform.system/machine` + `nvidia-smi` 出口，覆盖 4 个组合
- `manager.py`：state.json 原子写、并发读、过期判定 ± 1s 边界
- `watchdog.py`：mock time + os.kill，验证 TERM→KILL 顺序、cleanup_failed 标记
- `download.py`：mock `hf_hub.snapshot_download`，验证 lockfile 防并发
- `pipeline/video.py`：mock ffprobe + ffmpeg，验证抽帧策略 + scene 聚合算法（阈值边界 0.84/0.85/0.86）
- `config.py`：优先级链 CLI > env > toml > default 四级覆盖

**Integration**

- 用 CPU backend + tiny GGUF 跑端到端 image / video
- doctor 全流程（除模型下载用 fixture）
- spawn 真 sleep server + watchdog，timer 触发，验证 PID 消失 + state cleared

**E2E（必须手测）**

- Mac：`minicpm-v doctor` → 拉 4bit → 跑 sample 图 → 跑 30s sample 视频
- 设 `--ttl 10`，等 server 自动消失（ps 验证）
- Claude Code 里实际触发 SKILL.md，验证 JSON 能被读懂
- `cleanup_failed` 路径：手动让 server hung（pause signal），验证 SIGKILL + 标记

### 15.3 不测的（避免无意义覆盖）

- 第三方 server 内部行为（mlx-vlm / vllm / llama-server）
- Windows 上的 bwrap 路径（不存在）
- mac Intel 路径（v1 不支持）

---

## 16. 开放项 / v2 候选

| 项 | 当前决定 | v2 候选 |
|---|---|---|
| ASR / 语音 | v1 不做 | 接 whisper.cpp + faster-whisper |
| 简单/复杂图片 router | v1 默认全本地 | 加 confidence + 启发式分流 |
| 镜像源切换 | v1 只 HF | 加 ModelScope / hf-mirror fallback |
| 移动端 | v1 不做 | 评估 OpenBMB 边缘代码移植 |
| 多模型并存 | v1 单模型 | 多 server 进程 + 路由 |
| MCP server 形态 | v1 仅 CLI + Skill | 视使用情况评估是否补 MCP |
| OCR pipeline | v1 prompt 触发 | 独立子命令 + 后处理（结构化表格） |

---

## 17. 验证目标（success criteria）

v1 视为完成的条件：

- [ ] Mac (Apple Silicon) 上 `minicpm-v doctor` 全程跑通，权重落到 `~/.cache/…`
- [ ] `minicpm-v image sample.jpg` 在 cold 状态下 ≤ 15s 出 JSON；warm 状态 ≤ 2s
- [ ] `minicpm-v video sample-30s.mp4` 出包含 ≥ 3 个 frame 和 ≥ 1 个 scene 的 JSON
- [ ] 不传 `--ttl`，5 分钟后 `ps aux | grep mlx_vlm` 看不到进程；state.json `alive=false`
- [ ] `--ttl 10` 调用结束后 10s 内 server 进程消失
- [ ] `--ttl 0` 调用结束立即销毁
- [ ] Claude Code 里 SKILL.md 能被自动触发，JSON stdout 能被 Claude 解析使用
- [ ] 手动 SIGSTOP 模拟 hung 后，watchdog 走完 TERM→KILL→cleanup_failed 全流程
- [ ] 测试覆盖：核心模块 ≥ 80%，CLI/doctor ≥ 60%
- [ ] README + SKILL.md 都写明"5 分钟自动卸载"

**v1 Definition of Done**：以上所有 ☐ 在 Mac 上全部 ✓。Linux + CUDA 与 CPU 平台的端到端推理验证在用户拥有相应硬件时补充；**v1 内只要求 doctor 能在这两个平台上跑通环境检查 + backend 代码路径有 unit/integration 覆盖**（即使模型未下载 / 未实跑）。

---

## 附录 A：已决议事项（brainstorming 阶段）

| # | 决议 | 来源 |
|---|---|---|
| 1 | 产物三层：Skill 壳 + Python 库底座 + CLI；MCP 留待 v2 评估 | 用户回答 Q1 |
| 2 | 目标平台：Mac arm64 + Linux+CUDA + Linux/Win CPU；不含 Intel Mac / 移动端 | 用户回答 Q2 |
| 3 | 结果回传：结构化 JSON 到 stdout，Claude 同对话读 | 用户回答 Q3 |
| 4 | 视频输出：时间轴事件 + 关键帧 caption（双轨） | 用户回答 Q4 |
| 5 | ASR：v1 不做，v2 接 whisper.cpp | 用户回答 |
| 6 | 下载源：HuggingFace 官方为主，无国内镜像（用户不在国内） | 用户回答（含纠正） |
| 7 | 销毁主控：Claude 主控（CLI flag）+ server 默认 idle timer 兑底 | 用户回答 |
| 8 | 销毁粒度：kill server 进程（不做 model unload） | 设计默认，未被 push back |
| 9 | Idle timer 起点：last_used_at = 最后一次请求完成时 | 设计默认 |
| 10 | 视频期间 timer 暂停（自然续期） | 设计默认 |
| 11 | `--done` 等价 `--ttl 0`，v1 只保留后者 | 设计默认 |
| 12 | 抽帧策略：scene-detect + 10s 兜底，参数全可配 | 设计默认 |
| 13 | state 路径：`~/.run/minicpm-v-local/state.json`（与持久 config 分离） | 设计默认 |
| 14 | 隔离化：默认 off，doctor 询问，cleanup_failed 提示，不用 Docker | 用户需求 + 设计 |

## 附录 B：参考资料

- OpenBMB/MiniCPM-V GitHub: https://github.com/OpenBMB/MiniCPM-V
- Model card: https://huggingface.co/openbmb/MiniCPM-V-4.6
- MLX 量化版: https://huggingface.co/mlx-community/MiniCPM-V-4.6-4bit
- mlx-vlm: https://github.com/Blaizzy/mlx-vlm
- vLLM docs: https://docs.vllm.ai
- llama.cpp: https://github.com/ggerganov/llama.cpp
- Artificial Analysis 评测: https://artificialanalysis.ai/models/minicpm-v4-6-1-3b
