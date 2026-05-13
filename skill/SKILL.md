---
name: minicpm-v
description: |
  Local visual preprocessing using MiniCPM-V 4.6 (1.3B). Captions images and
  video timelines locally without sending pixels to the main model.
  Trigger when the user asks to analyze, describe, summarize, or extract
  information from images or videos and a local fast pass would save tokens.
---

# minicpm-v skill

## Prerequisites (first use)

Before this skill works, the user must run **once** in their terminal:

```
minicpm-v doctor
```

`doctor` detects the platform (mlx / cuda / cpu), checks `ffmpeg` / `ffprobe`,
downloads the MiniCPM-V 4.6 weights (~1 GB) to `~/.cache/minicpm-v-local/`, and
writes `~/.config/minicpm-v-local/config.toml`. Without this step, calls below
will fail with a "config missing" error.

If you (the main model) call this skill and it errors with exit code 2,
tell the user: "Please run `minicpm-v doctor` in your terminal first to set up
the local model." Override autodetect with `minicpm-v doctor --backend mlx|cuda|cpu`.

## When to use

- 用户给了图片或视频，让你描述、总结、抽信息
- 你需要"看一眼"图但不想把像素发给主模型
- 视频处理（默认主线）

## When NOT to use

- 用户要你"用自己的视觉能力"
- 需要图表里的精确数值（OCR 精度不一定够）
- 需要主模型的世界知识（地标识别、人物识别等）

## How to call

Two equivalent forms — pick whichever your runtime allows:

**Direct CLI (post `pip install`)**:
```
minicpm-v image <path> [--ttl <sec>] [--prompt "..."]
minicpm-v video <path> [--ttl <sec>] [--prompt "..."]
```

**Via Skill bundle (relative to this SKILL.md's directory)**:
```
bash scripts/run.sh image <path> [--ttl <sec>]
bash scripts/run.sh video <path> [--ttl <sec>]
```

Both produce identical JSON on stdout.

## Prompt presets (use the right preset for each scenario)

Pass `--prompt-preset <name>` for built-in scenarios:

| Preset | When to use | Output focus |
|---|---|---|
| (default) | General image description, Chinese-friendly | 整体场景 + 文字内容 + 物体 + UI 元素 |
| `ui` | App screenshots, dashboards, settings panels | Buttons, menus, inputs, labels, layout hierarchy |
| `photo` | Real-world photos, landscapes, portraits | Scene, subject, mood, lighting, composition |
| `doc` | Document scans, papers, slides, tables | Verbatim text + structure (headings, lists, tables) |
| `chart` | Bar / line / pie / scatter plots | Chart type, axes, data series, key values, trend |

Example:
```
bash scripts/run.sh image dashboard.png --prompt-preset ui
bash scripts/run.sh image report.png --prompt-preset doc
```

Or pass `--prompt "<your custom prompt>"` for full control (mutually exclusive with `--prompt-preset`).

## Output schema

### `image` returns

```json
{
  "version": 1,
  "input":  { "path": "...", "sha256": "..." },
  "model":  "mlx-community/MiniCPM-V-4.6-4bit",
  "result": {
    "caption":  "<one to several sentences>",
    "objects":  [],                  // v1: always empty, prompt-driven population in v2
    "ocr_text": null                 // v1: null placeholder; pass --prompt to elicit
  },
  "timing_ms": { "load": 0, "infer": <int> }
}
```

**How to use it**: read `result.caption` as the primary description. `objects` and `ocr_text` are non-null only when the user provides a `--prompt` asking for them.

### `video` returns

```json
{
  "version": 1,
  "input":  { "path": "...", "sha256": "...", "duration_s": <float>, "fps": <float> },
  "model":  "...",
  "frames": [
    { "t": <float seconds>, "path": "/tmp/.../scene_NNNN.jpg",
      "caption": "<frame caption>", "error": null }
  ],
  "scenes": [
    { "start": <float>, "end": <float>, "summary": "<scene caption>",
      "frame_indices": [<int>, ...] }
  ],
  "timing_ms": { "ffmpeg": <int>, "infer_total": <int> }
}
```

**How to use it**:
- `scenes[]` is the high-level timeline — use scene summaries for narrative
- `frames[]` has per-frame captions with timestamps — use for precise time queries
- A `frame.error` non-null means that one frame's inference failed; the overall output still ships

## Lifecycle / TTL

The local server auto-unloads after **5 minutes** of inactivity. Tune per call:

| Flag | When to use |
|---|---|
| `--ttl 600` | You expect ≥1 more image/video query within ~10 minutes (e.g. user is mid-conversation about a document) |
| `--ttl 0` | This is the last visual call you'll make in this session |
| `--keep` | Long-running multi-turn task; pin the server until manual `minicpm-v stop` |
| `--max-lifetime 1800` | Cap total server uptime at 30 min regardless of TTL renewals |

If you don't pass any flag, the default `idle_timeout` (300 s) applies.

## Lifecycle commands (debugging)

```
minicpm-v status         # show pid / port / expire_at as JSON
minicpm-v stop           # graceful SIGTERM, then SIGKILL after 5s
minicpm-v stop --force   # same as above (force flag reserved for future use)
```

Use these if you suspect a stale server, or before reporting environment issues to the user.

## Troubleshooting (errors you may surface)

| Symptom | Likely cause | Action |
|---|---|---|
| Exit code 2 / "config missing" | `doctor` not yet run | Ask user to run `minicpm-v doctor` |
| Exit code 3 / "server health check failed" | Backend deps missing or port range exhausted | Ask user to re-run `minicpm-v doctor`, or `rm ~/.run/minicpm-v-local/state.json` and retry |
| Exit code 4 / HTTP error during inference | Server crashed mid-call | Retry once; if still failing, `minicpm-v stop --force` then retry |
| Exit code 5 / "non-image" / "non-video" | Bad input path or unsupported format | Verify the file with the user |
| Exit code 10 / download failure | Network issue during model fetch | Ask user to re-run `minicpm-v doctor` after fixing network |
| `minicpm-v: command not found` | CLI not on PATH | Ask user to add `~/.local/bin` (Linux/Mac) or `%APPDATA%\Python\Scripts` (Windows) to PATH |
| Cold-start latency (3–15 s on first call) | Model loading into VRAM/RAM | Normal; subsequent calls are sub-second |

## Notes

- `minicpm-v` is the CLI. The Python distribution name is `minicpm-v-local` (`import minicpm_v_local`).
- Skill bundle is installed by `bash skill/install.sh` from the project — it places this file plus `scripts/run.sh` into `~/.claude/skills/minicpm-v/` and `~/.deepseek/skills/minicpm-v/` (when the latter exists).
