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
writes `~/.config/minicpm-v-local/config.toml`. Without this step, calls to
this skill will fail with a "config missing" error.

If you (the main model) call this skill and it errors with exit code 2,
tell the user: "Please run `minicpm-v doctor` in your terminal first to set up
the local model."

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
