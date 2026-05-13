# MiniCPM-V Local

Local visual preprocessing using **MiniCPM-V 4.6** (1.3B). Captions images and
video timelines locally so the main model never needs to ingest raw pixels.

Status: **v1 (in development)** — see `docs/specs/2026-05-13-minicpm-v-local-design.md`
for the full design and `docs/plans/2026-05-13-minicpm-v-local-implementation.md`
for the implementation roadmap.

## Platforms

| Platform | Backend | Server |
|---|---|---|
| Apple Silicon Mac | MLX | `mlx_vlm.server` |
| Linux + NVIDIA GPU | CUDA | `vllm serve` |
| Linux / Windows CPU | CPU | `llama-server` (llama.cpp) |

All backends expose an OpenAI-compatible HTTP API; we are a thin client + lifecycle manager.

## Quick start (preview)

```bash
pip install -e .[dev,mlx]   # or .[cuda] / .[cpu]
minicpm-v doctor             # first-time setup: deps + model download + config
minicpm-v image path/to/foo.jpg
minicpm-v video path/to/clip.mp4
```

## Claude Code Skill

```bash
bash skill/install.sh        # copies SKILL.md + run.sh to ~/.claude/skills/minicpm-v/
```

Then in Claude Code, the `minicpm-v` skill becomes available for image/video
preprocessing.

## ⚠️ Auto-unload

The local model server **auto-unloads from memory/VRAM after 5 minutes of
inactivity** to avoid permanently occupying resources. The next call reloads
automatically (cold start ≈ 3–15s).

Tune via `idle_timeout` in `~/.config/minicpm-v-local/config.toml`, env
`MINICPM_IDLE_TIMEOUT`, or per-call `--ttl <seconds>` (Claude passes this).

| Flag | Meaning |
|---|---|
| `--ttl 600` | Keep alive 10 min after this call (Claude says "still using it") |
| `--ttl 0` | Destroy immediately after this call |
| `--keep` | Pin alive (debug) |
| `--max-lifetime 1800` | Hard ceiling regardless of TTL renewals |

## License

Apache-2.0
