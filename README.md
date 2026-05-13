# MiniCPM-V Local

Local visual preprocessing using **MiniCPM-V 4.6** (1.3B params).
Captions images and produces timeline-aware video summaries on your machine, so the main Claude model never receives raw pixels.

> ⚠️ **Auto-unload (default 5 min)**
> The local model server is killed from memory/VRAM after **5 minutes with no requests**.
> The next call automatically reloads it (cold start ≈ 3–15 s on Apple Silicon).
> Override with `--ttl <sec>`, `--keep`, or `--ttl 0` (immediate teardown).

## Platforms & backends

| Platform | Backend | Install extra |
|---|---|---|
| macOS (Apple Silicon, arm64) | `mlx-vlm` | `pip install -e .[mlx]` |
| Linux + NVIDIA GPU | `vLLM` | `pip install -e .[cuda]` |
| Linux / Windows CPU | `llama-server` (llama.cpp, system binary) | `pip install -e .[cpu]` |

## Installation

```bash
git clone <repo>
cd MiniCPM-V-skills
pip install -e .[mlx]   # or [cuda] / [cpu]
```

Then run the doctor to verify the environment and download the model:

```bash
minicpm-v doctor

# Or override platform autodetect / skip prompts (headless setups):
minicpm-v doctor --backend mlx --quant 4bit --non-interactive   # -y / --yes also accepted
```

`doctor` checks the platform, picks a backend, asks about isolation mode, writes `~/.config/minicpm-v-local/config.toml`, and fetches the appropriate weights into `~/.cache/minicpm-v-local/<backend>/`.

## Usage

### Single image

```bash
minicpm-v image path/to/photo.jpg
```

Output (stdout JSON):

```json
{
  "version": 1,
  "input": { "path": "path/to/photo.jpg", "sha256": "..." },
  "model": "mlx-community/MiniCPM-V-4.6-4bit",
  "result": {
    "caption": "A red square on a white background.",
    "objects": [],
    "ocr_text": null
  },
  "timing_ms": { "load": 0, "infer": 412 }
}
```

### Video

```bash
minicpm-v video path/to/clip.mp4
```

Output:

```json
{
  "version": 1,
  "input": { "path": "clip.mp4", "sha256": "...", "duration_s": 30.0, "fps": 30.0 },
  "model": "...",
  "frames": [
    { "t": 0.0, "path": "/tmp/.../scene_0001.jpg", "caption": "...", "error": null }
  ],
  "scenes": [
    { "start": 0.0, "end": 5.0, "summary": "...", "frame_indices": [0, 1] }
  ],
  "timing_ms": { "ffmpeg": 320, "infer_total": 4100 }
}
```

### Lifecycle commands

```bash
minicpm-v status     # show running server (PID / port / expire_at)
minicpm-v stop       # graceful shutdown (SIGTERM, falls back to SIGKILL)
minicpm-v stop --force
```

### Common flags

| Flag | Meaning |
|---|---|
| `--ttl <sec>` | Keep server alive for N seconds after this call (`0` = immediate teardown) |
| `--max-lifetime <sec>` | Hard cap on total server uptime |
| `--keep` | Don't expire after this call (manual stop required) |
| `--isolated` | Run server under sandbox-exec / bwrap |
| `--backend {auto,mlx,cuda,cpu}` | Override autodetection |
| `--quant <tag>` | Override quantization (`Q4_K_M`, `4bit`, etc.) |
| `--prompt "..."` | Override default caption prompt |

## Claude Code Skill

```bash
bash skill/install.sh
```

Installs to `~/.claude/skills/minicpm-v/`. Claude Code will auto-trigger the skill on image/video questions; the skill calls `minicpm-v` and feeds the JSON back.

## Testing & coverage

> The repository ships only source + tests. Install the dev extra before running tests.

```bash
pip install -e .[dev]
pytest tests/unit -v                                         # unit suite
pytest tests/integration -v                                  # integration (needs ffmpeg for video test)
pytest --cov=minicpm_v_local --cov-report=term-missing tests/
```

Target coverage (per spec §15.1):

- Core modules (`runtime`, `server`, `pipeline`, `download`, `config`): ≥ 80 %
- `cli`, `doctor`: ≥ 60 %

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `doctor` says GGUF for 4.6 not on HF yet | OpenBMB only ships fp16 + AWQ at release; community GGUFs land within ~1 week | Use `--backend cpu --quant Q5_K_M` against an older 4.5 GGUF, or wait |
| vLLM OOM at start | Default `gpu-memory-utilization` too high for your card | Lower via `~/.config/minicpm-v-local/config.toml` |
| Sandbox launch fails on macOS | `sandbox-exec` profile too strict | Re-run `minicpm-v doctor`, decline isolation, or relax the profile in `server/isolation.py` |
| Server doesn't auto-stop | Watchdog process killed manually | `minicpm-v stop --force`; next call re-spawns clean |
| `port_range exhausted` | Stale state file | `rm ~/.run/minicpm-v-local/state.json` |

## Design doc

See [`docs/specs/2026-05-13-minicpm-v-local-design.md`](docs/specs/2026-05-13-minicpm-v-local-design.md) for the full design (17 sections + appendices).
