# Mac E2E Acceptance — TEMPLATE

> Copy to `docs/e2e-mac-acceptance-YYYY-MM-DD.md` and fill in.
> Required by plan §8.4 + spec §17 success criteria.

**Tester:** _______
**Date:** _______
**Hardware:** Apple Silicon (M? / RAM ?? GB / macOS ??)
**Repo SHA:** _______

## Step 1 — Install (dev + mlx)

- [ ] `pip install -e .[dev,mlx]` succeeds
- [ ] `minicpm-v --help` prints subcommands

Notes:

## Step 2 — `minicpm-v doctor` cold run

- [ ] Detects `mlx`
- [ ] Prompts for isolation mode (answer recorded: ____)
- [ ] Downloads 4-bit MLX weights into `~/.cache/minicpm-v-local/mlx/`
- [ ] Writes `~/.config/minicpm-v-local/config.toml`
- [ ] Exits 0

Notes:

## Step 3 — `minicpm-v image tests/fixtures/sample.jpg`

- [ ] Cold-start latency ≤ 15 s (spec §17)
- [ ] Stdout is valid JSON with `version`, `input.sha256`, `result.caption`
- [ ] `result.caption` is non-empty and matches the image (red square)

Latency: ____ ms. Notes:

## Step 4 — `minicpm-v video tests/fixtures/sample-5s.mp4`

- [ ] JSON has `frames[]` with ≥ 3 entries
- [ ] JSON has `scenes[]` with ≥ 1 entry
- [ ] `timing_ms.ffmpeg` and `timing_ms.infer_total` both present

Notes:

## Step 5 — `--ttl 10` auto-teardown

- [ ] `minicpm-v image sample.jpg --ttl 10` returns OK
- [ ] After 10 s, `ps aux | grep mlx_vlm` returns no rows
- [ ] `~/.run/minicpm-v-local/state.json` shows `alive: false` (or file is gone)

Wall-clock time observed: ____ s.

## Step 6 — `minicpm-v status` after teardown

- [ ] Reports `alive: false`

## Step 7 — Skill install

- [ ] `bash skill/install.sh` exits 0
- [ ] `~/.claude/skills/minicpm-v/SKILL.md` exists
- [ ] `~/.claude/skills/minicpm-v/scripts/run.sh` is executable

## Step 8 — Skill trigger in Claude Code

- [ ] Open a Claude Code session, drop an image, ask "describe this image"
- [ ] Claude invokes the `minicpm-v` skill
- [ ] Returned JSON is read and surfaced in the assistant reply

## Step 9 — Additional spec §17 criteria

- [ ] Warm path `image` ≤ 2 s
- [ ] `--ttl 0` immediate teardown verified
- [ ] No-`--ttl` (default 5 min) teardown verified
- [ ] Manual SIGSTOP → watchdog goes TERM → KILL → `cleanup_failed=true` (`server/watchdog.py`)
- [ ] Coverage: core ≥ 80 %, cli/doctor ≥ 60 %
- [ ] README + SKILL.md both mention "5-minute auto-unload"

## Outcome

- [ ] **PASS** — all boxes ticked
- [ ] **FAIL** — see notes below

Blockers / follow-ups:
