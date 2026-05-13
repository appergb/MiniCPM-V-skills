# v1 Acceptance Checklist

> Run by user on Mac (Apple Silicon). Tick each box after verifying.
> Sources: spec §17 + plan Task 8.4.

## A. Spec §17 success criteria (10 items)
- [ ] Mac: `minicpm-v doctor` 全程跑通，权重落到 `~/.cache/…`
- [ ] `minicpm-v image sample.jpg` cold ≤ 15s 出 JSON；warm ≤ 2s
- [ ] `minicpm-v video sample-30s.mp4` 出 ≥ 3 frame + ≥ 1 scene 的 JSON
- [ ] 不传 `--ttl`，5 分钟后 `ps aux | grep mlx_vlm` 无进程；state.json `alive=false`
- [ ] `--ttl 10` 调用结束后 10s 内 server 进程消失
- [ ] `--ttl 0` 调用结束立即销毁
- [ ] Claude Code 里 SKILL.md 能被自动触发，JSON stdout 能解析
- [ ] 手动 SIGSTOP 模拟 hung 后，watchdog 走完 TERM→KILL→cleanup_failed
- [ ] 测试覆盖：核心模块 ≥ 80%，CLI/doctor ≥ 60%
- [ ] README + SKILL.md 都写明"5 分钟自动卸载"

## B. Plan Task 8.4 manual E2E (9 steps)
- [ ] `pip install -e .[dev,mlx]`
- [ ] `minicpm-v doctor` → 拉 4bit 模型
- [ ] `minicpm-v image tests/fixtures/sample.jpg` → JSON 正确
- [ ] `minicpm-v video tests/fixtures/sample-5s.mp4` → JSON 含 frames + scenes
- [ ] `--ttl 10` 验证 10s 后 ps aux 找不到 server
- [ ] `minicpm-v status` 显示 alive=false
- [ ] `bash skill/install.sh` → `~/.claude/skills/minicpm-v/SKILL.md` 存在
- [ ] Claude Code 实际触发 SKILL.md（手测）
- [ ] 结果记到 `docs/e2e-mac-acceptance-2026-05-13.md`

## C. Final cleanup steps (plan 9.2)
- [ ] `bash scripts/cleanup.sh`
- [ ] `python -m venv .venv && source .venv/bin/activate && pip install -e .[dev] && pytest tests/unit -v`
- [ ] `deactivate && rm -rf .venv`
- [ ] `git log --oneline` 用作交付历史

**v1 Definition of Done**: A + B + C all ✓ on Mac.
