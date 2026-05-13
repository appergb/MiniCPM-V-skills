#!/usr/bin/env bash
# Install the minicpm-v skill bundle to one or more agent skill dirs.
#
# Default behavior (no env vars): install to ~/.claude/skills/minicpm-v/, plus
# ~/.deepseek/skills/minicpm-v/ if ~/.deepseek/ exists.
#
# Override with TARGET env var to write to a single explicit path:
#   TARGET=$HOME/.codex/skills/minicpm-v bash skill/install.sh
set -euo pipefail

SRC_DIR="$(cd "$(dirname "$0")" && pwd)"

if [[ -n "${TARGET:-}" ]]; then
  TARGETS=("$TARGET")
else
  TARGETS=("$HOME/.claude/skills/minicpm-v")
  if [[ -d "$HOME/.deepseek" ]]; then
    TARGETS+=("$HOME/.deepseek/skills/minicpm-v")
  fi
fi

for target in "${TARGETS[@]}"; do
  mkdir -p "$target/scripts"
  cp "$SRC_DIR/SKILL.md" "$target/SKILL.md"
  cp "$SRC_DIR/scripts/run.sh" "$target/scripts/run.sh"
  chmod +x "$target/scripts/run.sh"
  echo "Skill installed to $target"
done
