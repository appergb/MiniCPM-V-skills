#!/usr/bin/env bash
set -euo pipefail
TARGET="$HOME/.claude/skills/minicpm-v"
mkdir -p "$TARGET/scripts"
cp "$(dirname "$0")/SKILL.md" "$TARGET/SKILL.md"
cp "$(dirname "$0")/scripts/run.sh" "$TARGET/scripts/run.sh"
chmod +x "$TARGET/scripts/run.sh"
echo "Skill installed to $TARGET"
