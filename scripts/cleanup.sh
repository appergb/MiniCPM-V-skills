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
