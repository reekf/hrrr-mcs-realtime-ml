#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
OUT_DIR="${1:-/tmp}"

mkdir -p "$OUT_DIR"

PROMPT_FILE="$OUT_DIR/agy_repo_review_prompt_$(date -u +%Y%m%dT%H%M%SZ).md"

{
  echo "# Antigravity second-opinion review"
  echo
  echo "You are reviewing a weather ML / GitHub Pages codebase."
  echo
  echo "Focus on correctness, not style."
  echo
  echo "Review for:"
  echo "1. Bugs in the current diff"
  echo "2. WPC ERO risk-category handling"
  echo "3. v33 verification/statistics logic"
  echo "4. GitHub Pages publishing and archive behavior"
  echo "5. Cron/automation silent-failure paths"
  echo "6. Any risk of exposing internal trigger/debug details publicly"
  echo
  echo "Do not edit files. Return concise, actionable findings only."
  echo
  echo "## Repository"
  echo
  echo '```text'
  echo "$REPO_DIR"
  echo '```'
  echo
  echo "## git branch"
  echo
  echo '```text'
  git -C "$REPO_DIR" branch --show-current || true
  echo '```'
  echo
  echo "## git status --short"
  echo
  echo '```text'
  git -C "$REPO_DIR" status --short || true
  echo '```'
  echo
  echo "## git diff --stat"
  echo
  echo '```text'
  git -C "$REPO_DIR" diff --stat || true
  echo '```'
  echo
  echo "## git diff"
  echo
  echo '```diff'
  git -C "$REPO_DIR" diff || true
  echo '```'
} > "$PROMPT_FILE"

echo "$PROMPT_FILE"
