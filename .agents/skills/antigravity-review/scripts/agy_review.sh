#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
PROMPT_FILE="${1:-}"

if [[ -z "$PROMPT_FILE" ]]; then
  PROMPT_FILE="/tmp/agy_repo_review_prompt_$(date -u +%Y%m%dT%H%M%SZ).md"

  {
    echo "# Antigravity second-opinion review"
    echo
    echo "You are reviewing a weather ML / GitHub Pages codebase."
    echo
    echo "Focus on correctness, not style."
    echo
    echo "Please review for:"
    echo "1. Bugs in the current diff"
    echo "2. WPC ERO risk-category handling"
    echo "3. Public website publishing/verification automation"
    echo "4. Any risk of exposing internal trigger/debug details publicly"
    echo "5. Missing tests or commands that should be run"
    echo
    echo "Do not edit files. Return a concise review with actionable issues only."
    echo
    echo "## Repository"
    echo
    echo '```text'
    echo "$REPO_DIR"
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

  echo "Created prompt file: $PROMPT_FILE" >&2
else
  if [[ ! -f "$PROMPT_FILE" ]]; then
    echo "ERROR: Prompt file does not exist: $PROMPT_FILE" >&2
    exit 2
  fi
fi

if ! command -v agy >/dev/null 2>&1; then
  echo "ERROR: Antigravity CLI command 'agy' was not found in PATH." >&2
  echo "The skill exists, but Antigravity is not installed/authenticated in this environment yet." >&2
  echo "Prompt file was created here:" >&2
  echo "  $PROMPT_FILE" >&2
  exit 127
fi

WORKDIR="$(mktemp -d /tmp/agy-review.XXXXXX)"
trap 'rm -rf "$WORKDIR"' EXIT

cp "$PROMPT_FILE" "$WORKDIR/prompt.md"
cd "$WORKDIR"

echo "Running Antigravity review from temp dir: $WORKDIR" >&2
echo "agy version: $(agy --version 2>/dev/null || true)" >&2

set +e
agy -p "$(cat prompt.md)" > response.md 2> stderr.log
STATUS=$?
set -e

echo "agy exit status: $STATUS" >&2

if [[ -s stderr.log ]]; then
  echo "--- agy stderr ---" >&2
  cat stderr.log >&2
  echo "--- end agy stderr ---" >&2
fi

if [[ ! -s response.md ]]; then
  echo "ERROR: agy produced no stdout response." >&2
  echo "This may mean Antigravity is not installed, not authenticated, or requires interactive login." >&2
  exit 1
fi

cat response.md
