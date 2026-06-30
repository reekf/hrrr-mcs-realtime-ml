#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
PROMPT_FILE="${1:-}"

if [[ -z "$PROMPT_FILE" ]]; then
  PROMPT_FILE="$("$REPO_DIR/.agents/skills/antigravity-review/scripts/make_agy_prompt.sh" /tmp)"
  echo "Created prompt file: $PROMPT_FILE" >&2
else
  if [[ ! -f "$PROMPT_FILE" ]]; then
    echo "ERROR: Prompt file does not exist: $PROMPT_FILE" >&2
    exit 2
  fi
fi

if ! command -v agy >/dev/null 2>&1; then
  echo "ERROR: Antigravity CLI command 'agy' was not found in PATH." >&2
  echo "Prompt file is ready here:" >&2
  echo "  $PROMPT_FILE" >&2
  exit 127
fi

cd "$REPO_DIR"

echo "Running Antigravity review from repo: $REPO_DIR" >&2
echo "agy version: $(agy --version 2>/dev/null || true)" >&2

RESPONSE_FILE="/tmp/agy_review_response_$(date -u +%Y%m%dT%H%M%SZ).md"
STDERR_FILE="/tmp/agy_review_stderr_$(date -u +%Y%m%dT%H%M%SZ).log"

set +e
agy -p "$(cat "$PROMPT_FILE")" > "$RESPONSE_FILE" 2> "$STDERR_FILE"
STATUS=$?
set -e

echo "agy exit status: $STATUS" >&2

if [[ -s "$STDERR_FILE" ]]; then
  echo "--- agy stderr ---" >&2
  cat "$STDERR_FILE" >&2
  echo "--- end agy stderr ---" >&2
fi

if [[ $STATUS -ne 0 || ! -s "$RESPONSE_FILE" ]]; then
  echo "ERROR: agy did not produce a usable response." >&2
  echo "Prompt file:" >&2
  echo "  $PROMPT_FILE" >&2
  echo "stderr file:" >&2
  echo "  $STDERR_FILE" >&2
  exit 1
fi

cat "$RESPONSE_FILE"
