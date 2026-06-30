#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
PROMPT_FILE="${1:-}"

LMSTUDIO_BASE_URL="${LMSTUDIO_BASE_URL:-http://localhost:1234/v1}"
LMSTUDIO_MODEL="${LMSTUDIO_MODEL:-local-model}"
LMSTUDIO_TEMPERATURE="${LMSTUDIO_TEMPERATURE:-0.2}"
LMSTUDIO_MAX_TOKENS="${LMSTUDIO_MAX_TOKENS:-2048}"

if [[ -z "$PROMPT_FILE" ]]; then
  PROMPT_FILE="$("$REPO_DIR/.agents/skills/lmstudio-review/scripts/make_lmstudio_prompt.sh" /tmp)"
  echo "Created prompt file: $PROMPT_FILE" >&2
else
  if [[ ! -f "$PROMPT_FILE" ]]; then
    echo "ERROR: Prompt file does not exist: $PROMPT_FILE" >&2
    exit 2
  fi
fi

echo "Checking LM Studio server: $LMSTUDIO_BASE_URL/models" >&2

if ! curl -fsS "$LMSTUDIO_BASE_URL/models" >/tmp/lmstudio_models.json 2>/tmp/lmstudio_models_stderr.log; then
  echo "ERROR: Could not reach LM Studio server at $LMSTUDIO_BASE_URL" >&2
  echo "Make sure LM Studio Local Server is running." >&2
  cat /tmp/lmstudio_models_stderr.log >&2 || true
  echo "Prompt file is ready here:" >&2
  echo "  $PROMPT_FILE" >&2
  exit 127
fi

python - "$PROMPT_FILE" "$LMSTUDIO_BASE_URL" "$LMSTUDIO_MODEL" "$LMSTUDIO_TEMPERATURE" "$LMSTUDIO_MAX_TOKENS" <<'PY'
import json
import sys
import urllib.request
import urllib.error

prompt_file, base_url, model, temperature, max_tokens = sys.argv[1:]
temperature = float(temperature)
max_tokens = int(max_tokens)

with open(prompt_file, "r", encoding="utf-8") as f:
    prompt = f.read()

payload = {
    "model": model,
    "messages": [
        {
            "role": "system",
            "content": (
                "You are a careful code reviewer. Be concise. "
                "Focus on correctness bugs, missing validation, automation failure paths, "
                "WPC ERO risk logic, v33 verification/statistics issues, and website publishing issues. "
                "Do not suggest broad rewrites unless necessary."
            ),
        },
        {
            "role": "user",
            "content": prompt,
        },
    ],
    "temperature": temperature,
    "max_tokens": max_tokens,
    "stream": False,
}

data = json.dumps(payload).encode("utf-8")

req = urllib.request.Request(
    base_url.rstrip("/") + "/chat/completions",
    data=data,
    headers={"Content-Type": "application/json"},
    method="POST",
)

try:
    with urllib.request.urlopen(req, timeout=240) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
except urllib.error.HTTPError as e:
    body = e.read().decode("utf-8", errors="replace")
    print(f"ERROR: LM Studio HTTP {e.code}", file=sys.stderr)
    print(body, file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"ERROR: LM Studio request failed: {e!r}", file=sys.stderr)
    sys.exit(1)

try:
    obj = json.loads(raw)
    msg = obj["choices"][0]["message"]["content"]
except Exception:
    print("ERROR: Could not parse LM Studio response:", file=sys.stderr)
    print(raw[:2000], file=sys.stderr)
    sys.exit(1)

print(msg)
PY
