#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_EXE="${PYTHON:-python3}"

exec "${PYTHON_EXE}" "${SCRIPT_DIR}/make_run_hazard_ml_v34_compact_thresholds.py" "$@"
