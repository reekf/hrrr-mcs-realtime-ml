#!/usr/bin/env bash
set -euo pipefail

# Keep one authoritative publishing implementation and one Git checkout.
# The former in-repository copy targeted a separate, stale xgbffp clone, which
# could create local publish commits on a branch that was already behind main.
PUBLISHER_SCRIPT="${XGBFFP_PUBLISHER_SCRIPT:-/home/tyreekfrazier/ISU_Research_LOCAL_RUN/mesoanalysis/xgbffp-publisher/publish_latest_ml_output.sh}"

if [[ ! -x "$PUBLISHER_SCRIPT" ]]; then
  echo "ERROR: XGBFFP publisher is missing or not executable: $PUBLISHER_SCRIPT" >&2
  exit 1
fi

exec "$PUBLISHER_SCRIPT" "$@"
