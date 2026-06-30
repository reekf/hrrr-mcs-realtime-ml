#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/home/tyreekfrazier/ISU_Research_LOCAL_RUN/mesoanalysis/gempak-scripts"
PROJECT_DIR="/home/tyreekfrazier/ISU_Research_LOCAL_RUN/fall_2025_ml_proj"
OUT_DIR="${PROJECT_DIR}/v33_realtime_radiusstats_forecasts/mcs_triggered_figures"

DATE_ARG="${1:-$(date -u +%Y%m%d)}"
RADII="${RADII:-40 60 75 100}"

cd "$REPO_DIR"

echo "======================================================================"
echo "Realtime ML site publish"
echo "Date: ${DATE_ARG}"
echo "Started UTC: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Repo: ${REPO_DIR}"
echo "Output dir: ${OUT_DIR}"
echo "======================================================================"

# Make sure Pages branch is current.
git switch main
git pull --ff-only origin main

echo
echo "Running realtime MCS-triggered ML plotter..."
python realtime_mcs_trigger_plot.py \
  --date "$DATE_ARG" \
  --radii $RADII

echo
echo "Finding script outputs..."

STATUS_SRC="$(find "$OUT_DIR" -maxdepth 1 -type f -name "status_${DATE_ARG}_*.json" -printf "%T@ %p\n" 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2- || true)"
PNG_SRC="$(find "$OUT_DIR" -maxdepth 1 -type f -name "*${DATE_ARG}*.png" -printf "%T@ %p\n" 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2- || true)"

mkdir -p docs/latest docs/archive/"${DATE_ARG}"

if [[ -z "${STATUS_SRC}" || ! -f "${STATUS_SRC}" ]]; then
  echo "WARNING: No status JSON found. Publishing site status as failed/no output."
  cat > docs/latest/status.json <<EOF_STATUS
{
  "published": false,
  "date": "${DATE_ARG}",
  "site_updated_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "message": "No status JSON was produced by realtime_mcs_trigger_plot.py."
}
EOF_STATUS
  rm -f docs/latest/latest.png
else
  echo "Status source: ${STATUS_SRC}"

  if [[ -n "${PNG_SRC}" && -f "${PNG_SRC}" ]]; then
    echo "PNG source: ${PNG_SRC}"

    cp "${PNG_SRC}" docs/latest/latest.png
    cp "${PNG_SRC}" "docs/archive/${DATE_ARG}/$(basename "${PNG_SRC}")"
    cp "${STATUS_SRC}" "docs/archive/${DATE_ARG}/$(basename "${STATUS_SRC}")"

    python - "$STATUS_SRC" docs/latest/status.json "$(basename "${PNG_SRC}")" <<'PY'
import json
import sys
from datetime import datetime, timezone

src, dst, png_name = sys.argv[1], sys.argv[2], sys.argv[3]

with open(src, "r") as f:
    status = json.load(f)

status["published"] = True
status["plot_available"] = True
status["latest_plot"] = "latest.png"
status["archive_plot_name"] = png_name
status["site_updated_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

with open(dst, "w") as f:
    json.dump(status, f, indent=2, sort_keys=True)
PY

  else
    echo "No PNG found. Publishing status without plot."
    rm -f docs/latest/latest.png

    python - "$STATUS_SRC" docs/latest/status.json <<'PY'
import json
import sys
from datetime import datetime, timezone

src, dst = sys.argv[1], sys.argv[2]

with open(src, "r") as f:
    status = json.load(f)

status["published"] = False
status["plot_available"] = False
status["latest_plot"] = None
status["site_updated_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

if "message" not in status:
    status["message"] = "Realtime script ran, but no plot PNG was produced."

with open(dst, "w") as f:
    json.dump(status, f, indent=2, sort_keys=True)
PY

  fi
fi

echo
echo "Committing website update if changed..."

git add -f docs/latest/status.json
if [[ -f docs/latest/latest.png ]]; then
  git add -f docs/latest/latest.png
fi
git add -f docs/archive/"${DATE_ARG}" || true

if git diff --cached --quiet; then
  echo "No website changes to commit."
else
  git commit -m "Publish realtime ML output for ${DATE_ARG}"
  git push origin main
fi

echo
echo "Done UTC: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
