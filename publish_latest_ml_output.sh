#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/home/tyreekfrazier/ISU_Research_LOCAL_RUN/mesoanalysis/gempak-scripts"
PROJECT_DIR="/home/tyreekfrazier/ISU_Research_LOCAL_RUN/fall_2025_ml_proj"
OUT_DIR="${PROJECT_DIR}/v33_realtime_radiusstats_forecasts/mcs_triggered_figures"

DATE_ARG="${1:-$(date -u +%Y%m%d)}"
RADII="${RADII:-40 60 75 100}"
PUBLIC_PNG_NAME="realtime_ml_public_${DATE_ARG}_valid12to12_radii_wpc.png"
PUBLIC_PNG_SRC="${OUT_DIR}/${PUBLIC_PNG_NAME}"

cd "$REPO_DIR"

echo "======================================================================"
echo "Realtime ML site publish"
echo "Date: ${DATE_ARG}"
echo "Started UTC: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Repo: ${REPO_DIR}"
echo "Output dir: ${OUT_DIR}"
echo "======================================================================"

# Pages deploys from main/docs.
git switch main
git pull --ff-only origin main

# Prevent stale contour-era graphics from being copied if the plotter fails.
rm -f "${PUBLIC_PNG_SRC}"

echo
echo "Running realtime ML plotter..."
python realtime_mcs_trigger_plot.py \
  --date "$DATE_ARG" \
  --radii $RADII

echo
echo "Finding script outputs..."

STATUS_SRC="$(find "$OUT_DIR" -maxdepth 1 -type f -name "status_${DATE_ARG}_*.json" -printf "%T@ %p\n" 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2- || true)"
PNG_SRC=""
if [[ -f "${PUBLIC_PNG_SRC}" ]]; then
  PNG_SRC="${PUBLIC_PNG_SRC}"
fi

mkdir -p docs/latest "docs/archive/${DATE_ARG}"

# Build a public status JSON from scratch. Do not expose internal generation-gate metadata.
write_public_status() {
  local dst="$1"
  local published="$2"
  local plot_available="$3"
  local message="${4:-}"
  python - "$DATE_ARG" "$dst" "$published" "$plot_available" "$message" <<'PY'
import json
import sys
from datetime import datetime, timezone, timedelta

date, dst, published, plot_available, message = sys.argv[1:6]
published = published.lower() == "true"
plot_available = plot_available.lower() == "true"

start = datetime.strptime(date + "12", "%Y%m%d%H").replace(tzinfo=timezone.utc)
end = start + timedelta(days=1)
status = {
    "published": published,
    "plot_available": plot_available,
    "date": date,
    "valid_start_utc": start.isoformat().replace("+00:00", "Z"),
    "valid_end_utc": end.isoformat().replace("+00:00", "Z"),
    "valid_period_label": f"{start:%Y-%m-%d} 12Z to {end:%Y-%m-%d} 12Z",
    "latest_plot": "latest.png" if plot_available else None,
    "site_updated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "product_description": "Machine-learning radius products plus WPC ERO.",
}
if message:
    status["message"] = message
elif not plot_available:
    status["message"] = "No forecast graphic is available for this date."
with open(dst, "w") as f:
    json.dump(status, f, indent=2, sort_keys=True)
PY
}

if [[ -n "${PNG_SRC}" && -f "${PNG_SRC}" ]]; then
  echo "Public PNG source: ${PNG_SRC}"
  cp "${PNG_SRC}" docs/latest/latest.png
  cp "${PNG_SRC}" "docs/archive/${DATE_ARG}/latest.png"
  write_public_status docs/latest/status.json true true ""
  cp docs/latest/status.json "docs/archive/${DATE_ARG}/status.json"
else
  echo "WARNING: Public PNG was not produced: ${PUBLIC_PNG_SRC}"
  rm -f docs/latest/latest.png "docs/archive/${DATE_ARG}/latest.png"
  write_public_status docs/latest/status.json false false "Realtime script ran, but no public forecast graphic was produced."
  cp docs/latest/status.json "docs/archive/${DATE_ARG}/status.json"
fi

# Build archive manifest from all public archived status files.
python - <<'PY'
import json
from pathlib import Path
from datetime import datetime, timezone

archive_root = Path("docs/archive")
entries = []
if archive_root.exists():
    for day_dir in sorted([p for p in archive_root.iterdir() if p.is_dir()], reverse=True):
        status_path = day_dir / "status.json"
        if not status_path.exists():
            continue
        try:
            status = json.loads(status_path.read_text())
        except Exception:
            status = {}
        date = status.get("date") or day_dir.name
        plot_exists = (day_dir / "latest.png").exists()
        verification_exists = (day_dir / "verification.png").exists()
        entries.append({
            "date": str(date),
            "valid_period_label": status.get("valid_period_label", ""),
            "published": bool(status.get("published", False)),
            "plot_available": bool(plot_exists and status.get("plot_available", False)),
            "site_updated_utc": status.get("site_updated_utc", ""),
            "status_href": f"archive/{day_dir.name}/status.json",
            "plot_href": f"archive/{day_dir.name}/latest.png" if plot_exists else None,
            "verification_available": bool(verification_exists),
            "verification_plot_href": f"archive/{day_dir.name}/verification.png" if verification_exists else None,
            "verification_updated_utc": status.get("verification_updated_utc", ""),
        })

out = {
    "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "entries": entries,
}
Path("docs/archive").mkdir(parents=True, exist_ok=True)
Path("docs/archive/index.json").write_text(json.dumps(out, indent=2, sort_keys=True))
PY

echo
echo "Committing website update if changed..."

git add -f docs/latest/status.json docs/archive/index.json
if [[ -f docs/latest/latest.png ]]; then
  git add -f docs/latest/latest.png
else
  git rm -f --ignore-unmatch docs/latest/latest.png >/dev/null 2>&1 || true
fi
git add -f "docs/archive/${DATE_ARG}/status.json"
if [[ -f "docs/archive/${DATE_ARG}/latest.png" ]]; then
  git add -f "docs/archive/${DATE_ARG}/latest.png"
else
  git rm -f --ignore-unmatch "docs/archive/${DATE_ARG}/latest.png" >/dev/null 2>&1 || true
fi

PUBLISH_PATHS=(docs/latest/status.json docs/archive/index.json docs/latest/latest.png "docs/archive/${DATE_ARG}/status.json" "docs/archive/${DATE_ARG}/latest.png")
if git diff --cached --quiet -- "${PUBLISH_PATHS[@]}"; then
  echo "No website changes to commit."
else
  git commit -m "Publish realtime ML forecast for ${DATE_ARG}" -- "${PUBLISH_PATHS[@]}"
  git push origin main
fi

echo
echo "Done UTC: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
