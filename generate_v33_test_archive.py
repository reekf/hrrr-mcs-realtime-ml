#!/usr/bin/env python3
"""Generate forecast-only website archive images for every v33 test case.

The source data are the v33 radius-sensitivity viewer prediction caches and its
historical WPC/PP grid.  Only the four ML radius forecasts and WPC ERO are
published; verification and proxy fields remain internal.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from realtime_mcs_trigger_plot import RuntimePaths, plot_realtime_ero_panels, radius_prob_col


RADII = (40, 60, 75, 100)
PROJECT_DIR = Path("/home/tyreekfrazier/ISU_Research_LOCAL_RUN/fall_2025_ml_proj")
PREDICTION_DIR = PROJECT_DIR / "v33_singletarget_radius_sensitivity_viewer_prediction_cache"
WPC_GRID = PROJECT_DIR / "df_pp_viewer_with_wpc_ero_day1.parquet"
REPO_DIR = Path(__file__).resolve().parent
ARCHIVE_DIR = REPO_DIR / "docs" / "archive"


def prediction_path(radius: int) -> Path:
    return PREDICTION_DIR / f"v33_singletarget_radius_sensitivity_predictions_r{radius}km.parquet"


def read_case(path: Path, date: str, columns: list[str]) -> pd.DataFrame:
    return pd.read_parquet(path, columns=columns, filters=[("Date", "==", date)])


def available_dates() -> list[str]:
    table = pq.read_table(prediction_path(RADII[0]), columns=["Date"])
    return sorted({str(value)[:8] for value in table.column("Date").to_pylist()})


def build_case_dataframe(date: str) -> pd.DataFrame:
    base = None
    expected_keys = None
    for radius in RADII:
        frame = read_case(
            prediction_path(radius),
            date,
            ["Date", "Lat", "Lon", "ML_Forecast_Prob"],
        ).rename(columns={"ML_Forecast_Prob": radius_prob_col(radius)})
        if frame.empty:
            raise RuntimeError(f"No v33 r{radius} prediction rows for {date}")
        frame = frame.sort_values(["Lat", "Lon"]).reset_index(drop=True)
        keys = frame[["Date", "Lat", "Lon"]]
        if base is None:
            base = frame
            expected_keys = keys
        else:
            if len(frame) != len(base) or not keys.equals(expected_keys):
                raise RuntimeError(f"v33 prediction grids do not align for {date}, r{radius}")
            base[radius_prob_col(radius)] = frame[radius_prob_col(radius)].to_numpy()

    wpc = read_case(WPC_GRID, date, ["Date", "Lat", "Lon", "WPC_ERO_Risk"])
    if wpc.empty:
        raise RuntimeError(f"No historical WPC viewer rows for {date}")
    wpc = wpc.sort_values(["Lat", "Lon"]).reset_index(drop=True)
    if len(wpc) != len(base) or not wpc[["Date", "Lat", "Lon"]].equals(expected_keys):
        raise RuntimeError(f"Historical WPC grid does not align with v33 predictions for {date}")
    base["WPC_ERO_Risk"] = wpc["WPC_ERO_Risk"].fillna(0).to_numpy()
    return base


def runtime_paths(outdir: Path) -> RuntimePaths:
    cache = PROJECT_DIR / "v33_realtime_radiusstats_forecasts"
    return RuntimePaths(
        project_dir=PROJECT_DIR,
        script_dir=REPO_DIR,
        cache_dir=cache,
        feature_cache_dir=cache / "features",
        prediction_cache_dir=cache / "predictions",
        verified_cache_dir=cache / "verified",
        ufvs_cache_dir=cache / "ufvs_raw",
        wpc_cache_dir=PROJECT_DIR / "realtime_wpc_ero_cache_v33",
        pp_cache_dir=PROJECT_DIR / "realtime_pp_from_ufvs_cache_v33",
        outdir=outdir,
        original_root=Path("/home/tyreekfrazier/ISU_Research"),
        local_root=Path("/home/tyreekfrazier/ISU_Research_LOCAL_RUN"),
    )


def write_status(date: str, destination: Path) -> None:
    start = datetime.strptime(date + "12", "%Y%m%d%H").replace(tzinfo=timezone.utc)
    end = start + timedelta(days=1)
    status = {
        "published": True,
        "plot_available": True,
        "date": date,
        "valid_start_utc": start.isoformat().replace("+00:00", "Z"),
        "valid_end_utc": end.isoformat().replace("+00:00", "Z"),
        "valid_period_label": f"{start:%Y-%m-%d} 12Z to {end:%Y-%m-%d} 12Z",
        "latest_plot": "latest.png",
        "site_updated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "product_description": "Machine-learning radius products plus WPC ERO.",
    }
    destination.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")


def rebuild_archive_index() -> None:
    entries = []
    for day_dir in sorted((p for p in ARCHIVE_DIR.iterdir() if p.is_dir()), reverse=True):
        status_path = day_dir / "status.json"
        if not status_path.exists():
            continue
        status = json.loads(status_path.read_text())
        plot_exists = (day_dir / "latest.png").exists()
        entries.append(
            {
                "date": str(status.get("date") or day_dir.name),
                "valid_period_label": status.get("valid_period_label", ""),
                "published": bool(status.get("published", False)),
                "plot_available": bool(plot_exists and status.get("plot_available", False)),
                "site_updated_utc": status.get("site_updated_utc", ""),
                "status_href": f"archive/{day_dir.name}/status.json",
                "plot_href": f"archive/{day_dir.name}/latest.png" if plot_exists else None,
            }
        )
    payload = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "entries": entries,
    }
    (ARCHIVE_DIR / "index.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def generate_case(date: str, force: bool = False) -> None:
    day_dir = ARCHIVE_DIR / date
    output = day_dir / "latest.png"
    status = day_dir / "status.json"
    if output.exists() and status.exists() and not force:
        print(f"[{date}] already archived; skipping", flush=True)
        return

    print(f"[{date}] loading v33 viewer caches", flush=True)
    frame = build_case_dataframe(date)
    day_dir.mkdir(parents=True, exist_ok=True)
    generated = plot_realtime_ero_panels(
        frame,
        date=date,
        rp=runtime_paths(day_dir),
        radii=list(RADII),
        include_wpc=True,
        include_ufvs=False,
        include_pp=False,
    )
    os.replace(generated, output)
    write_status(date, status)
    print(f"[{date}] wrote {output}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", action="append", help="Generate only this YYYYMMDD date; repeatable.")
    parser.add_argument("--force", action="store_true", help="Replace existing archive images/status files.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dates = sorted(args.date) if args.date else available_dates()
    print(f"Generating {len(dates)} v33 test cases sequentially", flush=True)
    for date in dates:
        generate_case(str(date)[:8], force=args.force)
    rebuild_archive_index()
    print(f"Updated {ARCHIVE_DIR / 'index.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
