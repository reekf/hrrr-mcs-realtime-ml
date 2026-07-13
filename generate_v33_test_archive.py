#!/usr/bin/env python3
"""Generate website archive images and maps for every v33 historical test case.

The source data are the v33 radius-sensitivity viewer prediction caches and its
historical WPC/PP grid. The archive contains the four original ML radius
forecasts, the distinct density-weighted r60kmV2 member, their ensemble mean,
WPC ERO, and the viewer's Practically Perfect Any flood-proxy verification.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from realtime_mcs_trigger_plot import (
    RuntimePaths,
    add_ensemble_mean,
    plot_realtime_ero_panels,
    predict_realtime_r60km_v2_case,
    radius_prob_col,
)
from generate_interactive_map_data import _merge_aligned, load_realtime, write_frame_map_data


RADII = (40, 60, 75, 100)
PROJECT_DIR = Path("/home/tyreekfrazier/ISU_Research_LOCAL_RUN/fall_2025_ml_proj")
PREDICTION_DIR = PROJECT_DIR / "v33_singletarget_radius_sensitivity_viewer_prediction_cache"
WPC_GRID = PROJECT_DIR / "df_pp_viewer_with_wpc_ero_day1.parquet"
REPO_DIR = Path(__file__).resolve().parent
ARCHIVE_DIR = REPO_DIR / "docs" / "archive"


def prediction_path(radius: int) -> Path:
    return PREDICTION_DIR / f"v33_singletarget_radius_sensitivity_predictions_r{radius}km.parquet"


R60KM_V2_PREDICTION = PREDICTION_DIR / "v33_singletarget_radius_sensitivity_predictions_r60kmV2_expanded40union.parquet"


def read_case(path: Path, date: str, columns: list[str]) -> pd.DataFrame:
    return pd.read_parquet(path, columns=columns, filters=[("Date", "==", date)])


def available_dates() -> list[str]:
    paths = [prediction_path(radius) for radius in RADII] + [R60KM_V2_PREDICTION]
    date_sets = []
    for path in paths:
        table = pq.read_table(path, columns=["Date"])
        date_sets.append({str(value)[:8] for value in table.column("Date").to_pylist()})
    return sorted(set.intersection(*date_sets))


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

    r60v2 = read_case(
        R60KM_V2_PREDICTION,
        date,
        ["Date", "Lat", "Lon", "ML_Forecast_Prob"],
    ).rename(columns={"ML_Forecast_Prob": "ML_r60kmV2_Prob"})
    if r60v2.empty:
        raise RuntimeError(f"No v33 r60kmV2 prediction rows for {date}")
    r60v2 = r60v2.sort_values(["Lat", "Lon"]).reset_index(drop=True)
    if len(r60v2) != len(base) or not r60v2[["Date", "Lat", "Lon"]].equals(expected_keys):
        raise RuntimeError(f"v33 prediction grids do not align for {date}, r60kmV2")
    base["ML_r60kmV2_Prob"] = r60v2["ML_r60kmV2_Prob"].to_numpy()

    wpc = read_case(
        WPC_GRID,
        date,
        ["Date", "Lat", "Lon", "WPC_ERO_Risk", "PP_Any flood proxy"],
    )
    if wpc.empty:
        raise RuntimeError(f"No historical WPC viewer rows for {date}")
    wpc = wpc.sort_values(["Lat", "Lon"]).reset_index(drop=True)
    if len(wpc) != len(base) or not wpc[["Date", "Lat", "Lon"]].equals(expected_keys):
        raise RuntimeError(f"Historical WPC grid does not align with v33 predictions for {date}")
    base["WPC_ERO_Risk"] = wpc["WPC_ERO_Risk"].fillna(0).to_numpy()
    base["PP_Any flood proxy"] = wpc["PP_Any flood proxy"].fillna(0).to_numpy()
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
        "product_description": "Machine-learning radius products including density-weighted r60kmV2, ensemble mean, WPC ERO, and Practically Perfect verification.",
        "map_available": True,
        "map_data": "map.json",
        "map_updated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "verification_available": True,
        "verification_plot": "latest.png",
        "verification_embedded_in_forecast": True,
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
        map_exists = (day_dir / "map.json").exists()
        verification_exists = (day_dir / "verification.png").exists()
        verification_embedded = bool(status.get("verification_embedded_in_forecast", False)) or (
            "practically perfect verification" in str(status.get("product_description", "")).lower()
        )
        entries.append(
            {
                "date": str(status.get("date") or day_dir.name),
                "valid_period_label": status.get("valid_period_label", ""),
                "published": bool(status.get("published", False)),
                "plot_available": bool(plot_exists and status.get("plot_available", False)),
                "site_updated_utc": status.get("site_updated_utc", ""),
                "status_href": f"archive/{day_dir.name}/status.json",
                "plot_href": f"archive/{day_dir.name}/latest.png" if plot_exists else None,
                "map_available": bool(map_exists),
                "map_href": f"archive/{day_dir.name}/map.json" if map_exists else None,
                "map_updated_utc": status.get("map_updated_utc", status.get("site_updated_utc", "")),
                "verification_available": bool(verification_exists or (verification_embedded and plot_exists)),
                "verification_plot_href": (
                    f"archive/{day_dir.name}/verification.png" if verification_exists
                    else (f"archive/{day_dir.name}/latest.png" if verification_embedded and plot_exists else None)
                ),
                "verification_embedded_in_forecast": bool(verification_embedded and not verification_exists),
                "verification_updated_utc": status.get("verification_updated_utc", status.get("site_updated_utc", "")),
            }
        )
    payload = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "entries": entries,
    }
    (ARCHIVE_DIR / "index.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def realtime_archive_dates() -> list[str]:
    historical = set(available_dates())
    feature_dir = PROJECT_DIR / "v33_realtime_radiusstats_forecasts" / "features"
    out = []
    for day_dir in ARCHIVE_DIR.iterdir():
        if not day_dir.is_dir() or day_dir.name in historical:
            continue
        feature_path = feature_dir / f"realtime_features_v33_r60km_{day_dir.name}.parquet"
        if feature_path.exists():
            out.append(day_dir.name)
    return sorted(out)


def generate_realtime_case(date: str, force: bool = False) -> None:
    """Add r60kmV2 to an already-generated operational website date."""
    day_dir = ARCHIVE_DIR / date
    output = day_dir / "latest.png"
    map_output = day_dir / "map.json"
    status_path = day_dir / "status.json"
    print(f"[{date}] adding r60kmV2 to operational archive", flush=True)
    rp = runtime_paths(day_dir)
    v2 = predict_realtime_r60km_v2_case(
        date,
        rp=rp,
        force_predict=force,
        force_features=False,
        allow_feature_nan_fill_zero=True,
    )
    frame = load_realtime(date)
    v2 = v2.rename(columns={"ML_Forecast_Prob": "ML_r60kmV2_Prob"})
    frame = _merge_aligned(frame, v2, ["ML_r60kmV2_Prob"])
    frame = add_ensemble_mean(frame, list(RADII))
    generated = plot_realtime_ero_panels(
        frame,
        date=date,
        rp=rp,
        radii=list(RADII),
        include_wpc=True,
        include_ufvs=False,
        include_pp=False,
    )
    os.replace(generated, output)
    write_frame_map_data(frame, date, map_output, "realtime")
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    status.update({
        "published": True,
        "plot_available": True,
        "map_available": True,
        "map_data": "map.json",
        "map_updated_utc": now,
        "site_updated_utc": now,
        "product_description": "Machine-learning radius products including density-weighted r60kmV2, ensemble mean, and WPC ERO.",
    })
    status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
    print(f"[{date}] updated operational PNG/map/status", flush=True)


def generate_case(date: str, force: bool = False) -> None:
    day_dir = ARCHIVE_DIR / date
    output = day_dir / "latest.png"
    map_output = day_dir / "map.json"
    status = day_dir / "status.json"
    if output.exists() and map_output.exists() and status.exists() and not force:
        print(f"[{date}] already archived; skipping", flush=True)
        return

    print(f"[{date}] loading v33 viewer caches", flush=True)
    frame = add_ensemble_mean(build_case_dataframe(date), list(RADII))
    day_dir.mkdir(parents=True, exist_ok=True)
    if force or not output.exists():
        generated = plot_realtime_ero_panels(
            frame,
            date=date,
            rp=runtime_paths(day_dir),
            radii=list(RADII),
            include_wpc=True,
            include_ufvs=False,
            include_pp=True,
        )
        os.replace(generated, output)
    write_frame_map_data(frame, date, map_output, "historical")
    write_status(date, status)
    print(f"[{date}] wrote {output}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", action="append", help="Generate only this YYYYMMDD date; repeatable.")
    parser.add_argument("--force", action="store_true", help="Replace existing archive images/status files.")
    parser.add_argument("--realtime-only", action="store_true", help="Backfill r60kmV2 only for operational archive dates that have existing R60 feature caches.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.realtime_only:
        dates = sorted(args.date) if args.date else realtime_archive_dates()
        print(f"Backfilling {len(dates)} operational archive cases sequentially", flush=True)
        for date in dates:
            generate_realtime_case(str(date)[:8], force=args.force)
    else:
        dates = sorted(args.date) if args.date else available_dates()
        print(f"Generating {len(dates)} v33 test cases sequentially", flush=True)
        for date in dates:
            generate_case(str(date)[:8], force=args.force)
    rebuild_archive_index()
    print(f"Updated {ARCHIVE_DIR / 'index.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
