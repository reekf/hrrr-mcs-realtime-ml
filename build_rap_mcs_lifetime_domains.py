#!/usr/bin/env python3
"""Run PyFLEXTRKR on cached RAP test cases and build 400-km viewer domains."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import re

import numpy as np

from mcs_lifetime_domains import (
    DEFAULT_BOX_SIZE_KM,
    date8,
    discover_robust_stats,
    domain_mask,
    domain_from_robust_stats,
    save_domains,
)
from pyflextrkr_hrrr import prepare_and_run_pyflextrkr


DEFAULT_PROJECT_DIR = Path("/home/tyreekfrazier/ISU_Research_LOCAL_RUN/fall_2025_ml_proj")
DEFAULT_RAP_DIR = Path("/home/tyreekfrazier/ISU_Research_LOCAL_RUN/RAP_BACKGROUND")
DEFAULT_CASE_ROOT = DEFAULT_PROJECT_DIR / "rap_pyflextrkr_test_cases"
DEFAULT_DOMAIN_JSON = Path(__file__).resolve().with_name("mcs_lifetime_domains_400km.json")
DEFAULT_VIEWER_PARQUET = DEFAULT_PROJECT_DIR / "df_pp_viewer_with_wpc_ero_day1.parquet"
DEFAULT_CONUS_EXTENT = (-125.0, -66.0, 24.0, 50.0)
RAP_AWS_BASE = "https://noaa-rap-pds.s3.amazonaws.com"


def _viewer_dates(path: Path, years: set[str]) -> list[str]:
    import pandas as pd

    frame = pd.read_parquet(path, columns=["Date"])
    dates = frame["Date"].astype(str).str.replace(r"\D", "", regex=True).str[:8]
    return sorted(value for value in dates.unique() if value[:4] in years)


def _wpc_anchors(path: Path, dates: list[str]) -> dict[str, tuple[float, float, str]]:
    import pandas as pd

    frame = pd.read_parquet(path, columns=["Date", "Lat", "Lon", "WPC_ERO_Risk"])
    frame["Date"] = frame["Date"].astype(str).str.replace(r"\D", "", regex=True).str[:8]
    frame = frame[frame["Date"].isin(dates)].copy()
    anchors = {}
    for case_date, group in frame.groupby("Date"):
        risk = pd.to_numeric(group["WPC_ERO_Risk"], errors="coerce")
        maximum = float(risk.max())
        top = group[np.isclose(risk, maximum) & (risk > 0)]
        if top.empty:
            anchors[str(case_date)] = (
                0.5 * (float(group["Lat"].min()) + float(group["Lat"].max())),
                0.5 * (float(group["Lon"].min()) + float(group["Lon"].max())),
                "viewer-grid center fallback because WPC ERO had no positive risk",
            )
        else:
            anchors[str(case_date)] = (
                float(top["Lat"].mean()),
                float(top["Lon"].mean()),
                "centroid of highest WPC ERO risk category",
            )
    return anchors


def _complete_grib(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < 20:
        return False
    with path.open("rb") as handle:
        if handle.read(4) != b"GRIB":
            return False
        handle.seek(-4, 2)
        return handle.read(4) == b"7777"


def _download_hourly_subset(session, date: str, cycle: str, fhr: int, output: Path, force: bool) -> Path:
    if _complete_grib(output) and not force:
        return output
    filename = f"rap.t{int(cycle):02d}z.awp130pgrbf{int(fhr):02d}.grib2"
    data_url = f"{RAP_AWS_BASE}/rap.{date}/{filename}"
    index_response = session.get(data_url + ".idx", timeout=120)
    index_response.raise_for_status()
    entries = []
    for line in index_response.text.splitlines():
        parts = line.split(":")
        if len(parts) >= 5:
            entries.append((int(parts[1]), parts[3], parts[4], line))
    selected = []
    for variable, level in [("REFC", "entire atmosphere"), ("SBT124", "top of atmosphere")]:
        match = next(
            (entry for entry in entries if entry[1].upper() == variable and level in entry[2].lower()),
            None,
        )
        if match is None:
            raise RuntimeError(f"{variable}:{level} absent from {data_url}.idx")
        selected.append(match)
    chunks = []
    for entry in selected:
        index = entries.index(entry)
        if index + 1 < len(entries):
            end = entries[index + 1][0] - 1
        else:
            head = session.head(data_url, timeout=120)
            head.raise_for_status()
            end = int(head.headers["content-length"]) - 1
        response = session.get(data_url, headers={"Range": f"bytes={entry[0]}-{end}"}, timeout=120)
        if response.status_code not in (200, 206):
            raise RuntimeError(f"RAP archive returned {response.status_code} for {entry[3]}")
        chunk = response.content
        if response.status_code == 200 and len(chunk) > end - entry[0] + 1:
            chunk = chunk[entry[0]:end + 1]
        if chunk[:4] != b"GRIB":
            raise RuntimeError(f"RAP byte range did not begin with GRIB: {entry[3]}")
        chunks.append(chunk)
    output.parent.mkdir(parents=True, exist_ok=True)
    part = output.with_suffix(output.suffix + ".part")
    with part.open("wb") as handle:
        for chunk in chunks:
            handle.write(chunk)
    part.replace(output)
    return output


def _rap_path_for_fhr(args, session, date: str, fhr: int) -> Path:
    valid_offset = int(fhr) - 3
    pattern = f"rap_130_{date}_{args.rap_cycle}z_valid{valid_offset:02d}_f{int(fhr):03d}.grib2"
    matches = sorted(args.rap_dir.glob(pattern))
    if not matches:
        output = (
            args.case_root / f"{date}_12z" / "rap_grib_subsets"
            / f"rap_{date}_t{int(args.rap_cycle):02d}z_f{int(fhr):02d}_REFC_SBT124.grib2"
        )
        return _download_hourly_subset(session, date, args.rap_cycle, fhr, output, args.force_download)
    if len(matches) != 1:
        raise RuntimeError(f"Ambiguous cached RAP files for {date} f{fhr:03d}: {matches}")
    return matches[0]


def _select_message(grib, short_names: tuple[str, ...]):
    wanted = {value.lower() for value in short_names}
    for message in grib:
        candidates = {
            str(getattr(message, "shortName", "")).lower(),
            str(getattr(message, "name", "")).lower(),
            str(getattr(message, "parameterName", "")).lower(),
        }
        if candidates & wanted:
            return message
    raise RuntimeError(f"None of {short_names} found in {grib.name}")


def _read_rap_fields(path: Path, extent: tuple[float, float, float, float]):
    import pygrib

    grib = pygrib.open(str(path))
    try:
        ref_message = _select_message(grib, ("refc", "Maximum/Composite radar reflectivity"))
        refc = np.asarray(ref_message.values, dtype=float)
        lat, lon = ref_message.latlons()
    finally:
        grib.close()
    grib = pygrib.open(str(path))
    try:
        # RAP's SBT123 is GOES-12 channel 3 (water vapor) and is too cold for
        # the 241-K cloud-shield criterion.  PyFLEXTRKR needs the longwave-IR
        # window channel: RAP SBT124 / GOES-12 channel 4.
        bt_message = _select_message(
            grib,
            ("sbt124", "Simulated Brightness Temperature for GOES 12, Channel 4"),
        )
        bt = np.asarray(bt_message.values, dtype=float)
    finally:
        grib.close()

    lat = np.asarray(lat, dtype=float)
    lon = ((np.asarray(lon, dtype=float) + 180.0) % 360.0) - 180.0
    lon_min, lon_max, lat_min, lat_max = map(float, extent)
    inside = (
        np.isfinite(lat)
        & np.isfinite(lon)
        & (lon >= lon_min)
        & (lon <= lon_max)
        & (lat >= lat_min)
        & (lat <= lat_max)
    )
    rows, cols = np.where(inside)
    if rows.size == 0:
        raise RuntimeError(f"No RAP grid cells inside {extent} for {path}")
    row_slice = slice(int(rows.min()), int(rows.max()) + 1)
    col_slice = slice(int(cols.min()), int(cols.max()) + 1)
    sub_inside = inside[row_slice, col_slice]
    sub_lat = np.where(sub_inside, lat[row_slice, col_slice], np.nan)
    sub_lon = np.where(sub_inside, lon[row_slice, col_slice], np.nan)
    sub_refc = np.where(sub_inside, refc[row_slice, col_slice], np.nan)
    sub_bt = np.where(sub_inside, bt[row_slice, col_slice], np.nan)
    return sub_bt, sub_refc, sub_lat, sub_lon


def _run_case(args, case_date: str) -> Path:
    import requests

    ir_by_offset = {}
    refc_by_offset = {}
    lat = lon = None
    session = requests.Session()
    session.headers.update({"User-Agent": "xgbffp-rap-pyflextrkr-domain/1.0"})
    for fhr in args.forecast_hours:
        offset = int(fhr) - 3
        path = _rap_path_for_fhr(args, session, case_date, fhr)
        bt, refc, field_lat, field_lon = _read_rap_fields(path, tuple(args.tracking_extent))
        if lat is None:
            lat, lon = field_lat, field_lon
        elif bt.shape != lat.shape:
            raise RuntimeError(f"RAP grid shape changed within {case_date}: {bt.shape} vs {lat.shape}")
        ir_by_offset[int(offset)] = bt
        refc_by_offset[int(offset)] = refc

    case_dir = args.case_root / f"{case_date}_12z"
    result = prepare_and_run_pyflextrkr(
        ir_by_offset,
        refc_by_offset,
        lat,
        lon,
        run_date=case_date,
        cycle="12",
        case_dir=case_dir,
        extent=tuple(args.tracking_extent),
        bt_threshold_k=args.bt_threshold_k,
        cloud_area_threshold_km2=args.cloud_area_threshold_km2,
        precipitation_threshold_dbz=args.precipitation_threshold_dbz,
        precipitation_major_axis_threshold_km=args.precipitation_major_axis_km,
        convective_threshold_dbz=args.convective_threshold_dbz,
        cloud_duration_hours=args.cloud_duration_hours,
        structural_duration_hours=args.structural_duration_hours,
        overlap_threshold=args.overlap_threshold,
        cell_area_km2=args.rap_cell_area_km2,
        force=args.force,
        source_model="RAP",
        ir_required=True,
        ir_field_name="RAP SBT124 (GOES-12 channel 4 longwave IR)",
    )
    stats_path = Path(result.robust_stats_path) if result.robust_stats_path else discover_robust_stats(args.case_root, case_date)
    print(
        f"{case_date}: detected={result.detected} robust={result.max_joint_duration_hours}h "
        f"stats={stats_path}"
    )
    return stats_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dates", nargs="*", help="YYYYMMDD cases; default reads the viewer parquet")
    parser.add_argument("--years", nargs="+", default=["2024", "2025"])
    parser.add_argument("--viewer-parquet", type=Path, default=DEFAULT_VIEWER_PARQUET)
    parser.add_argument("--rap-dir", type=Path, default=DEFAULT_RAP_DIR)
    parser.add_argument("--rap-cycle", default="09")
    parser.add_argument("--forecast-hours", nargs="+", type=int, default=list(range(3, 28)))
    parser.add_argument("--case-root", type=Path, default=DEFAULT_CASE_ROOT)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_DOMAIN_JSON)
    parser.add_argument("--box-size-km", type=float, default=DEFAULT_BOX_SIZE_KM)
    parser.add_argument("--tracking-extent", nargs=4, type=float, default=DEFAULT_CONUS_EXTENT)
    parser.add_argument("--rap-cell-area-km2", type=float, default=169.0)
    parser.add_argument("--bt-threshold-k", type=float, default=241.0)
    parser.add_argument("--cloud-area-threshold-km2", type=float, default=40000.0)
    parser.add_argument("--precipitation-threshold-dbz", type=float, default=25.0)
    parser.add_argument("--precipitation-major-axis-km", type=float, default=100.0)
    parser.add_argument("--convective-threshold-dbz", type=float, default=45.0)
    parser.add_argument("--cloud-duration-hours", type=int, default=3)
    parser.add_argument("--structural-duration-hours", type=int, default=4)
    parser.add_argument("--overlap-threshold", type=float, default=0.5)
    parser.add_argument("--stats-only", action="store_true", help="Only rebuild JSON from existing robust stats")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--workers", type=int, default=1, help="Independent cases to process concurrently")
    return parser.parse_args()


def _process_case(args, case_date: str, anchor: tuple[float, float, str]):
    stats_path = discover_robust_stats(args.case_root, case_date) if args.stats_only else _run_case(args, case_date)
    anchor_lat, anchor_lon, anchor_name = anchor
    domain = domain_from_robust_stats(
        stats_path,
        date=case_date,
        box_size_km=args.box_size_km,
        anchor_lat=anchor_lat,
        anchor_lon=anchor_lon,
        anchor_name=anchor_name,
    )
    return domain


def _validate_viewer_grid_coverage(viewer_path: Path, domains):
    """Reject boxes the existing historical ML grid cannot cover completely."""
    import pandas as pd

    frame = pd.read_parquet(viewer_path, columns=["Date", "Lat", "Lon"])
    frame["Date"] = frame["Date"].astype(str).str.replace(r"\D", "", regex=True).str[:8]
    accepted = []
    excluded = []
    for domain in domains:
        grid = frame[frame["Date"] == domain.date]
        if grid.empty:
            excluded.append({"date": domain.date, "reason": "no rows in the historical ML viewer grid"})
            continue
        bounds = {
            "lon_min": float(grid["Lon"].min()),
            "lon_max": float(grid["Lon"].max()),
            "lat_min": float(grid["Lat"].min()),
            "lat_max": float(grid["Lat"].max()),
        }
        complete = (
            domain.lon_min >= bounds["lon_min"]
            and domain.lon_max <= bounds["lon_max"]
            and domain.lat_min >= bounds["lat_min"]
            and domain.lat_max <= bounds["lat_max"]
        )
        rows = int(
            domain_mask(
                grid["Lat"].to_numpy(float),
                grid["Lon"].to_numpy(float),
                domain,
            ).sum()
        )
        if not complete:
            excluded.append(
                {
                    "date": domain.date,
                    "reason": "400-km MCS box extends beyond the available historical ML grid",
                    "viewer_rows_inside_partial_box": rows,
                    "domain_extent": domain.extent,
                    "viewer_grid_bounds": [
                        bounds["lon_min"], bounds["lon_max"], bounds["lat_min"], bounds["lat_max"]
                    ],
                    "selected_track_number": domain.track_number,
                    "selected_center": [domain.center_lat, domain.center_lon],
                }
            )
        elif rows == 0:
            excluded.append({"date": domain.date, "reason": "400-km MCS box contains no historical ML rows"})
        else:
            accepted.append(domain)
    return accepted, excluded


def main() -> int:
    args = parse_args()
    years = {str(value) for value in args.years}
    dates = sorted({date8(value) for value in args.dates}) if args.dates else _viewer_dates(args.viewer_parquet, years)
    wpc_anchors = _wpc_anchors(args.viewer_parquet, dates)
    domains = []
    failures = []
    if int(args.workers) > 1:
        with ProcessPoolExecutor(max_workers=int(args.workers)) as executor:
            future_dates = {
                executor.submit(_process_case, args, case_date, wpc_anchors[case_date]): case_date
                for case_date in dates
            }
            for future in as_completed(future_dates):
                case_date = future_dates[future]
                try:
                    domain = future.result()
                    domains.append(domain)
                    print(
                        f"{case_date}: track={domain.track_number} duration={domain.mcs_duration_hours:g}h "
                        f"center=({domain.center_lat:.3f}, {domain.center_lon:.3f})"
                    )
                except Exception as exc:
                    failures.append((case_date, repr(exc)))
                    print(f"ERROR {case_date}: {exc}")
    else:
        for case_date in dates:
            try:
                domain = _process_case(args, case_date, wpc_anchors[case_date])
                domains.append(domain)
                print(
                    f"{case_date}: track={domain.track_number} duration={domain.mcs_duration_hours:g}h "
                    f"center=({domain.center_lat:.3f}, {domain.center_lon:.3f})"
                )
            except Exception as exc:
                failures.append((case_date, repr(exc)))
                print(f"ERROR {case_date}: {exc}")
    domains.sort(key=lambda domain: domain.date)
    domains, coverage_exclusions = _validate_viewer_grid_coverage(args.viewer_parquet, domains)
    save_domains(domains, args.output_json, excluded_cases=coverage_exclusions)
    print(f"Saved {len(domains)}/{len(dates)} complete case domains: {args.output_json}")
    for item in coverage_exclusions:
        print(f"EXCLUDED {item['date']}: {item['reason']}")
    if failures:
        for case_date, error in failures:
            print(f"FAILED {case_date}: {error}")
        return 2
    if coverage_exclusions:
        print(
            "The excluded cases require ML features/predictions on a broader source grid; "
            "they are not scored on a silently clipped box."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
