#!/usr/bin/env python3
"""
Rebuild/repair the WPC Day-1 ERO risk grid used by the v33 stats notebook.

This script updates the actual v33 historical/test-set WPC/PP grid:

  /home/tyreekfrazier/ISU_Research_LOCAL_RUN/fall_2025_ml_proj/df_pp_viewer_with_wpc_ero_day1.parquet

It fetches the default IEM WPC/SPC outlook shapefile for each date, selects the
Day-1 ERO whose valid period matches date 12Z -> next-day 12Z, rasterizes the
polygons to the existing Lat/Lon grid, and enforces highest-risk-wins.

Important:
  * Uses the default IEM request only. Do not pass geom=layers/cookie/etc.
  * Does not change PP/proxy truth fields.
  * Does not recompute metrics; it only repairs WPC_ERO_Risk and metadata.
  * Dry-run by default. Pass --write to write the patched parquet.
  * Will not write if unexpected date errors occur unless --write-with-errors is passed.

Examples:
  python rebuild_v33_wpc_ero_grid_from_iem.py --dates 20240620 20240621
  python rebuild_v33_wpc_ero_grid_from_iem.py --all-dates --write
  python rebuild_v33_wpc_ero_grid_from_iem.py --years 2024 2025 --write
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests


PROJECT_DIR_DEFAULT = Path("/home/tyreekfrazier/ISU_Research_LOCAL_RUN/fall_2025_ml_proj")
PP_WPC_GRID_DEFAULT = PROJECT_DIR_DEFAULT / "df_pp_viewer_with_wpc_ero_day1.parquet"
IEM_OUTLOOK_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/gis/outlooks.py"

RISK_TO_LABEL = {
    0.00: "None",
    0.05: "Marginal",
    0.15: "Slight",
    0.40: "Moderate",
    0.70: "High",
}


def log(msg: str) -> None:
    print(msg, flush=True)


def date8(x: str) -> str:
    m = re.search(r"(20\d{6})", str(x))
    if not m:
        raise ValueError(f"Could not parse YYYYMMDD from {x!r}")
    return m.group(1)


def risk_label_from_value(v: float) -> str:
    v = float(v) if np.isfinite(v) else 0.0
    if v >= 0.70:
        return "High"
    if v >= 0.40:
        return "Moderate"
    if v >= 0.15:
        return "Slight"
    if v >= 0.05:
        return "Marginal"
    return "None"


def risk_from_row(row: pd.Series) -> float:
    """Robust WPC ERO risk parser for IEM/WPC shapefile rows."""

    vals = []
    for col in row.index:
        if str(col).lower() == "geometry":
            continue
        val = row.get(col, "")
        if val is not None:
            vals.append(str(val))

    text = " ".join(vals).upper()

    # Text encodings.
    if re.search(r"\bHIGH\b|\bHIGH_RISK\b|\bHIGH RISK\b", text):
        return 0.70
    if re.search(r"\bMDT\b|\bMOD\b|\bMODERATE\b|\bMODERATE_RISK\b|\bMODERATE RISK\b", text):
        return 0.40
    if re.search(r"\bSLGT\b|\bSLIGHT\b|\bSLIGHT_RISK\b|\bSLIGHT RISK\b", text):
        return 0.15
    if re.search(r"\bMRGL\b|\bMARG\b|\bMARGINAL\b|\bMARGINAL_RISK\b|\bMARGINAL RISK\b", text):
        return 0.05

    # Numeric encodings, including the IEM "risk" field if present.
    numeric_vals = []
    for col in row.index:
        if str(col).lower() == "geometry":
            continue
        v = pd.to_numeric(pd.Series([row.get(col)]), errors="coerce").iloc[0]
        if pd.notna(v):
            numeric_vals.append(float(v))

    # Direct probabilities or percent values.
    for v in numeric_vals:
        if np.isclose(v, 0.70) or np.isclose(v, 70):
            return 0.70
        if np.isclose(v, 0.40) or np.isclose(v, 40):
            return 0.40
        if np.isclose(v, 0.15) or np.isclose(v, 15):
            return 0.15
        if np.isclose(v, 0.05) or np.isclose(v, 5):
            return 0.05

    # Possible ordinal encoding: 1=MRGL, 2=SLGT, 3=MDT, 4=HIGH.
    # Only reaches here if text/direct probability encodings were absent.
    for v in numeric_vals:
        if np.isclose(v, 4):
            return 0.70
        if np.isclose(v, 3):
            return 0.40
        if np.isclose(v, 2):
            return 0.15
        if np.isclose(v, 1):
            return 0.05

    return 0.0


def parse_datetime_col(gdf: pd.DataFrame, names: Iterable[str]) -> tuple[pd.Series, str | None]:
    upper_to_col = {str(c).upper().strip(): c for c in gdf.columns}
    for name in names:
        key = str(name).upper().strip()
        if key in upper_to_col:
            col = upper_to_col[key]
            s = pd.to_datetime(gdf[col], errors="coerce", utc=True)
            if s.notna().any():
                return s, str(col)
    return pd.Series(pd.NaT, index=gdf.index, dtype="datetime64[ns, UTC]"), None


@dataclass
class SourceGDFResult:
    gdf: object
    target_start: pd.Timestamp
    target_end: pd.Timestamp
    selected_valid_start: str
    selected_valid_end: str
    selected_prodiss: str
    selected_reason: str
    source_max: float
    source_counts: dict[str, int]
    source_rows: int


def fetch_default_iem_wpc_gdf(date: str, download_dir: Path) -> SourceGDFResult:
    """Fetch default IEM WPC Day-1 ERO polygons and select the case valid window."""

    try:
        import geopandas as gpd
    except Exception as exc:
        raise RuntimeError("geopandas is required for WPC shapefile ingest") from exc

    d = date8(date)
    start_dt = datetime.strptime(d, "%Y%m%d")

    # The query window is deliberately wider than the target valid window.
    # We select the exact 12Z-to-12Z product from the returned DBF times.
    sts = (start_dt - timedelta(hours=6)).strftime("%Y-%m-%dT%H:%MZ")
    ets = (start_dt + timedelta(days=1, hours=18)).strftime("%Y-%m-%dT%H:%MZ")

    params = {"type": "E", "d": "1", "sts": sts, "ets": ets}

    download_dir.mkdir(parents=True, exist_ok=True)
    zip_path = download_dir / f"iem_wpc_ero_day1_{d}_default_valid12to12.zip"

    if zip_path.exists() and zip_path.stat().st_size > 512:
        content = zip_path.read_bytes()
    else:
        resp = requests.get(IEM_OUTLOOK_URL, params=params, timeout=60)
        if resp.status_code != 200:
            raise RuntimeError(f"IEM HTTP {resp.status_code} for {d}: {resp.text[:200]}")
        content = resp.content
        if not content.startswith(b"PK"):
            raise RuntimeError(f"IEM response for {d} was not a zip/shapefile; first bytes={content[:80]!r}")
        zip_path.write_bytes(content)

    try:
        gdf = gpd.read_file(f"zip://{zip_path}")
    except Exception:
        gdf = gpd.read_file(str(zip_path))

    if gdf is None or gdf.empty:
        raise RuntimeError(f"IEM shapefile empty for {d}")

    upper_to_col = {str(c).upper().strip(): c for c in gdf.columns}

    # Keep ERO type rows if schema has TYPE.
    if "TYPE" in upper_to_col:
        c = upper_to_col["TYPE"]
        gdf = gdf[gdf[c].astype(str).str.upper().str.contains("E", na=False)].copy()

    # Keep Day 1 rows if schema has DAY.
    if "DAY" in upper_to_col:
        c = upper_to_col["DAY"]
        gdf = gdf[pd.to_numeric(gdf[c], errors="coerce").fillna(-999).astype(int) == 1].copy()

    if gdf.empty:
        raise RuntimeError(f"No ERO Day-1 rows after TYPE/DAY filtering for {d}")

    target_start = pd.Timestamp(datetime.strptime(d, "%Y%m%d"), tz="UTC") + pd.Timedelta(hours=12)
    target_end = target_start + pd.Timedelta(days=1)

    issue, issue_col = parse_datetime_col(gdf, ["ISSUE", "VALID", "START", "BEGINTIME"])
    expire, expire_col = parse_datetime_col(gdf, ["EXPIRE", "EXPIRATION", "END", "ENDTIME"])
    prodiss, prodiss_col = parse_datetime_col(gdf, ["PRODISS", "PRODUCTISSUANCE", "ISSUANCE"])

    if issue.notna().any() and expire.notna().any():
        tol = pd.Timedelta(minutes=2)
        exact = (issue.sub(target_start).abs() <= tol) & (expire.sub(target_end).abs() <= tol)

        if exact.any():
            sel = exact.copy()
            reason = "exact_12z_to_12z"
        else:
            candidates = (issue < target_end) & (expire > target_start)
            if not candidates.any():
                raise RuntimeError(f"No WPC ERO valid-period rows overlap {target_start} to {target_end} for {d}")

            overlap_start = pd.Series(
                np.maximum(issue[candidates].astype("int64"), target_start.value),
                index=issue[candidates].index,
            )
            overlap_end = pd.Series(
                np.minimum(expire[candidates].astype("int64"), target_end.value),
                index=expire[candidates].index,
            )
            overlap_hours = (overlap_end - overlap_start) / 1e9 / 3600.0
            start_offset_hours = (issue[candidates] - target_start).abs() / pd.Timedelta(hours=1)
            score = overlap_hours - 0.25 * start_offset_hours
            best_index = score.idxmax()
            best_issue = issue.loc[best_index]
            best_expire = expire.loc[best_index]
            sel = (issue == best_issue) & (expire == best_expire)
            reason = "best_overlap"

        if prodiss.notna().any() and sel.any():
            latest_prod = prodiss[sel].max()
            sel = sel & (prodiss == latest_prod)
        else:
            latest_prod = pd.NaT

        gdf = gdf.loc[sel].copy()
        selected_valid_start = str(issue.loc[gdf.index].min()) if not gdf.empty else ""
        selected_valid_end = str(expire.loc[gdf.index].max()) if not gdf.empty else ""
        selected_prodiss = str(latest_prod) if pd.notna(latest_prod) else ""

    elif prodiss.notna().any():
        latest_prod = prodiss.max()
        gdf = gdf.loc[prodiss == latest_prod].copy()
        selected_valid_start = ""
        selected_valid_end = ""
        selected_prodiss = str(latest_prod)
        reason = "latest_prodiss_no_valid_window"
    else:
        selected_valid_start = ""
        selected_valid_end = ""
        selected_prodiss = ""
        reason = "unfiltered_no_valid_window"

    if gdf.empty:
        raise RuntimeError(f"No WPC ERO rows selected for target valid window {target_start} to {target_end} for {d}")

    if getattr(gdf, "crs", None) is not None:
        gdf = gdf.to_crs(epsg=4326)

    gdf["__WPC_ERO_Risk"] = gdf.apply(risk_from_row, axis=1).astype(float)
    gdf = gdf[gdf["__WPC_ERO_Risk"] > 0].copy()

    if gdf.empty:
        raise RuntimeError(f"WPC ERO polygons existed, but no risk categories were recognized for {d}")

    source_counts_raw = gdf["__WPC_ERO_Risk"].value_counts().sort_index().to_dict()
    source_counts = {str(k): int(v) for k, v in source_counts_raw.items()}
    source_max = float(gdf["__WPC_ERO_Risk"].max())

    return SourceGDFResult(
        gdf=gdf,
        target_start=target_start,
        target_end=target_end,
        selected_valid_start=selected_valid_start,
        selected_valid_end=selected_valid_end,
        selected_prodiss=selected_prodiss,
        selected_reason=reason,
        source_max=source_max,
        source_counts=source_counts,
        source_rows=int(len(gdf)),
    )


def inside_mask_for_geom(geom, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Vectorized point-in-polygon with a robust fallback."""

    # Include boundaries by applying an extremely tiny degree-buffer before contains_xy.
    # This avoids dropping grid points that lie exactly on polygon boundaries.
    try:
        geom_for_test = geom.buffer(1.0e-10)
    except Exception:
        geom_for_test = geom

    try:
        import shapely
        contains_xy = getattr(shapely, "contains_xy", None)
        if contains_xy is not None:
            return np.asarray(contains_xy(geom_for_test, x, y), dtype=bool)
    except Exception:
        pass

    try:
        from shapely import vectorized
        return np.asarray(vectorized.contains(geom_for_test, x, y), dtype=bool)
    except Exception:
        pass

    from shapely.geometry import Point
    from shapely.prepared import prep

    pg = prep(geom_for_test)
    return np.asarray([pg.covers(Point(float(xx), float(yy))) for xx, yy in zip(x, y)], dtype=bool)


def rasterize_wpc_to_existing_points(src: SourceGDFResult, lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rasterize WPC polygons to point grid. Highest risk wins."""

    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)

    out_risk = np.zeros(len(lon), dtype=np.float32)
    out_label = np.full(len(lon), "None", dtype=object)

    finite = np.isfinite(lon) & np.isfinite(lat)

    # Sort ascending so higher polygons overwrite lower ones via >= comparison.
    gdf = src.gdf.sort_values("__WPC_ERO_Risk").copy()

    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        risk = float(row["__WPC_ERO_Risk"])
        label = risk_label_from_value(risk)

        minx, miny, maxx, maxy = geom.bounds
        bbox = finite & (lon >= minx) & (lon <= maxx) & (lat >= miny) & (lat <= maxy)
        idx = np.flatnonzero(bbox)
        if idx.size == 0:
            continue

        inside = inside_mask_for_geom(geom, lon[idx], lat[idx])
        if not np.any(inside):
            continue

        hit_idx = idx[inside]
        update = risk >= out_risk[hit_idx]
        if np.any(update):
            hit_idx = hit_idx[update]
            out_risk[hit_idx] = np.maximum(out_risk[hit_idx], risk)
            out_label[hit_idx] = label

    return out_risk, out_label


def count_thresholds(values: np.ndarray) -> dict[str, int]:
    s = np.asarray(values, dtype=float)
    return {
        "n_ge_005": int(np.sum(s >= 0.05)),
        "n_ge_015": int(np.sum(s >= 0.15)),
        "n_ge_040": int(np.sum(s >= 0.40)),
        "n_ge_070": int(np.sum(s >= 0.70)),
    }


def build_zero_wpc_result(d: str, n: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return an explicit no-risk WPC grid for dates with no matching Day-1 ERO rows."""

    target_start = pd.Timestamp(datetime.strptime(d, "%Y%m%d"), tz="UTC") + pd.Timedelta(hours=12)
    target_end = target_start + pd.Timedelta(days=1)
    new_vals = np.zeros(n, dtype=np.float32)
    new_labels = np.full(n, "None", dtype=object)
    meta = {
        "source_rows": 0,
        "source_max": 0.0,
        "source_counts": {},
        "selected_prodiss": "",
        "selected_reason": "no_matching_day1_ero_rows_assumed_no_risk",
        "target_start": target_start,
        "target_end": target_end,
    }
    return new_vals, new_labels, meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(PP_WPC_GRID_DEFAULT), help="Input PP/WPC parquet")
    ap.add_argument("--output", default=None, help="Output parquet. Default: same as input")
    ap.add_argument("--cache-dir", default=None, help="Directory for downloaded IEM zips and audit CSV")
    ap.add_argument("--dates", nargs="*", default=None, help="Specific YYYYMMDD dates to rebuild")
    ap.add_argument("--all-dates", action="store_true", help="Rebuild all dates in the PP/WPC grid")
    ap.add_argument("--years", nargs="*", default=None, help="Restrict all-dates mode to these years, e.g. 2024 2025")
    ap.add_argument("--write", action="store_true", help="Actually write output parquet. Default is dry-run")
    ap.add_argument("--no-backup", action="store_true", help="Do not make a backup when writing in-place")
    ap.add_argument("--stop-on-error", action="store_true", help="Stop at first failed date instead of keeping old WPC values")
    ap.add_argument("--write-with-errors", action="store_true", help="Write output even if some dates had unexpected errors")
    args = ap.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve() if args.output else input_path

    if args.cache_dir is None:
        cache_dir = input_path.parent / "wpc_ero_rebuild_from_iem_audit"
    else:
        cache_dir = Path(args.cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    log(f"Input PP/WPC grid: {input_path}")
    log(f"Output PP/WPC grid: {output_path}")
    log(f"Audit/cache dir:    {cache_dir}")
    log(f"Mode: {'WRITE' if args.write else 'DRY-RUN'}")

    if not input_path.exists():
        raise FileNotFoundError(input_path)

    df = pd.read_parquet(input_path)

    required = ["Date", "Lat", "Lon", "WPC_ERO_Risk"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Input PP/WPC grid missing required columns: {missing}")

    df["Date"] = df["Date"].astype(str).str.slice(0, 8)
    df["Lat"] = pd.to_numeric(df["Lat"], errors="coerce")
    df["Lon"] = pd.to_numeric(df["Lon"], errors="coerce")
    df["WPC_ERO_Risk"] = pd.to_numeric(df["WPC_ERO_Risk"], errors="coerce").fillna(0.0).astype(np.float32)

    if args.dates:
        target_dates = sorted({date8(d) for d in args.dates})
    else:
        # Default to all dates if neither --dates nor --all-dates is passed, but make this explicit in log.
        target_dates = sorted(df["Date"].dropna().astype(str).unique().tolist())

    if args.years:
        years = {str(y) for y in args.years}
        target_dates = [d for d in target_dates if d[:4] in years]

    if not args.all_dates and not args.dates:
        log("No --dates passed; defaulting to all dates in the grid. Use --years to restrict if needed.")

    if not target_dates:
        raise RuntimeError("No dates selected to rebuild.")

    log(f"Selected dates: {len(target_dates)}")
    log(f"First/last selected date: {target_dates[0]} / {target_dates[-1]}")

    # Ensure output columns exist and have assignment-safe dtypes.
    # Important: older/parquet-created columns may use pandas string dtype.
    # Assigning floats into string dtype caused the previous all-date TypeError.
    if "WPC_ERO_Category" not in df.columns:
        df["WPC_ERO_Category"] = "None"
    df["WPC_ERO_Category"] = df["WPC_ERO_Category"].astype("object")

    string_meta_cols = [
        "WPC_ERO_Source_ProdIssue",
        "WPC_ERO_Source_ValidStart",
        "WPC_ERO_Source_ValidEnd",
        "WPC_ERO_Source_SelectedReason",
    ]
    for col in string_meta_cols:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].astype("object")

    numeric_meta_cols = [
        "WPC_ERO_Source_MaxRisk",
        "WPC_ERO_Raster_MaxRisk",
    ]
    for col in numeric_meta_cols:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")

    summary_rows = []
    errors = []

    for n, d in enumerate(target_dates, start=1):
        log("")
        log("=" * 96)
        log(f"[{n}/{len(target_dates)}] Rebuilding WPC ERO grid for {d}")
        log("=" * 96)

        mask = df["Date"] == d
        if not mask.any():
            log(f"  No rows for {d}; skipping")
            continue

        old_vals = df.loc[mask, "WPC_ERO_Risk"].to_numpy(float)
        old_counts = count_thresholds(old_vals)
        old_max = float(np.nanmax(old_vals)) if old_vals.size else 0.0

        try:
            lon = df.loc[mask, "Lon"].to_numpy(float)
            lat = df.loc[mask, "Lat"].to_numpy(float)

            try:
                src = fetch_default_iem_wpc_gdf(d, cache_dir)
                new_vals, new_labels = rasterize_wpc_to_existing_points(src, lon=lon, lat=lat)
                src_meta = {
                    "source_rows": src.source_rows,
                    "source_max": src.source_max,
                    "source_counts": src.source_counts,
                    "selected_prodiss": src.selected_prodiss,
                    "selected_reason": src.selected_reason,
                    "target_start": src.target_start,
                    "target_end": src.target_end,
                }
            except RuntimeError as fetch_exc:
                msg0 = str(fetch_exc)
                if "No WPC ERO valid-period rows overlap" in msg0 or "No WPC ERO rows selected" in msg0:
                    log(f"  No matching Day-1 ERO rows for {d}; treating as explicit no-risk WPC grid.")
                    src = None
                    new_vals, new_labels, src_meta = build_zero_wpc_result(d, int(mask.sum()))
                else:
                    raise

            new_counts = count_thresholds(new_vals)
            new_max = float(np.nanmax(new_vals)) if new_vals.size else 0.0

            changed = np.isfinite(old_vals) & np.isfinite(new_vals) & (np.abs(old_vals - new_vals) > 1.0e-6)
            n_changed = int(np.sum(changed))

            log(f"  Source: rows={src_meta['source_rows']} max={src_meta['source_max']:.2f} counts={src_meta['source_counts']}")
            log(
                f"  Old grid: max={old_max:.2f} "
                f">=5={old_counts['n_ge_005']:,} >=15={old_counts['n_ge_015']:,} "
                f">=40={old_counts['n_ge_040']:,} >=70={old_counts['n_ge_070']:,}"
            )
            log(
                f"  New grid: max={new_max:.2f} "
                f">=5={new_counts['n_ge_005']:,} >=15={new_counts['n_ge_015']:,} "
                f">=40={new_counts['n_ge_040']:,} >=70={new_counts['n_ge_070']:,} "
                f"changed_points={n_changed:,}"
            )

            warning = ""
            if float(src_meta["source_max"]) > new_max + 1.0e-6:
                warning = (
                    f"source max {float(src_meta['source_max']):.2f} exceeds raster max {new_max:.2f}; "
                    "higher-risk polygon may be outside grid/domain or rasterization needs inspection"
                )
                log("  WARNING: " + warning)

            if args.write:
                df.loc[mask, "WPC_ERO_Risk"] = new_vals.astype(np.float32)
                df.loc[mask, "WPC_ERO_Category"] = new_labels
                df.loc[mask, "WPC_ERO_Source_ProdIssue"] = str(src_meta["selected_prodiss"])
                df.loc[mask, "WPC_ERO_Source_ValidStart"] = str(src_meta["target_start"].strftime("%Y-%m-%dT%H:%M:%SZ"))
                df.loc[mask, "WPC_ERO_Source_ValidEnd"] = str(src_meta["target_end"].strftime("%Y-%m-%dT%H:%M:%SZ"))
                df.loc[mask, "WPC_ERO_Source_SelectedReason"] = str(src_meta["selected_reason"])
                df.loc[mask, "WPC_ERO_Source_MaxRisk"] = float(src_meta["source_max"])
                df.loc[mask, "WPC_ERO_Raster_MaxRisk"] = float(new_max)

            summary_rows.append({
                "Date": d,
                "Status": "ok",
                "Rows": int(mask.sum()),
                "SourceRows": int(src_meta["source_rows"]),
                "SourceMaxRisk": float(src_meta["source_max"]),
                "OldMaxRisk": float(old_max),
                "NewMaxRisk": float(new_max),
                "Old_N_GE_005": old_counts["n_ge_005"],
                "Old_N_GE_015": old_counts["n_ge_015"],
                "Old_N_GE_040": old_counts["n_ge_040"],
                "Old_N_GE_070": old_counts["n_ge_070"],
                "New_N_GE_005": new_counts["n_ge_005"],
                "New_N_GE_015": new_counts["n_ge_015"],
                "New_N_GE_040": new_counts["n_ge_040"],
                "New_N_GE_070": new_counts["n_ge_070"],
                "ChangedPoints": n_changed,
                "SelectedProdIssue": str(src_meta["selected_prodiss"]),
                "SelectedReason": str(src_meta["selected_reason"]),
                "Warning": warning,
                "Error": "",
            })

        except Exception as exc:
            msg = repr(exc)
            log(f"  ERROR for {d}: {msg}")
            errors.append((d, msg))
            summary_rows.append({
                "Date": d,
                "Status": "error",
                "Rows": int(mask.sum()),
                "SourceRows": 0,
                "SourceMaxRisk": np.nan,
                "OldMaxRisk": float(old_max),
                "NewMaxRisk": np.nan,
                "Old_N_GE_005": old_counts["n_ge_005"],
                "Old_N_GE_015": old_counts["n_ge_015"],
                "Old_N_GE_040": old_counts["n_ge_040"],
                "Old_N_GE_070": old_counts["n_ge_070"],
                "New_N_GE_005": np.nan,
                "New_N_GE_015": np.nan,
                "New_N_GE_040": np.nan,
                "New_N_GE_070": np.nan,
                "ChangedPoints": np.nan,
                "SelectedProdIssue": "",
                "SelectedReason": "",
                "Warning": "",
                "Error": msg,
            })
            if args.stop_on_error:
                raise

    summary = pd.DataFrame(summary_rows)
    summary_csv = cache_dir / f"v33_wpc_ero_rebuild_summary_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.csv"
    summary.to_csv(summary_csv, index=False)

    latest_csv = cache_dir / "v33_wpc_ero_rebuild_summary_latest.csv"
    summary.to_csv(latest_csv, index=False)

    log("")
    log("=" * 96)
    log("REBUILD SUMMARY")
    log("=" * 96)
    if not summary.empty:
        show_cols = [
            "Date", "Status", "SourceMaxRisk", "OldMaxRisk", "NewMaxRisk",
            "Old_N_GE_040", "New_N_GE_040", "ChangedPoints", "Warning", "Error"
        ]
        with pd.option_context("display.max_colwidth", 120):
            print(summary[show_cols].to_string(index=False))
    log(f"Wrote summary CSV: {summary_csv}")
    log(f"Wrote latest summary CSV: {latest_csv}")

    changed_dates = []
    if not summary.empty and "ChangedPoints" in summary.columns:
        tmp = summary[pd.to_numeric(summary["ChangedPoints"], errors="coerce").fillna(0) > 0]
        changed_dates = tmp["Date"].astype(str).tolist()

    log("")
    log(f"Dates with changed WPC grid values: {len(changed_dates)}")
    if changed_dates:
        log("  " + " ".join(changed_dates))

    if errors:
        log("")
        log(f"Dates with errors: {len(errors)}")
        for d, msg in errors:
            log(f"  {d}: {msg}")

    if args.write and errors and not args.write_with_errors:
        log("")
        log("Unexpected errors occurred. Not writing parquet. Re-run with --stop-on-error to debug, or --write-with-errors if you intentionally want partial output.")
        return 2

    if args.write:
        if output_path == input_path and not args.no_backup:
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            backup = input_path.with_suffix(f".backup_before_wpc_rebuild_{ts}.parquet")
            log(f"Writing backup: {backup}")
            shutil.copy2(input_path, backup)

        tmp_output = output_path.with_suffix(output_path.suffix + ".tmp")
        log(f"Writing rebuilt PP/WPC grid: {output_path}")
        df.to_parquet(tmp_output, index=False)
        tmp_output.replace(output_path)
        log("Write complete.")
    else:
        log("")
        log("Dry-run only. No parquet was written. Re-run with --write to update the grid.")

    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
