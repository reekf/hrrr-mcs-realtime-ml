#!/usr/bin/env python3
"""Build sanitized browser map data for one v33 forecast date.

The output contains only public forecast/verification fields: coordinates,
probabilities, categorical contour lines, and valid-period metadata.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_DIR = Path("/home/tyreekfrazier/ISU_Research_LOCAL_RUN/fall_2025_ml_proj")
REALTIME_DIR = PROJECT_DIR / "v33_realtime_radiusstats_forecasts" / "verified"
REALTIME_WPC_DIR = PROJECT_DIR / "realtime_wpc_ero_cache_v33"
HISTORICAL_GRID = PROJECT_DIR / "df_pp_viewer_with_wpc_ero_day1.parquet"
HISTORICAL_PREDICTIONS = PROJECT_DIR / "v33_singletarget_radius_sensitivity_viewer_prediction_cache"
RADII = (40, 60, 75, 100)
THRESHOLDS = (0.05, 0.15, 0.40, 0.70)

LAYER_SPECS = {
    "ml_r40": ("ML r40 km", "ML_r40_Prob", "forecast"),
    "ml_r60": ("ML r60 km", "ML_r60_Prob", "forecast"),
    "ml_r75": ("ML r75 km", "ML_r75_Prob", "forecast"),
    "ml_r100": ("ML r100 km", "ML_r100_Prob", "forecast"),
    "wpc": ("WPC ERO", "WPC_ERO_Risk", "reference"),
    "pp": ("Practically Perfect", "PP_Any flood proxy", "verification"),
}


def date8(value: str) -> str:
    text = "".join(ch for ch in str(value) if ch.isdigit())
    if len(text) < 8:
        raise ValueError(f"Expected YYYYMMDD date, got {value!r}")
    return text[:8]


def _read_date(path: Path, date: str, columns: list[str] | None = None) -> pd.DataFrame:
    kwargs = {"filters": [("Date", "==", date)]}
    if columns is not None:
        kwargs["columns"] = columns
    frame = pd.read_parquet(path, **kwargs)
    if frame.empty:
        # Some older parquet files store Date with a non-string dtype/filter encoding.
        frame = pd.read_parquet(path, columns=columns)
        frame = frame[frame["Date"].astype(str).str[:8] == date].copy()
    return frame


def _sort_grid(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["Date"] = out["Date"].astype(str).str[:8]
    return out.sort_values(["Lat", "Lon"]).reset_index(drop=True)


def _merge_aligned(base: pd.DataFrame, extra: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    left = _sort_grid(base)
    right = _sort_grid(extra)
    keys = ["Date", "Lat", "Lon"]
    if len(left) != len(right) or not left[keys].equals(right[keys]):
        raise RuntimeError("Map-data source grids do not align on Date/Lat/Lon")
    for column in columns:
        left[column] = right[column].to_numpy()
    return left


def load_historical(date: str) -> pd.DataFrame:
    base = _sort_grid(
        _read_date(
            HISTORICAL_GRID,
            date,
            ["Date", "Lat", "Lon", "WPC_ERO_Risk", "PP_Any flood proxy"],
        )
    )
    if base.empty:
        raise RuntimeError(f"No historical WPC/verification grid for {date}")
    for radius in RADII:
        path = HISTORICAL_PREDICTIONS / f"v33_singletarget_radius_sensitivity_predictions_r{radius}km.parquet"
        pred = _read_date(path, date, ["Date", "Lat", "Lon", "ML_Forecast_Prob"])
        pred = pred.rename(columns={"ML_Forecast_Prob": f"ML_r{radius}_Prob"})
        base = _merge_aligned(base, pred, [f"ML_r{radius}_Prob"])
    return base


def _preferred_realtime_forecast(date: str) -> Path:
    exact = REALTIME_DIR / f"realtime_verified_v33_multiradius_r40_r60_r75_r100_{date}.parquet"
    if exact.exists():
        return exact
    candidates = sorted(
        REALTIME_DIR.glob(f"realtime_verified_v33_multiradius_*_{date}.parquet"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        verification = _realtime_verification(date)
        if verification is not None:
            return verification
        raise RuntimeError(f"No realtime multi-radius forecast parquet for {date}")
    return candidates[0]


def _realtime_verification(date: str) -> Path | None:
    exact = REALTIME_DIR / f"realtime_ufvs_verified_v33_multiradius_r40_r60_r75_r100_{date}.parquet"
    if exact.exists():
        return exact
    candidates = sorted(
        REALTIME_DIR.glob(f"realtime_ufvs_verified_v33_multiradius_*_{date}.parquet"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def load_realtime(date: str) -> pd.DataFrame:
    base = _sort_grid(pd.read_parquet(_preferred_realtime_forecast(date)))
    verification_path = _realtime_verification(date)
    if verification_path is not None:
        verification = pd.read_parquet(verification_path)
        pp_columns = [column for column in verification.columns if column == "PP_Any flood proxy"]
        if pp_columns:
            base = _merge_aligned(base, verification, pp_columns)
    if "WPC_ERO_Risk" not in base.columns:
        wpc_candidates = sorted(
            REALTIME_WPC_DIR.glob(f"wpc_ero_risk_grid_{date}_valid12to12_*rows.parquet"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if wpc_candidates:
            wpc = pd.read_parquet(wpc_candidates[0])
            base = _merge_aligned(base, wpc, ["WPC_ERO_Risk"])
    return base


def load_case(date: str, source: str = "auto") -> tuple[pd.DataFrame, str]:
    if source in {"auto", "realtime"}:
        try:
            return load_realtime(date), "realtime"
        except Exception:
            if source == "realtime":
                raise
    return load_historical(date), "historical"


def probability_millipercent(values: pd.Series) -> list[int]:
    numeric = pd.to_numeric(values, errors="coerce").fillna(0.0).clip(0.0, 1.0).to_numpy(float)
    # 0..1000 represents probability percent to one decimal place in the browser.
    return np.rint(numeric * 1000.0).astype(np.uint16).tolist()


def contour_segments(lon: np.ndarray, lat: np.ndarray, values: np.ndarray) -> dict[str, list[list[list[float]]]]:
    values = np.nan_to_num(np.asarray(values, dtype=float), nan=0.0, posinf=1.0, neginf=0.0)
    result: dict[str, list[list[list[float]]]] = {}
    fig, ax = plt.subplots(figsize=(2, 2))
    try:
        contours = ax.tricontour(lon, lat, values, levels=THRESHOLDS)
        for threshold, groups in zip(THRESHOLDS, contours.allsegs):
            lines = []
            for group in groups:
                if len(group) < 2:
                    continue
                # Leaflet consumes [lat, lon]. Four decimals is ~10 m and keeps files compact.
                line = [[round(float(y), 4), round(float(x), 4)] for x, y in group]
                lines.append(line)
            result[str(int(round(threshold * 100)))] = lines
    finally:
        plt.close(fig)
    return result


def build_payload(frame: pd.DataFrame, date: str, source: str) -> dict:
    required = ["Date", "Lat", "Lon"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise RuntimeError(f"Map dataframe missing required columns: {missing}")
    frame = _sort_grid(frame)
    lat = pd.to_numeric(frame["Lat"], errors="coerce").to_numpy(float)
    lon = pd.to_numeric(frame["Lon"], errors="coerce").to_numpy(float)
    if not np.isfinite(lat).all() or not np.isfinite(lon).all():
        raise RuntimeError("Map grid contains invalid coordinates")

    layers = {}
    contours = {}
    for key, (label, column, kind) in LAYER_SPECS.items():
        if column not in frame.columns:
            continue
        numeric = pd.to_numeric(frame[column], errors="coerce").fillna(0.0).clip(0.0, 1.0)
        layers[key] = {
            "label": label,
            "kind": kind,
            "values": probability_millipercent(numeric),
        }
        contours[key] = contour_segments(lon, lat, numeric.to_numpy(float))

    start = datetime.strptime(date + "12", "%Y%m%d%H").replace(tzinfo=timezone.utc)
    end = start + timedelta(days=1)
    return {
        "schema_version": 1,
        "date": date,
        "valid_period_label": f"{start:%Y-%m-%d} 12Z to {end:%Y-%m-%d} 12Z",
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_class": source,
        "probability_encoding": "integer 0..1000; divide by 10 for percent",
        "risk_threshold_percent": [5, 15, 40, 70],
        "grid": {
            "lat": np.round(lat, 5).tolist(),
            "lon": np.round(lon, 5).tolist(),
        },
        "layers": layers,
        "contours": contours,
    }


def write_frame_map_data(frame: pd.DataFrame, date: str, output: Path, source: str) -> Path:
    date = date8(date)
    payload = build_payload(frame, date, source)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    print(
        f"Wrote interactive map data: {output} "
        f"rows={len(frame):,} layers={list(payload['layers'])} size={output.stat().st_size:,} bytes",
        flush=True,
    )
    return output


def write_map_data(date: str, output: Path, source: str = "auto") -> Path:
    date = date8(date)
    frame, selected_source = load_case(date, source=source)
    return write_frame_map_data(frame, date, output, selected_source)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True, help="Forecast valid-start date YYYYMMDD")
    parser.add_argument("--output", required=True, help="Destination map.json")
    parser.add_argument("--source", choices=("auto", "realtime", "historical"), default="auto")
    args = parser.parse_args()
    write_map_data(args.date, Path(args.output), source=args.source)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
