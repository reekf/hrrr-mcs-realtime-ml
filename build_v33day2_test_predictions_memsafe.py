#!/usr/bin/env python3
"""Build the four v33 Day-2 test prediction caches outside Jupyter.

This launches the bounded-memory prediction functions embedded in the canonical
Day-2 viewer, one radius at a time. Running outside the notebook prevents stale
open editor cells and already-resident viewer dataframes from consuming memory
during prediction generation.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import resource
from pathlib import Path


ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "hazard_ml_v33_day2_verification_viewer.ipynb"
PREDICTION_SECTION_MARKER = (
    "# ======================================================================================\n"
    "# 4. Load PP/WPC grid and radius predictions\n"
    "# ======================================================================================\n"
)
CORE_CACHE_COLUMNS = [
    "Date",
    "RAP_Init_Date",
    "Year",
    "Lat",
    "Lon",
    "ML_Target_Radius_km",
    "ML_Model_Label",
    "ML_Forecast_Prob",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build resumable v33 Day-2 test prediction caches with bounded memory."
        )
    )
    parser.add_argument(
        "--radii",
        nargs="+",
        type=int,
        default=[40, 60, 75, 100],
        choices=[40, 60, 75, 100],
        help="Target radii to process in order.",
    )
    parser.add_argument(
        "--batch-rows",
        type=int,
        default=6000,
        help="Source rows per inference batch (default: 6000).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute completed fragments and consolidated caches.",
    )
    parser.add_argument(
        "--no-normalize-existing",
        action="store_true",
        help="Do not rewrite older caches to the narrow eight-column schema.",
    )
    return parser.parse_args()


def _load_prediction_namespace(batch_rows: int) -> dict:
    if not NOTEBOOK.exists():
        raise FileNotFoundError(NOTEBOOK)
    os.environ["XGBFFP_DAY2_PREDICT_BATCH_ROWS"] = str(int(batch_rows))
    notebook = json.loads(NOTEBOOK.read_text())
    setup_cells = [
        cell
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
        and "def _day2_stream_single_radius_predictions" in "".join(
            cell.get("source", [])
        )
    ]
    if len(setup_cells) != 1:
        raise RuntimeError(
            "Could not identify exactly one Day-2 prediction setup cell."
        )
    source = "".join(setup_cells[0]["source"])
    if PREDICTION_SECTION_MARKER not in source:
        raise RuntimeError("Prediction setup marker is missing from the viewer.")
    setup_source = source.split(PREDICTION_SECTION_MARKER, 1)[0]
    namespace = {
        "__name__": "xgbffp_day2_prediction_cache_builder",
        "__file__": str(NOTEBOOK),
    }
    exec(compile(setup_source, str(NOTEBOOK), "exec"), namespace)
    if "build_or_load_radius_predictions" not in namespace:
        raise RuntimeError("Bounded-memory prediction builder was not loaded.")
    return namespace


def _peak_rss_gib() -> float:
    # Linux reports ru_maxrss in KiB.
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 2**20


def _normalize_cache(namespace: dict, radius: int, model_label: str, frame) -> None:
    cache_path = Path(
        namespace["prediction_cache_path_for_radius"](
            radius, model_label=model_label
        )
    )
    available = set(namespace["pq"].ParquetFile(cache_path).schema.names)
    if available == set(CORE_CACHE_COLUMNS):
        return
    temporary = cache_path.with_suffix(cache_path.suffix + ".normalize.tmp")
    temporary.unlink(missing_ok=True)
    frame[CORE_CACHE_COLUMNS].to_parquet(
        temporary,
        index=False,
        compression="zstd",
    )
    os.replace(temporary, cache_path)
    print(
        "Normalized existing cache to eight columns with explicit RAP/valid "
        f"dates: {cache_path}"
    )


def _normalize_verification_grid(namespace: dict) -> None:
    cache_path = Path(namespace["PP_WPC_GRID_CACHE"])
    if not cache_path.exists():
        print(f"Verification grid is not present; skipped migration: {cache_path}")
        return
    pq = namespace["pq"]
    if "RAP_Init_Date" in pq.ParquetFile(cache_path).schema.names:
        print(
            "Verification grid already uses the historical valid-start date "
            f"convention: {cache_path}"
        )
        return
    pd = namespace["pd"]
    frame = pd.read_parquet(cache_path)
    event_dates = pd.to_datetime(
        frame["Date"].astype(str).str[:8],
        format="%Y%m%d",
        errors="coerce",
    )
    if event_dates.isna().any():
        raise ValueError(f"Unparseable Date values in {cache_path}")
    frame["Date"] = event_dates.dt.strftime("%Y%m%d")
    frame["RAP_Init_Date"] = (
        event_dates - pd.Timedelta(days=1)
    ).dt.strftime("%Y%m%d")
    frame["Year"] = frame["Date"].str[:4]
    temporary = cache_path.with_suffix(cache_path.suffix + ".normalize.tmp")
    temporary.unlink(missing_ok=True)
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, cache_path)
    print(
        "Normalized verification grid: Date is the event-valid start and "
        f"RAP_Init_Date is the preceding model initialization: {cache_path}"
    )


def main() -> int:
    args = _parse_args()
    if args.batch_rows < 500:
        raise ValueError("--batch-rows must be at least 500")
    namespace = _load_prediction_namespace(args.batch_rows)
    if not args.no_normalize_existing:
        _normalize_verification_grid(namespace)
    requested = set(args.radii)
    specs = [
        spec
        for spec in namespace["MODEL_SPECS"]
        if int(spec["radius_km"]) in requested
    ]
    print(
        "Standalone Day-2 prediction cache builder\n"
        f"  radii: {[int(spec['radius_km']) for spec in specs]}\n"
        f"  batch rows: {args.batch_rows:,}\n"
        f"  force: {args.force}"
    )
    for index, spec in enumerate(specs, start=1):
        radius = int(spec["radius_km"])
        model_label = str(spec["label"])
        print(
            "\n"
            + "=" * 88
            + f"\n[{index}/{len(specs)}] {model_label} test predictions"
            + "\n"
            + "=" * 88
        )
        frame = namespace["build_or_load_radius_predictions"](
            radius,
            force=bool(args.force),
            model_label=model_label,
        )
        if not args.no_normalize_existing:
            _normalize_cache(namespace, radius, model_label, frame)
        print(
            f"{model_label} ready: {len(frame):,} rows; "
            f"process peak RSS so far: {_peak_rss_gib():.2f} GiB"
        )
        del frame
        gc.collect()
    print(
        "\nAll requested Day-2 prediction caches are ready. "
        f"Final process peak RSS: {_peak_rss_gib():.2f} GiB"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
