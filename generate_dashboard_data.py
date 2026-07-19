#!/usr/bin/env python3
"""Publish XGBFFP model-skill assets and realtime rolling verification.

This script consumes saved final figures/tables and already-published map JSON.
It never imports the training notebook or retrains an XGBoost model.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parent
DOCS_DIR = REPO_DIR / "docs"
DEFAULT_PROJECT_DIR = REPO_DIR.parents[1] / "fall_2025_ml_proj"
THRESHOLDS = (5, 15, 40, 70)
THRESHOLD_LABELS = {
    5: "Marginal or greater",
    15: "Slight or greater",
    40: "Moderate or greater",
    70: "High",
}
PRODUCTS = (
    "ml_r40",
    "ml_r60",
    "ml_r60v2",
    "ml_r75",
    "ml_r100",
    "ml_mean",
    "wpc",
)
PRODUCT_LABELS = {
    "ml_r40": "ML r40",
    "ml_r60": "ML r60",
    "ml_r60v2": "ML r60kmV2",
    "ml_r75": "ML r75",
    "ml_r100": "ML r100",
    "ml_mean": "ML ensemble mean",
    "wpc": "WPC ERO",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    value = numerator / denominator
    return round(value, 8) if math.isfinite(value) else None


def categorical_metrics(hits: int, misses: int, false_alarms: int, correct_negatives: int) -> dict:
    total = hits + misses + false_alarms + correct_negatives
    random_hits = safe_ratio((hits + misses) * (hits + false_alarms), total)
    ets_denominator = None if random_hits is None else hits + misses + false_alarms - random_hits
    return {
        "ets": None if random_hits is None else safe_ratio(hits - random_hits, ets_denominator),
        "csi": safe_ratio(hits, hits + misses + false_alarms),
        "pod": safe_ratio(hits, hits + misses),
        "far": safe_ratio(false_alarms, hits + false_alarms),
        "frequency_bias": safe_ratio(hits + false_alarms, hits + misses),
    }


def daily_product(values: list[int], truth_values: list[int], threshold: int) -> dict:
    encoded_threshold = threshold * 10
    hits = misses = false_alarms = correct_negatives = 0
    squared_error_sum = 0.0
    for forecast_encoded, truth_encoded in zip(values, truth_values):
        forecast_yes = forecast_encoded >= encoded_threshold
        truth_yes = truth_encoded >= encoded_threshold
        if forecast_yes and truth_yes:
            hits += 1
        elif truth_yes:
            misses += 1
        elif forecast_yes:
            false_alarms += 1
        else:
            correct_negatives += 1
        probability = max(0.0, min(1.0, float(forecast_encoded) / 1000.0))
        squared_error_sum += (probability - float(truth_yes)) ** 2
    sample_count = hits + misses + false_alarms + correct_negatives
    metrics = categorical_metrics(hits, misses, false_alarms, correct_negatives)
    metrics.update(
        {
            "hits": hits,
            "misses": misses,
            "false_alarms": false_alarms,
            "correct_negatives": correct_negatives,
            "sample_count": sample_count,
            "truth_positive_count": hits + misses,
            "forecast_positive_count": hits + false_alarms,
            "squared_error_sum": round(squared_error_sum, 8),
            "brier_score": safe_ratio(squared_error_sum, sample_count),
        }
    )
    return metrics


def load_realtime_daily(archive_dir: Path) -> list[dict]:
    records = []
    for path in sorted(archive_dir.glob("20??????/map.json")):
        payload = json.loads(path.read_text())
        if payload.get("source_class") != "realtime":
            continue
        layers = payload.get("layers", {})
        truth_values = layers.get("pp", {}).get("values")
        if not isinstance(truth_values, list) or not truth_values:
            continue
        if any(not isinstance(value, (int, float)) or not 0 <= value <= 1000 for value in truth_values):
            raise ValueError(f"{path}: PP probabilities must be finite values from 0 to 1000")
        grid_count = len(payload.get("grid", {}).get("lat", []))
        if grid_count != len(truth_values):
            raise ValueError(f"{path}: PP/grid lengths differ")
        products = {}
        for product in PRODUCTS:
            values = layers.get(product, {}).get("values")
            if not isinstance(values, list):
                continue
            if len(values) != grid_count:
                raise ValueError(f"{path}: {product}/grid lengths differ")
            if any(not isinstance(value, (int, float)) or not 0 <= value <= 1000 for value in values):
                raise ValueError(f"{path}: {product} probabilities must be finite values from 0 to 1000")
            products[product] = {
                str(threshold): daily_product(values, truth_values, threshold)
                for threshold in THRESHOLDS
            }
        if not products:
            continue
        records.append(
            {
                "schema_version": 2,
                "dataset_class": "realtime-issued-verification",
                "verification_target": "Practically Perfect: Any flood proxy",
                "date": str(payload["date"]),
                "valid_period_label": payload.get("valid_period_label", ""),
                "products": products,
            }
        )
    return records


def parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y%m%d").date()


def season_bounds(day: date) -> tuple[date, date, str]:
    if day.month in (12, 1, 2):
        start_year = day.year if day.month == 12 else day.year - 1
        return date(start_year, 12, 1), date(start_year + 1, 2, 28), "DJF"
    if day.month in (3, 4, 5):
        return date(day.year, 3, 1), date(day.year, 5, 31), "MAM"
    if day.month in (6, 7, 8):
        return date(day.year, 6, 1), date(day.year, 8, 31), "JJA"
    return date(day.year, 9, 1), date(day.year, 11, 30), "SON"


def select_windows(records: list[dict]) -> dict[str, tuple[list[dict], date, date, str]]:
    if not records:
        return {}
    ordered = sorted(records, key=lambda row: row["date"])
    latest = parse_date(ordered[-1]["date"])
    weekly_rows = ordered[-7:]
    weekly_start = parse_date(weekly_rows[0]["date"])
    monthly_start = latest - timedelta(days=29)
    seasonal_start, seasonal_end, season = season_bounds(latest)
    return {
        "weekly": (weekly_rows, weekly_start, latest, "Latest seven verified forecasts"),
        "monthly": (
            [row for row in ordered if monthly_start <= parse_date(row["date"]) <= latest],
            monthly_start,
            latest,
            "Trailing 30 calendar days",
        ),
        "seasonal": (
            [row for row in ordered if seasonal_start <= parse_date(row["date"]) <= latest],
            seasonal_start,
            min(latest, seasonal_end),
            f"{season} meteorological season to date",
        ),
    }


def aggregate_window(
    records: list[dict], start: date, end: date, definition: str, window_name: str
) -> dict:
    products = {}
    for product in PRODUCTS:
        threshold_payload = {}
        for threshold in THRESHOLDS:
            rows = [
                record["products"][product][str(threshold)]
                for record in records
                if product in record["products"]
            ]
            if not rows:
                continue
            summed = {
                key: sum(int(row[key]) for row in rows)
                for key in (
                    "hits",
                    "misses",
                    "false_alarms",
                    "correct_negatives",
                    "sample_count",
                    "truth_positive_count",
                    "forecast_positive_count",
                )
            }
            squared_error_sum = sum(float(row["squared_error_sum"]) for row in rows)
            metrics = categorical_metrics(
                summed["hits"],
                summed["misses"],
                summed["false_alarms"],
                summed["correct_negatives"],
            )
            brier = safe_ratio(squared_error_sum, summed["sample_count"])
            metrics.update(summed)
            metrics.update(
                {
                    "brier_score": brier,
                    "risk_case_count": sum(
                        int(row["forecast_positive_count"]) > 0 for row in rows
                    ),
                    "verified_forecast_count": len(rows),
                }
            )
            threshold_payload[str(threshold)] = metrics
        if threshold_payload:
            products[product] = threshold_payload
    expected_days = (end - start).days + 1
    return {
        "schema_version": 2,
        "dataset_class": "realtime-issued-verification",
        "verification_target": "Practically Perfect: Any flood proxy",
        "window": window_name,
        "definition": definition,
        "start_date": start.strftime("%Y%m%d"),
        "end_date": end.strftime("%Y%m%d"),
        "verified_forecast_count": len(records),
        "expected_calendar_days": expected_days,
        "missing_day_count": max(0, expected_days - len(records)),
        "completeness_percent": round(len(records) / expected_days * 100.0, 1),
        "verified_dates": [record["date"] for record in records],
        "products": products,
    }


def copy_asset(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"Required finalized figure is unavailable: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def publish_skill_assets(project_dir: Path, docs_dir: Path, generated: str) -> dict:
    final = project_dir / "paper_verification_bs_ets_final"
    specs = [
        {
            "source_name": "ets_any_flood_proxy_ets.png",
            "path": "model-skill/ets_any_flood_proxy.png",
            "title": "Any Flood Proxy ETS",
            "metric": "ETS",
            "target": "Any flood proxy",
            "thresholds_percent": [5],
            "source_function": "compute_ets_pod_far / run_final_bs_ets_verification_plots",
        },
        {
            "source_name": "ets_pp_any_flood_proxy_ets.png",
            "path": "model-skill/ets_practically_perfect.png",
            "title": "Practically Perfect ETS by threshold",
            "metric": "ETS",
            "target": "Practically Perfect: Any flood proxy",
            "thresholds_percent": list(THRESHOLDS),
            "source_function": "compute_ets_pod_far / run_final_bs_ets_verification_plots",
        },
        {
            "source_name": "bs_any_flood_proxy_include_exclude_marginal.png",
            "path": "model-skill/brier_any_flood_proxy_including_excluding_marginal.png",
            "title": "Any Flood Proxy Brier Score",
            "metric": "Brier Score",
            "target": "Any flood proxy",
            "thresholds_percent": [5, 15],
            "evaluations": ["Including Marginal", "Excluding Marginal"],
            "source_function": "run_final_bs_ets_verification_plots",
        },
    ]
    for stale_name in [
        "ets_ufvs_any_violin.png",
        "brier_ufvs_any_violin.png",
        "bss_ufvs_any_violin.png",
        "risk_area_occurrence.png",
        "ets_mrms_ffg.png",
        "brier_mrms_ffg_including_excluding_marginal.png",
    ]:
        (docs_dir / "model-skill" / stale_name).unlink(missing_ok=True)

    figures = []
    for spec in specs:
        source = final / spec["source_name"]
        relative = spec["path"]
        destination = docs_dir / relative
        copy_asset(source, destination)
        figure = {
            "title": spec["title"],
            "metric": spec["metric"],
            "target": spec["target"],
            "thresholds_percent": spec["thresholds_percent"],
            "test_period": "2024–2025",
            "test_case_count": 45,
            "test_date_range": "20240610–20250729",
            "model": "XGBoost v33 radius configurations and WPC ERO",
            "source_script": (
                "hazard_ml_v33_radiusstats_WORKING_BASELINE_PLUS_VERIFICATION_SHAP_"
                "REALTIME_MULTIRADIUS_ENSEMBLE_WPC_VALIDFIX_METRICS_PREDICTORS_"
                "v18_PP_EXCLUSIVE_PROXY_CUMULATIVE_VIOLINS.ipynb"
            ),
            "source_directory": "paper_verification_bs_ets_final",
            "source_asset": f"paper_verification_bs_ets_final/{spec['source_name']}",
            "source_function": spec["source_function"],
            "generated_utc": generated,
            "path": relative,
        }
        if "evaluations" in spec:
            figure["evaluations"] = spec["evaluations"]
        figures.append(figure)
    manifest = {
        "schema_version": 1,
        "dataset_class": "formal-independent-test-set",
        "test_period": "2024–2025",
        "generated_utc": generated,
        "figures": figures,
    }
    write_json(docs_dir / "model-skill/manifest.json", manifest)
    return manifest


def publish_risk_frequency(project_dir: Path, docs_dir: Path, generated: str) -> dict:
    source = project_dir / "paper_verification_bs_ets_final/ets_pp_any_flood_proxy_metrics.csv"
    if not source.is_file():
        raise FileNotFoundError(source)
    excluded_sources = {"ML Local PMM 100km", "ML Ensemble Max", "ML r100kmV2"}
    grouped: dict[tuple[str, int], dict[str, int]] = defaultdict(
        lambda: {"hits": 0, "false_alarms": 0, "misses": 0}
    )
    with source.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["Source"] in excluded_sources:
                continue
            threshold = int(round(float(row["Threshold"]) * 100))
            if threshold not in THRESHOLDS:
                continue
            key = (row["Source"], threshold)
            grouped[key]["hits"] += int(row["Hits"])
            grouped[key]["false_alarms"] += int(row["False Alarms"])
            grouped[key]["misses"] += int(row["Misses"])
    products: dict[str, dict] = defaultdict(dict)
    for (source_label, threshold), values in sorted(grouped.items()):
        products[source_label][str(threshold)] = {
            "threshold_label": THRESHOLD_LABELS[threshold],
            "hit_grid_cell_count": values["hits"],
            "false_alarm_grid_cell_count": values["false_alarms"],
            "miss_grid_cell_count": values["misses"],
        }
    payload = {
        "schema_version": 3,
        "dataset_class": "formal-independent-test-set",
        "test_period": "2024–2025",
        "verification_target": "Practically Perfect: Any flood proxy",
        "count_unit": "summed grid-cell contingency counts across 45 test cases",
        "excluded_products": sorted(excluded_sources),
        "generated_utc": generated,
        "products": dict(products),
        "source_table": "paper_verification_bs_ets_final/ets_pp_any_flood_proxy_metrics.csv",
    }
    write_json(docs_dir / "model-skill/risk-frequency.json", payload)
    return payload


def publish_explainability_assets(project_dir: Path, docs_dir: Path, generated: str) -> dict:
    global_dir = project_dir / "paper_shap_figures"
    dependence_dir = project_dir / "shap_dependence_r100"
    specs = [
        (global_dir / "global_shap_beeswarm_r60km.png", "explainability/shap/r60/beeswarm.png", "r60", "beeswarm", "Global SHAP beeswarm: r60"),
        (global_dir / "global_shap_importance_r60km.png", "explainability/shap/r60/importance.png", "r60", "importance", "Mean absolute SHAP importance: r60"),
        (global_dir / "global_shap_beeswarm_r100km.png", "explainability/shap/r100/beeswarm.png", "r100", "beeswarm", "Global SHAP beeswarm: r100"),
        (global_dir / "global_shap_importance_r100km.png", "explainability/shap/r100/importance.png", "r100", "importance", "Mean absolute SHAP importance: r100"),
        (dependence_dir / "top5_shap_dependence_subplots_r100km.png", "explainability/dependence/r100/top5.png", "r100", "dependence", "Top-five SHAP dependence plots: r100"),
        (dependence_dir / "selected_shap_dependence_rank1_rank8to10_r100km.png", "explainability/dependence/r100/selected.png", "r100", "dependence", "Selected SHAP dependence plots: r100"),
    ]
    figures = []
    for source, relative, model, kind, title in specs:
        copy_asset(source, docs_dir / relative)
        figures.append(
            {
                "title": title,
                "kind": kind,
                "model": model,
                "test_period": "2024–2025",
                "source_function": "plot_shap_global_summary" if kind != "dependence" else "selected SHAP dependence plot block",
                "generated_utc": generated,
                "path": relative,
            }
        )
    manifest = {
        "schema_version": 1,
        "dataset_class": "formal-independent-test-set-explainability",
        "test_period": "2024–2025",
        "generated_utc": generated,
        "figures": figures,
    }
    write_json(docs_dir / "explainability/manifest.json", manifest)
    return manifest


def validate_manifest_paths(docs_dir: Path, manifest: dict) -> None:
    for figure in manifest.get("figures", []):
        if not (docs_dir / figure["path"]).is_file():
            raise FileNotFoundError(f"Manifest path is missing: {figure['path']}")


def publish_realtime_verification(docs_dir: Path, generated: str) -> dict:
    output_root = docs_dir / "verification"
    records = load_realtime_daily(docs_dir / "archive")
    for record in records:
        write_json(output_root / f"daily/{record['date']}.json", record)
    windows = {}
    for name, (selected, start, end, definition) in select_windows(records).items():
        window = aggregate_window(selected, start, end, definition, name)
        window["generated_utc"] = generated
        windows[name] = window
        write_json(output_root / f"rolling/{name}.json", window)
    latest = {
        "schema_version": 2,
        "dataset_class": "realtime-issued-verification",
        "generated_utc": generated,
        "windows": windows,
    }
    write_json(output_root / "rolling/latest.json", latest)
    index = {
        "schema_version": 2,
        "dataset_class": "realtime-issued-verification",
        "generated_utc": generated,
        "daily_record_count": len(records),
        "daily_dates": [row["date"] for row in records],
        "daily_path_template": "verification/daily/{date}.json",
        "rolling_paths": {
            name: f"verification/rolling/{name}.json" for name in windows
        },
    }
    write_json(output_root / "index.json", index)
    return index


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, default=DEFAULT_PROJECT_DIR)
    parser.add_argument("--docs-dir", type=Path, default=DOCS_DIR)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--verification-only",
        action="store_true",
        help="Refresh only realtime daily/rolling verification JSON.",
    )
    mode.add_argument(
        "--skill-only",
        action="store_true",
        help="Refresh only formal test-set skill figures and contingency counts.",
    )
    args = parser.parse_args()
    generated = utc_now()
    if not args.verification_only:
        skill = publish_skill_assets(args.project_dir, args.docs_dir, generated)
        publish_risk_frequency(args.project_dir, args.docs_dir, generated)
        validate_manifest_paths(args.docs_dir, skill)
        if not args.skill_only:
            explainability = publish_explainability_assets(args.project_dir, args.docs_dir, generated)
            validate_manifest_paths(args.docs_dir, explainability)
    if not args.skill_only:
        publish_realtime_verification(args.docs_dir, generated)
    print(f"Published XGBFFP dashboard data under {args.docs_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
