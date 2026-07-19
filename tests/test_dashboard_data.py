#!/usr/bin/env python3
"""Unit tests for XGBFFP rolling-verification aggregation."""

from datetime import date
import json
import math
from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import generate_dashboard_data as dashboard


REPO_DIR = Path(__file__).resolve().parents[1]


def test_categorical_metrics_zero_denominators_are_null():
    metrics = dashboard.categorical_metrics(0, 0, 0, 10)
    assert metrics["ets"] is None
    assert metrics["csi"] is None
    assert metrics["pod"] is None
    assert metrics["far"] is None
    assert metrics["frequency_bias"] is None


def test_threshold_boundaries_are_inclusive():
    truth = [50, 150, 400, 700]
    for threshold, index in zip(dashboard.THRESHOLDS, range(4)):
        metrics = dashboard.daily_product(truth, truth, threshold)
        expected_positives = 4 - index
        assert metrics["hits"] == expected_positives
        assert metrics["misses"] == 0
        assert metrics["false_alarms"] == 0


def test_december_is_assigned_to_following_djf():
    start, end, name = dashboard.season_bounds(date(2026, 12, 15))
    assert (start, end, name) == (date(2026, 12, 1), date(2027, 2, 28), "DJF")
    start, end, name = dashboard.season_bounds(date(2027, 1, 5))
    assert (start, end, name) == (date(2026, 12, 1), date(2027, 2, 28), "DJF")


def test_pooled_counts_are_recalculated():
    def record(day, hits, misses, false_alarms, correct_negatives):
        sample_count = hits + misses + false_alarms + correct_negatives
        row = {
            "hits": hits,
            "misses": misses,
            "false_alarms": false_alarms,
            "correct_negatives": correct_negatives,
            "sample_count": sample_count,
            "truth_positive_count": hits + misses,
            "forecast_positive_count": hits + false_alarms,
            "squared_error_sum": 1.0,
        }
        return {
            "date": day,
            "products": {"ml_r40": {str(value): row.copy() for value in dashboard.THRESHOLDS}},
        }

    window = dashboard.aggregate_window(
        [
            record("20260701", 2, 1, 1, 6),
            record("20260702", 3, 2, 1, 4),
            record("20260703", 0, 3, 0, 7),
        ],
        date(2026, 7, 1),
        date(2026, 7, 3),
        "test",
        "weekly",
    )
    result = window["products"]["ml_r40"]["5"]
    assert result["hits"] == 5
    assert result["misses"] == 6
    assert result["false_alarms"] == 2
    assert result["sample_count"] == 30
    assert result["risk_case_count"] == 2
    assert result["verified_forecast_count"] == 3
    assert "brier_skill_score" not in result


def test_published_manifests_and_verification_contracts():
    docs = REPO_DIR / "docs"
    for manifest_path in [
        docs / "model-skill/manifest.json",
        docs / "explainability/manifest.json",
    ]:
        manifest = json.loads(manifest_path.read_text())
        assert manifest["schema_version"] == 1
        for figure in manifest["figures"]:
            assert (docs / figure["path"]).is_file()

    skill = json.loads((docs / "model-skill/manifest.json").read_text())
    assert len(skill["figures"]) == 3
    assert [figure["metric"] for figure in skill["figures"]].count("ETS") == 2
    assert [figure["metric"] for figure in skill["figures"]].count("Brier Score") == 1
    assert {figure["source_directory"] for figure in skill["figures"]} == {
        "paper_verification_bs_ets_final"
    }
    for figure in skill["figures"]:
        assert "MRMS" not in figure["title"]
        assert "MRMS" not in figure["target"]
        assert "Brier Skill Score" not in figure["title"]
        assert "risk-area" not in figure["title"].lower()
        if figure["metric"] == "Brier Score":
            assert figure["evaluations"] == [
                "Including Marginal",
                "Excluding Marginal",
            ]

    contingency = json.loads((docs / "model-skill/risk-frequency.json").read_text())
    assert contingency["schema_version"] == 3
    assert set(contingency["excluded_products"]) == {
        "ML Local PMM 100km",
        "ML Ensemble Max",
        "ML r100kmV2",
    }
    assert not set(contingency["excluded_products"]) & set(contingency["products"])
    for thresholds in contingency["products"].values():
        for counts in thresholds.values():
            assert isinstance(counts["hit_grid_cell_count"], int)
            assert isinstance(counts["false_alarm_grid_cell_count"], int)
            assert isinstance(counts["miss_grid_cell_count"], int)
            assert counts["hit_grid_cell_count"] >= 0
            assert counts["false_alarm_grid_cell_count"] >= 0
            assert counts["miss_grid_cell_count"] >= 0

    index = json.loads((docs / "verification/index.json").read_text())
    assert index["dataset_class"] == "realtime-issued-verification"
    daily_dates = set(index["daily_dates"])
    for day in daily_dates:
        record = json.loads((docs / f"verification/daily/{day}.json").read_text())
        assert record["schema_version"] == 2
        assert record["dataset_class"] == "realtime-issued-verification"
        for thresholds in record["products"].values():
            assert set(thresholds) == {"5", "15", "40", "70"}
            for metrics in thresholds.values():
                for count_name in [
                    "hits",
                    "misses",
                    "false_alarms",
                    "correct_negatives",
                    "sample_count",
                ]:
                    assert isinstance(metrics[count_name], int)
                    assert metrics[count_name] >= 0
                for value in metrics.values():
                    assert value is None or not isinstance(value, float) or math.isfinite(value)
                assert "brier_skill_score" not in metrics

    rolling = json.loads((docs / "verification/rolling/latest.json").read_text())
    assert rolling["schema_version"] == 2
    assert rolling["dataset_class"] == "realtime-issued-verification"
    for window in rolling["windows"].values():
        assert window["schema_version"] == 2
        assert set(window["verified_dates"]).issubset(daily_dates)
        assert window["missing_day_count"] >= 0
        assert window["start_date"] <= window["end_date"]
        for thresholds in window["products"].values():
            for metrics in thresholds.values():
                assert "brier_skill_score" not in metrics
                assert 0 <= metrics["risk_case_count"] <= metrics["verified_forecast_count"]

    index_html = (docs / "index.html").read_text()
    assert '<option value="40" selected>Moderate or greater</option>' in index_html
    assert "Brier Skill Score" not in index_html


if __name__ == "__main__":
    test_categorical_metrics_zero_denominators_are_null()
    test_threshold_boundaries_are_inclusive()
    test_december_is_assigned_to_following_djf()
    test_pooled_counts_are_recalculated()
    test_published_manifests_and_verification_contracts()
    print("Dashboard data unit tests passed.")
