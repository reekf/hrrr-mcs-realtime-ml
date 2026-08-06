#!/usr/bin/env python3
"""Reproducible compact predictor contract for the v34 Day-1 experiments."""

from __future__ import annotations

import re
from collections.abc import Iterable


REFERENCE_MODEL_PATH = (
    "/home/tyreekfrazier/ISU_Research_LOCAL_RUN/fall_2025_ml_proj/prob_flood_models/"
    "current_v33_r100km_XGBoost_model.pkl"
)
REFERENCE_MODEL_SHA256 = "e704a905e70e8029136394c63eedd2c9c50240837436291ff4a297c7e23249ad"
REFERENCE_FEATURES_PATH = (
    "/home/tyreekfrazier/ISU_Research_LOCAL_RUN/fall_2025_ml_proj/prob_flood_models/"
    "current_v33_r100km_feature_names.json"
)
REFERENCE_FEATURES_SHA256 = "4a1cf4ae1ff0fc94494d1c3d800a5e35d641d8fef0ccf10b7b91c2f599f9bd44"
REFERENCE_IMPORTANCE_METHOD = "XGBClassifier.feature_importances_ (gain)"

# Captured directly from the authoritative v33 100-km Day-1 model above on
# 2026-08-06.  Keeping the values here makes the selection reproducible even
# when model aliases are later advanced to a newer experiment.
TOP_20_REFERENCE_FEATURES: tuple[tuple[str, float], ...] = (
    ("Forecast_APCP_Max_6h_Window_0to24h_to_Guidance_FFG_06h_Ratio_R100km_Mean", 0.09737909),
    ("Forecast_APCP_Max_6h_Window_0to24h_to_Guidance_FFG_06h_Ratio_R100km_Max", 0.05866297),
    ("Forecast_APCP_Max_6h_Window_0to24h_to_Guidance_FFG_06h_Ratio_R100km_Std", 0.05843756),
    ("SBCAPE_0_6_12_18_24h_Mean_R100km_Mean", 0.05402654),
    ("SBCAPE_0_6_12_18_24h_Max_R100km_Mean", 0.04976040),
    ("Forecast_APCP_to_Guidance_FFG_Ratio_Across_6h12h24h_Mean_R100km_Max", 0.03888644),
    ("Forecast_APCP_to_Guidance_FFG_Ratio_Across_6h12h24h_Mean_R100km_Std", 0.01989537),
    ("MLCIN_0_6_12_18_24h_Max_R100km_Mean", 0.01504305),
    ("Guidance_FFG_1h3h6h12h24h_mm_Mean_R100km_Mean", 0.01494844),
    ("Forecast_APCP_to_Guidance_FFG_Ratio_Across_6h12h24h_Min_R100km_Max", 0.01383717),
    ("10m_U_Wind_0_6_12_18_24h_Min_R100km_Min", 0.01148361),
    ("MLCAPE_0_6_12_18_24h_Min_R100km_Mean", 0.01143500),
    ("Forecast_APCP_RunningTotals_0to6_0to12_0to18_0to24h_mm_Std_R100km_Mean", 0.01126676),
    ("MLCAPE_0_6_12_18_24h_Mean_R100km_Max", 0.01123108),
    ("MLCAPE_0_6_12_18_24h_Mean_R100km_Mean", 0.01058620),
    ("MCS_Maintenance_Prob_RAPCalc_0_6_12_18_24h_Min_R100km_Mean", 0.00972000),
    ("Guidance_FFG_1h3h6h12h24h_mm_Mean_R100km_Min", 0.00967891),
    ("10m_V_Wind_0_6_12_18_24h_Mean_R100km_Std", 0.00711853),
    ("Forecast_APCP_RunningTotals_0to6_0to12_0to18_0to24h_mm_Max_R100km_Mean", 0.00697335),
    ("MLCAPE_0_6_12_18_24h_Max_R100km_Max", 0.00621405),
)

SUMMARY_STATS = ("Mean", "Min", "Max", "Std")
SPATIAL_SUFFIX_RE = re.compile(r"_R\d+km_(Mean|Min|Max|Std)$")
FORECAST_HOUR_SUFFIX_RE = re.compile(r"_0_6_12_18_24h_(Mean|Min|Max|Std)$")

# These are summaries across accumulation duration rather than forecast hour,
# but they receive the same mean/min/max/std expansion before spatial summaries.
DURATION_SUMMARY_BASES = (
    "Forecast_APCP_RunningTotals_0to6_0to12_0to18_0to24h_mm",
    "Guidance_FFG_1h3h6h12h24h_mm",
    "Forecast_APCP_to_Guidance_FFG_Ratio_Across_6h12h24h",
)

DIRECT_SUMMARY_BASES = (
    "Forecast_APCP_Max_6h_Window_0to24h_to_Guidance_FFG_06h_Ratio",
)

SIMULATED_REFLECTIVITY_BASE = "Simulated_Composite_Reflectivity_dBZ"


def base_family(feature_name: str) -> str:
    """Collapse one final spatial/temporal feature to its source family."""
    source = SPATIAL_SUFFIX_RE.sub("", str(feature_name))
    source = FORECAST_HOUR_SUFFIX_RE.sub("", source)
    for base in DURATION_SUMMARY_BASES:
        if source in {f"{base}_{stat}" for stat in SUMMARY_STATS}:
            return base
    if source in DIRECT_SUMMARY_BASES:
        return source
    return source


def _ordered_unique(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


TOP_20_BASE_FAMILIES = _ordered_unique(
    base_family(feature) for feature, _importance in TOP_20_REFERENCE_FEATURES
)


def summary_kind(base: str) -> str:
    if base in DIRECT_SUMMARY_BASES:
        return "direct"
    if base in DURATION_SUMMARY_BASES:
        return "duration"
    return "forecast_hour"


def source_summary_features(*, include_reflectivity: bool = True) -> tuple[str, ...]:
    """Return pre-spatial features selected from the top-20 base families."""
    bases = list(TOP_20_BASE_FAMILIES)
    if include_reflectivity:
        bases.append(SIMULATED_REFLECTIVITY_BASE)

    selected: list[str] = []
    for base in bases:
        kind = summary_kind(base)
        if kind == "direct":
            selected.append(base)
        elif kind == "duration":
            selected.extend(f"{base}_{stat}" for stat in SUMMARY_STATS)
        else:
            selected.extend(f"{base}_0_6_12_18_24h_{stat}" for stat in SUMMARY_STATS)
    return _ordered_unique(selected)


def expected_model_features(radius_km: int | float) -> tuple[str, ...]:
    """Return the exact ordered 164-predictor contract for one radius."""
    radius = int(round(float(radius_km)))
    if radius not in {75, 100}:
        raise ValueError(f"v34 compact training supports only 75 or 100 km, got {radius_km!r}")
    return tuple(
        f"{source}_R{radius}km_{spatial_stat}"
        for source in source_summary_features()
        for spatial_stat in SUMMARY_STATS
    )


def validate_contract() -> None:
    if len(TOP_20_REFERENCE_FEATURES) != 20:
        raise RuntimeError("The reference importance contract must contain exactly 20 features")
    if len(TOP_20_BASE_FAMILIES) != 10:
        raise RuntimeError(
            f"Expected the top 20 to collapse to 10 base families, got {TOP_20_BASE_FAMILIES}"
        )
    if len(source_summary_features()) != 41:
        raise RuntimeError("Expected 41 pre-spatial summaries including simulated reflectivity")
    if len(expected_model_features(75)) != 164 or len(expected_model_features(100)) != 164:
        raise RuntimeError("Expected exactly 164 compact predictors per model")


validate_contract()
