#!/usr/bin/env python3
"""Generate and optionally run the compact v34 Day-1 XGBoost experiments.

Experiment matrix
-----------------
* target radii: 75 and 100 km
* observed MRMS/FFG ratio thresholds: >=1.0 (default), >=1.5, >=2.0
* predictors: the base families represented in the current v33 Day-1 top 20,
  expanded through the existing temporal/spatial summaries, plus RAP simulated
  composite reflectivity

The authoritative v33 100-km script remains unchanged and is used as the
well-tested data/target/training implementation.  This generator creates
isolated v34 scripts and artifact paths for each of the six experiments.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from day1_compact_feature_contract import (
    REFERENCE_FEATURES_PATH,
    REFERENCE_FEATURES_SHA256,
    REFERENCE_IMPORTANCE_METHOD,
    REFERENCE_MODEL_PATH,
    REFERENCE_MODEL_SHA256,
    SIMULATED_REFLECTIVITY_BASE,
    TOP_20_BASE_FAMILIES,
    TOP_20_REFERENCE_FEATURES,
    expected_model_features,
    source_summary_features,
)


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_SCRIPT = Path(
    os.environ.get(
        "HAZARD_ML_V34_BASE_SCRIPT",
        SCRIPT_DIR
        / "generated_v33_radius_sensitivity_slimmaster_rowsample"
        / "hazard_ml_training_v33_r100km_singletarget_radiusstats_MEMSAFE.py",
    )
)
GENERATED_DIR = Path(
    os.environ.get("HAZARD_ML_V34_GENERATED_DIR", SCRIPT_DIR / "generated_v34_compact_day1")
)
PROJECT_DIR = Path(
    os.environ.get(
        "HAZARD_ML_PROJECT_DIR",
        "/home/tyreekfrazier/ISU_Research_LOCAL_RUN/fall_2025_ml_proj",
    )
)
MODEL_DIR = PROJECT_DIR / "prob_flood_models"
PYTHON_EXE = os.environ.get("PYTHON", sys.executable)

SUPPORTED_RADII = (75, 100)
SUPPORTED_RATIO_THRESHOLDS = (1.0, 1.5, 2.0)
DEFAULT_RATIO_THRESHOLD = 1.0
RUN_VERSION = "v34compact"

EXPECTED_DOMAIN = {
    "TRAIN_DOMAIN_LAT_MIN": 30.0,
    "TRAIN_DOMAIN_LAT_MAX": 50.0,
    "TRAIN_DOMAIN_LON_MIN": -105.0,
    "TRAIN_DOMAIN_LON_MAX": -80.5,
}


@dataclass(frozen=True, order=True)
class Experiment:
    radius_km: int
    ratio_threshold: float

    def __post_init__(self) -> None:
        radius = int(self.radius_km)
        ratio = float(self.ratio_threshold)
        if radius not in SUPPORTED_RADII:
            raise ValueError(f"Unsupported target radius {radius}; choose from {SUPPORTED_RADII}")
        if ratio not in SUPPORTED_RATIO_THRESHOLDS:
            raise ValueError(
                f"Unsupported MRMS/FFG ratio {ratio}; choose from {SUPPORTED_RATIO_THRESHOLDS}"
            )

    @property
    def ratio_tag(self) -> str:
        return f"ge{self.ratio_threshold:.1f}".replace(".", "p")

    @property
    def experiment_tag(self) -> str:
        return (
            f"{RUN_VERSION}_r{self.radius_km}km_mrmsffg_{self.ratio_tag}_"
            "top20refl_radiusstats_logloss_domain"
        )

    @property
    def target_output_tag(self) -> str:
        return (
            f"r{self.radius_km}km_mrmsffg_{self.ratio_tag}_top20refl_"
            f"target_{RUN_VERSION}_domain"
        )

    @property
    def ratio_column_tag(self) -> str:
        return f"RatioGE{self.ratio_threshold:.1f}".replace(".", "p")

    @property
    def target_column(self) -> str:
        return f"Target_MRMS_FFG_{self.ratio_column_tag}_R{self.radius_km}km"

    @property
    def point_target_column(self) -> str:
        return f"Obs_MRMS_FFG_{self.ratio_column_tag}_Point"

    @property
    def generated_script(self) -> Path:
        return GENERATED_DIR / f"hazard_ml_training_{self.experiment_tag}_MEMSAFE.py"

    @property
    def master_parquet(self) -> Path:
        return PROJECT_DIR / (
            f"pixel_domain_forecasts_rap09z_iem_mrms_ffg_{self.target_output_tag}.parquet"
        )

    @property
    def manifest(self) -> Path:
        return MODEL_DIR / f"active_artifacts_{self.experiment_tag}.json"


def _replace_exact(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Expected exactly one {label} block, found {count}")
    return text.replace(old, new, 1)


def _replace_regex_once(text: str, pattern: str, replacement: str, label: str) -> str:
    result, count = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE | re.DOTALL)
    if count != 1:
        raise RuntimeError(f"Expected exactly one {label} regex match, found {count}")
    return result


def _python_tuple(values) -> str:
    return repr(tuple(values))


def _assert_base_domain(source: str) -> None:
    for name, value in EXPECTED_DOMAIN.items():
        match = re.search(rf"^{re.escape(name)}\s*=\s*(-?\d+(?:\.\d+)?)\s*$", source, re.MULTILINE)
        if not match or not math.isclose(float(match.group(1)), value):
            raise RuntimeError(
                f"Base script domain drift: expected {name}={value}, found "
                f"{match.group(1) if match else 'missing'}"
            )


def _patch_radius_and_artifact_names(source: str, experiment: Experiment) -> str:
    radius = experiment.radius_km
    text = source

    replacements = {
        "v33_r100km_singletarget_radiusstats_mse_apcp13p7cv_domain": experiment.experiment_tag,
        "r100km_singletarget_radiusstats_target_v33_apcp13p7cv_domain": experiment.target_output_tag,
        "v33_r100km_single_target_apcp13p7cv_domain": experiment.experiment_tag,
        "v33_r100km_singletarget_radiusstats_apcp13p7cv_domain": experiment.experiment_tag,
        "current_v33_r100km": f"current_{RUN_VERSION}_r{radius}km_{experiment.ratio_tag}",
        "V33 RADIUS-SENSITIVITY": "V34 COMPACT THRESHOLD-SENSITIVITY",
        "v33 radius-sensitivity": "v34 compact threshold-sensitivity",
        "V28 SINGLE-TARGET R100KM": f"V34 COMPACT SINGLE-TARGET R{radius}KM",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    # Radius language, functions, diagnostic names, and final feature suffixes.
    for old, new in (
        ("R100km", f"R{radius}km"),
        ("r100km", f"r{radius}km"),
        ("100-km", f"{radius}-km"),
        ("100 km", f"{radius} km"),
        ("100km", f"{radius}km"),
    ):
        text = text.replace(old, new)

    text = re.sub(
        r"^R40KM_TARGET_RADIUS_KM\s*=\s*\d+(?:\.\d+)?\s*$",
        f"R40KM_TARGET_RADIUS_KM = {float(radius):.1f}",
        text,
        flags=re.MULTILINE,
    )
    text = re.sub(
        r"^R40KM_FEATURE_SMOOTH_RADIUS_KM\s*=\s*\d+(?:\.\d+)?\s*$",
        f"R40KM_FEATURE_SMOOTH_RADIUS_KM = {float(radius):.1f}",
        text,
        flags=re.MULTILINE,
    )
    return text


def _patch_ratio_target_names(source: str, experiment: Experiment) -> str:
    radius = experiment.radius_km
    ratio_tag = experiment.ratio_column_tag
    text = source.replace("Obs_MRMS_FFG_Exceeded", f"Obs_MRMS_FFG_{ratio_tag}")
    text = text.replace("Target_MRMS_FFG_Exceeded", f"Target_MRMS_FFG_{ratio_tag}")
    text = re.sub(
        r"^TARGET_RATIO_THRESHOLD\s*=\s*\d+(?:\.\d+)?\s*$",
        f"TARGET_RATIO_THRESHOLD = {experiment.ratio_threshold:.1f}",
        text,
        flags=re.MULTILINE,
    )
    text = re.sub(
        r'^TRAINING_OBJECTIVE_LABEL\s*=\s*"[^"]+"\s*$',
        f'TRAINING_OBJECTIVE_LABEL = "r{radius}km_binary_mrms_ffg_{experiment.ratio_tag}_classification"',
        text,
        flags=re.MULTILINE,
    )
    return text


def _compact_contract_block(experiment: Experiment) -> str:
    source_features = source_summary_features()
    model_features = expected_model_features(experiment.radius_km)
    top_names = tuple(name for name, _value in TOP_20_REFERENCE_FEATURES)
    return f'''\n# ----------------------------------------------------------------------------------------------------
# V34 COMPACT FEATURE CONTRACT
# ----------------------------------------------------------------------------------------------------
# The top-20 ranking is frozen from the authoritative v33 Day-1 100-km model.
# Base families are re-expanded using the normal temporal/duration summaries and
# {experiment.radius_km}-km spatial Mean/Min/Max/Std. Simulated composite reflectivity is added.
COMPACT_REFERENCE_MODEL_PATH = {REFERENCE_MODEL_PATH!r}
COMPACT_REFERENCE_MODEL_SHA256 = {REFERENCE_MODEL_SHA256!r}
COMPACT_REFERENCE_FEATURES_PATH = {REFERENCE_FEATURES_PATH!r}
COMPACT_REFERENCE_FEATURES_SHA256 = {REFERENCE_FEATURES_SHA256!r}
COMPACT_REFERENCE_IMPORTANCE_METHOD = {REFERENCE_IMPORTANCE_METHOD!r}
COMPACT_TOP20_FEATURES = {_python_tuple(top_names)}
COMPACT_TOP20_BASE_FAMILIES = {_python_tuple(TOP_20_BASE_FAMILIES)}
COMPACT_SOURCE_SUMMARY_FEATURES = {_python_tuple(source_features)}
COMPACT_MODEL_FEATURE_NAMES = {_python_tuple(model_features)}
SIMULATED_REFLECTIVITY_BASE = {SIMULATED_REFLECTIVITY_BASE!r}

if len(COMPACT_MODEL_FEATURE_NAMES) != 164:
    raise RuntimeError(f"V34 compact predictor contract drift: {{len(COMPACT_MODEL_FEATURE_NAMES)}} != 164")
'''


def _patch_compact_feature_engineering(source: str, experiment: Experiment) -> str:
    text = source
    insertion_anchor = 'USE_PREV24H_FFG_EXCEEDANCE_FEATURES = True\n'
    text = _replace_exact(
        text,
        insertion_anchor,
        "USE_PREV24H_FFG_EXCEEDANCE_FEATURES = False\n" + _compact_contract_block(experiment),
        "previous-day feature flag",
    )
    text = _replace_exact(
        text,
        "REUSE_PRIOR_RAP_FEATURE_CHUNKS = True",
        "REUSE_PRIOR_RAP_FEATURE_CHUNKS = False",
        "prior RAP reuse flag",
    )

    # Add RAP simulated composite reflectivity as an instantaneous forecast-hour
    # family. REFC is an entire-atmosphere column maximum in the RAP archive.
    raw_anchor = '''        "MLCIN": (["CIN", "CINH"], ["PRESSUREFROMGROUNDLAYER"], 9000),

        # Boundary layer / soil / precip rate.'''
    raw_replacement = '''        "MLCIN": (["CIN", "CINH"], ["PRESSUREFROMGROUNDLAYER"], 9000),
        "Simulated_Composite_Reflectivity_dBZ": (
            ["REFC"], ["ENTIREATMOSPHERE", "ATMOSPHERE", "UNKNOWN"], None
        ),

        # Boundary layer / soil / precip rate.'''
    text = _replace_exact(text, raw_anchor, raw_replacement, "simulated reflectivity raw variable")

    smoothing_anchor = '''                if grid is not None:
                    smoothed_grid = uniform_filter(grid.astype(np.float32), size=3)
                    full_fields[v_name] = smoothed_grid
                    processed_dict[f"{v_name}_{fhr_name}"] = (
                        smoothed_grid[domain_mask][keep_mask_1d].astype(np.float32)
                    )

            # MMP-only ingredients.'''
    smoothing_replacement = '''                if grid is not None:
                    # Preserve dBZ values before the normal temporal/spatial summaries.
                    # Linear averaging of dBZ in the legacy 3x3 prefilter is not physical.
                    feature_grid = (
                        grid.astype(np.float32)
                        if v_name == SIMULATED_REFLECTIVITY_BASE
                        else uniform_filter(grid.astype(np.float32), size=3)
                    )
                    full_fields[v_name] = feature_grid
                    processed_dict[f"{v_name}_{fhr_name}"] = (
                        feature_grid[domain_mask][keep_mask_1d].astype(np.float32)
                    )

            # MMP-only ingredients.'''
    text = _replace_exact(text, smoothing_anchor, smoothing_replacement, "raw feature extraction")

    reflectivity_guard_anchor = "    # APCP running total features.\n"
    reflectivity_guard = '''    missing_reflectivity_hours = [
        int(fhr) for fhr in FORECAST_HOURS
        if not np.isfinite(
            np.asarray(
                processed_dict.get(
                    f"{SIMULATED_REFLECTIVITY_BASE}_fhr_{int(fhr):02d}h",
                    np.asarray([], dtype=np.float32),
                ),
                dtype=np.float32,
            )
        ).any()
    ]
    if missing_reflectivity_hours:
        raise RuntimeError(
            f"Required RAP REFC simulated composite reflectivity is missing for {date_str}: "
            f"valid-hour offsets {missing_reflectivity_hours}. Refusing to train a partial feature contract."
        )

'''
    text = _replace_exact(
        text,
        reflectivity_guard_anchor,
        reflectivity_guard + reflectivity_guard_anchor,
        "reflectivity availability guard",
    )

    text = _replace_regex_once(
        text,
        r"def _should_r40_smooth_feature\(col\):.*?\n\n\ndef apply_v28_hydro_feature_engineering",
        '''def _should_r40_smooth_feature(col):
    """Smooth only the frozen top-20 family summaries plus reflectivity."""
    return str(col) in COMPACT_SOURCE_SUMMARY_FEATURES


def apply_v28_hydro_feature_engineering''',
        "compact spatial-summary selector",
    )

    text = text.replace(
        "if USE_RADIUS_ONLY_MODEL_FEATURES and not _is_radius_smoothed_stat_feature(c):",
        "if USE_RADIUS_ONLY_MODEL_FEATURES and c not in COMPACT_MODEL_FEATURE_NAMES:",
    )
    text = text.replace(
        "if USE_RADIUS_ONLY_MODEL_FEATURES and _is_radius_smoothed_stat_feature(c):\n        return True",
        "if USE_RADIUS_ONLY_MODEL_FEATURES and c in COMPACT_MODEL_FEATURE_NAMES:\n        return True",
    )

    selection_anchor = '''    X_train_raw = df_train.drop(columns=[c for c in drop_cols if c in df_train.columns])
    feature_names = list(X_train_raw.columns)
    if USE_RADIUS_ONLY_MODEL_FEATURES:
        bad_model_features = [c for c in feature_names if not _is_radius_smoothed_stat_feature(c)]
        if bad_model_features:
            raise RuntimeError(
                "Radius-only feature guard failed. Raw/unsmoothed predictors would enter the model: "
                f"{bad_model_features[:50]}"
            )
    validate_single_training_target_configuration(feature_names)'''
    selection_replacement = '''    X_train_raw = df_train.drop(columns=[c for c in drop_cols if c in df_train.columns])
    available_model_features = set(X_train_raw.columns)
    missing_contract_features = [c for c in COMPACT_MODEL_FEATURE_NAMES if c not in available_model_features]
    unexpected_model_features = [c for c in X_train_raw.columns if c not in COMPACT_MODEL_FEATURE_NAMES]
    if missing_contract_features:
        raise RuntimeError(
            "V34 compact master is missing required predictors: "
            f"{missing_contract_features[:50]}"
        )
    if unexpected_model_features:
        raise RuntimeError(
            "V34 compact exclusion guard allowed predictors outside the contract: "
            f"{unexpected_model_features[:50]}"
        )
    feature_names = list(COMPACT_MODEL_FEATURE_NAMES)
    X_train_raw = X_train_raw[feature_names]
    validate_single_training_target_configuration(feature_names)'''
    text = _replace_exact(text, selection_anchor, selection_replacement, "model feature selection")

    text = _replace_exact(
        text,
        '''    X_test_raw = df_test.drop(columns=[c for c in drop_cols if c in df_test.columns])
    Y_test = df_test[TRAIN_TARGET_COLUMN].to_numpy(dtype=np.int8)''',
        '''    X_test_raw = df_test.drop(columns=[c for c in drop_cols if c in df_test.columns])
    missing_test_features = [c for c in feature_names if c not in X_test_raw.columns]
    if missing_test_features:
        raise RuntimeError(f"Formal test data are missing compact predictors: {missing_test_features[:50]}")
    X_test_raw = X_test_raw[feature_names]
    Y_test = df_test[TRAIN_TARGET_COLUMN].to_numpy(dtype=np.int8)''',
        "formal-test feature selection",
    )

    availability_anchor = '''        feature_names = clean_features
        if not feature_names:
            raise RuntimeError("All predictor columns were dropped by the feature availability filter.")
        X_train_raw = X_train_raw[feature_names]
        X_test_raw = X_test_raw[feature_names]
        validate_single_training_target_configuration(feature_names)'''
    availability_replacement = '''        if dropped_features:
            raise RuntimeError(
                "V34 compact feature contract contains unavailable or constant predictors; "
                f"refusing silent feature loss: {dropped_features}"
            )
        feature_names = list(COMPACT_MODEL_FEATURE_NAMES)
        X_train_raw = X_train_raw[feature_names]
        X_test_raw = X_test_raw[feature_names]
        validate_single_training_target_configuration(feature_names)'''
    text = _replace_exact(
        text, availability_anchor, availability_replacement, "feature availability enforcement"
    )
    return text


def _patch_metadata(source: str, experiment: Experiment) -> str:
    lines = source.splitlines(keepends=True)
    text_parts: list[str] = []
    for index, line in enumerate(lines):
        text_parts.append(line)
        if '"target_radius_km": float(R40KM_TARGET_RADIUS_KM),' not in line:
            continue
        indent = line[: len(line) - len(line.lstrip())]
        following = "".join(lines[index + 1 : index + 5])
        metadata_rows = []
        if '"target_ratio_threshold"' not in following:
            metadata_rows.append('"target_ratio_threshold": float(TARGET_RATIO_THRESHOLD),')
        metadata_rows.extend(
            [
                '"compact_predictor_count": len(COMPACT_MODEL_FEATURE_NAMES),',
                '"compact_top20_base_families": list(COMPACT_TOP20_BASE_FAMILIES),',
                '"includes_simulated_composite_reflectivity": True,',
                '"reference_model_sha256": COMPACT_REFERENCE_MODEL_SHA256,',
                '"reference_features_sha256": COMPACT_REFERENCE_FEATURES_SHA256,',
            ]
        )
        text_parts.extend(f"{indent}{row}\n" for row in metadata_rows)
    text = "".join(text_parts)
    text = text.replace(
        "had observed MRMS QPE / FFG exceedance",
        f"had observed MRMS QPE / FFG ratio >= {experiment.ratio_threshold:.1f}",
    )
    text = text.replace(
        'from xgboost import XGBClassifier, XGBRegressor',
        'from xgboost import XGBClassifier',
    )
    header = f'''# ======================================================================================
# GENERATED V34 COMPACT DAY-1 TRAINING SCRIPT
# Target: any observed MRMS/FFG ratio >= {experiment.ratio_threshold:.1f} within {experiment.radius_km} km
# Default threshold experiment: {experiment.ratio_threshold == DEFAULT_RATIO_THRESHOLD}
# Predictors: 164 (top-20 v33 base families + simulated composite reflectivity)
# Domain: lat 30.0..50.0, lon -105.0..-80.5 (east of Rockies, east coast excluded)
# Objective: XGBClassifier binary:logistic; grouped-date CV logloss
# Generated by: {Path(__file__).name}
# ======================================================================================

'''
    return header + text


def generate_source(experiment: Experiment) -> str:
    if not BASE_SCRIPT.exists():
        raise FileNotFoundError(f"Authoritative v33 base script not found: {BASE_SCRIPT}")
    source = BASE_SCRIPT.read_text(encoding="utf-8")
    _assert_base_domain(source)
    source = _patch_radius_and_artifact_names(source, experiment)
    source = _patch_ratio_target_names(source, experiment)
    source = _patch_compact_feature_engineering(source, experiment)
    source = _patch_metadata(source, experiment)
    validate_generated_source(source, experiment)
    return source


def validate_generated_source(source: str, experiment: Experiment) -> None:
    ast.parse(source)
    required = (
        f"TARGET_RATIO_THRESHOLD = {experiment.ratio_threshold:.1f}",
        f"R40KM_TARGET_RADIUS_KM = {float(experiment.radius_km):.1f}",
        f"R40KM_FEATURE_SMOOTH_RADIUS_KM = {float(experiment.radius_km):.1f}",
        f'TRAIN_TARGET_COLUMN = "{experiment.target_column}"',
        "COMPACT_MODEL_FEATURE_NAMES = ",
        "len(COMPACT_MODEL_FEATURE_NAMES) != 164",
        '"Simulated_Composite_Reflectivity_dBZ":',
        "Required RAP REFC simulated composite reflectivity is missing",
        'objective="binary:logistic"',
        "REUSE_PRIOR_RAP_FEATURE_CHUNKS = False",
        "USE_PREV24H_FFG_EXCEEDANCE_FEATURES = False",
    )
    missing = [token for token in required if token not in source]
    if missing:
        raise RuntimeError(f"Generated source failed v34 contract; missing {missing}")
    for name, value in EXPECTED_DOMAIN.items():
        if f"{name} = {value}" not in source:
            raise RuntimeError(f"Generated source changed domain contract: {name}")
    if "R40KM_TARGET_RADIUS_KM = 40.0" in source or "R40KM_TARGET_RADIUS_KM = 60.0" in source:
        raise RuntimeError("Generated source retained a discarded 40/60-km target")
    if "XGBRegressor" in source:
        raise RuntimeError("Generated source must remain XGBClassifier-only")


def write_generated_script(experiment: Experiment) -> Path:
    source = generate_source(experiment)
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    experiment.generated_script.write_text(source, encoding="utf-8")
    return experiment.generated_script


def manifest_complete(experiment: Experiment) -> bool:
    if not experiment.manifest.exists() or not experiment.master_parquet.exists():
        return False
    try:
        payload = json.loads(experiment.manifest.read_text(encoding="utf-8"))
    except Exception:
        return False
    if payload.get("target_column") != experiment.target_column:
        return False
    if not math.isclose(float(payload.get("target_radius_km", -1)), experiment.radius_km):
        return False
    if not math.isclose(
        float(payload.get("target_ratio_threshold", -1)), experiment.ratio_threshold
    ):
        return False
    if int(payload.get("compact_predictor_count", -1)) != 164:
        return False
    for key in ("model_path", "scaler_path", "feature_names_path", "results_path"):
        path = payload.get(key)
        if not path or not Path(path).exists() or Path(path).stat().st_size <= 0:
            return False
    return experiment.master_parquet.stat().st_size > 0


def run_experiment(experiment: Experiment, *, force: bool, dry_run: bool) -> int:
    script = write_generated_script(experiment)
    print("\n" + "=" * 110, flush=True)
    print(
        f"V34 compact | radius={experiment.radius_km} km | "
        f"MRMS/FFG >= {experiment.ratio_threshold:.1f}",
        flush=True,
    )
    print(f"Generated script: {script}", flush=True)
    print(f"Expected predictors: {len(expected_model_features(experiment.radius_km))}", flush=True)
    print("=" * 110, flush=True)

    if not force and manifest_complete(experiment):
        print("Complete manifest/master found; skipping. Use --force to retrain.", flush=True)
        return 0

    command = [PYTHON_EXE, str(script)]
    print("Command:", " ".join(shlex.quote(value) for value in command), flush=True)
    if dry_run:
        return 0

    env = os.environ.copy()
    env.setdefault("HAZARD_ML_FORCE_FRESH_RUN", "1")
    env.setdefault("HAZARD_ML_FORCE_RETRAIN_MODEL", "1")
    env.setdefault("HAZARD_ML_MASTER_COMBINE_BATCH_SIZE", "4")
    env.setdefault("HAZARD_ML_MASTER_SLIM_READ", "1")
    env.setdefault("HAZARD_ML_MASTER_STREAM_ROW_SAMPLE", "1")
    env.setdefault("HAZARD_ML_MASTER_MAX_TRAIN_ROWS", "800000")
    env.setdefault("HAZARD_ML_MASTER_MAX_TEST_ROWS", "200000")
    env.setdefault("HAZARD_ML_SKIP_MASTER_FEATURE_REENGINEERING", "1")
    env.setdefault("HAZARD_ML_LOCAL_MODEL_N_JOBS", "1")
    env.setdefault("HAZARD_ML_USE_RAY_FOR_DATA_EXTRACTION", "0")
    env.setdefault("HAZARD_ML_LOCAL_OPTUNA_XGB_TRIALS", "30")
    env.setdefault("HAZARD_ML_LOCAL_TUNE_MAX_TRAIN_ROWS_PER_FOLD", "300000")
    env.setdefault("HAZARD_ML_LOCAL_TUNE_MAX_VAL_ROWS_PER_FOLD", "100000")
    return subprocess.run(command, env=env, check=False).returncode


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--radii",
        nargs="+",
        type=int,
        default=list(SUPPORTED_RADII),
        choices=SUPPORTED_RADII,
    )
    parser.add_argument(
        "--ratios",
        nargs="+",
        type=float,
        default=list(SUPPORTED_RATIO_THRESHOLDS),
        choices=SUPPORTED_RATIO_THRESHOLDS,
    )
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    experiments = [
        Experiment(radius, ratio)
        for ratio in args.ratios
        for radius in args.radii
    ]
    for experiment in experiments:
        path = write_generated_script(experiment)
        print(
            f"generated {path} | target={experiment.target_column} | "
            f"predictors={len(expected_model_features(experiment.radius_km))}"
        )

    if args.generate_only:
        return 0

    for experiment in experiments:
        code = run_experiment(experiment, force=args.force, dry_run=args.dry_run)
        if code:
            return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
