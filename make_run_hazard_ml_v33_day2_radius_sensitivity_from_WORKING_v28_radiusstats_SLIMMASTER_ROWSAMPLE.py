#!/usr/bin/env python3
"""
Generate and optionally run four separate Day-2 XGBoost classifiers.

Contract
--------
Case date V is the start date of the event/verification window.

* The original 0-24 h predictor families are retained.
* Day-2 RAP is initialized on V-1 at 09Z.
* Separate 24-48 h predictor summaries are added from that RAP's f27-f51
  (valid offsets 24, 30, 36, 42, and 48 h relative to V-1 12Z).
* The binary target is MRMS QPE >= FFG during V 12Z through V+1 12Z,
  identical to the Day-1 verification window for the same case.
* Target/feature radius pairs are R40, R60, R75, and R100 by default.
* Day-1 chunks, masters, models, aliases, and manifests are never reused.

The known-working v28 radiusstats script remains the extraction/training source
of truth.  This generator first applies the existing v33 radius conversion,
then applies the explicit Day-2 horizon conversion below.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import statistics
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

SCRIPT_DIR = Path(
    os.environ.get("HAZARD_ML_SCRIPT_DIR", Path(__file__).resolve().parent)
)
REMOTE_PROJECT_DIR = Path(
    "/home/tyreekfrazier/ISU_Research/fall_2025_ml_proj"
)
LOCAL_PROJECT_DIR = Path(
    "/home/tyreekfrazier/ISU_Research_LOCAL_RUN/fall_2025_ml_proj"
)
if os.environ.get("HAZARD_ML_PROJECT_DIR"):
    PROJECT_DIR = Path(os.environ["HAZARD_ML_PROJECT_DIR"])
elif REMOTE_PROJECT_DIR.is_dir():
    PROJECT_DIR = REMOTE_PROJECT_DIR
elif LOCAL_PROJECT_DIR.is_dir():
    PROJECT_DIR = LOCAL_PROJECT_DIR
else:
    PROJECT_DIR = REMOTE_PROJECT_DIR

# The shared radius generator reads these settings during import.
os.environ.setdefault("HAZARD_ML_SCRIPT_DIR", str(SCRIPT_DIR))
os.environ.setdefault("HAZARD_ML_PROJECT_DIR", str(PROJECT_DIR))

import make_run_hazard_ml_v33_radius_sensitivity_from_WORKING_v28_radiusstats_SLIMMASTER_ROWSAMPLE as basegen


MODEL_DIR = PROJECT_DIR / "prob_flood_models"
LOCAL_FALLBACK_ROOT = Path(
    os.environ.get(
        "HAZARD_ML_LOCAL_FALLBACK_ROOT",
        "/home/tyreekfrazier/ISU_Research_LOCAL_FALLBACK",
    )
)
BASE_V28_RADIUSSTATS_SCRIPT = Path(
    os.environ.get(
        "HAZARD_ML_BASE_V28_RADIUSSTATS_SCRIPT",
        SCRIPT_DIR / "hazard_ml_training_v28_r100km_singletarget_radiusstats_regression_MEMSAFE_V3.py",
    )
)
GEN_DIR = Path(
    os.environ.get(
        "HAZARD_ML_V33_DAY2_GENERATED_SCRIPT_DIR",
        SCRIPT_DIR / "generated_v33_day2_radius_sensitivity_slimmaster_rowsample",
    )
)

RUN_VERSION = os.environ.get(
    "HAZARD_ML_V33_DAY2_RUN_VERSION", "v33day2valid"
)
DEFAULT_RADII = "40 60 75 100"
RADIUS_LIST = [
    int(float(value))
    for value in os.environ.get("RADIUS_LIST", DEFAULT_RADII).replace(",", " ").split()
    if value.strip()
]
FEATURE_RADIUS_FOLLOWS_TARGET = (
    os.environ.get("FEATURE_RADIUS_FOLLOWS_TARGET", "1").strip().lower()
    in {"1", "true", "yes", "y"}
)
FORCE_RERUN = os.environ.get("FORCE_RERUN", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
}
DRY_RUN = os.environ.get("DRY_RUN", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
}
PYTHON_EXE = os.environ.get("PYTHON", sys.executable)

DAY2_VALID_OFFSETS_H = [0, 6, 12, 18, 24, 30, 36, 42, 48]
DAY2_TARGET_START_OFFSET_H = 24
DAY2_TARGET_WINDOW_H = 24
DAY2_RAP_INIT_DAY_OFFSET = -1
DAY2_CASE_DATE_CONTRACT = "event_valid_start_v2"
# Conservative fresh-run estimates based on the purged first-pass Day-2 files.
DAY2_FRESH_CHUNK_ESTIMATE_BYTES = 23 * 1024 ** 3
DAY2_FRESH_MASTER_ESTIMATE_BYTES = 16 * 1024 ** 3


def rap_init_date_for_event(case_date: str) -> str:
    """Return the preceding RAP initialization date for event case V."""
    match = re.search(r"(20\d{6}|19\d{6})", str(case_date))
    if match is None:
        raise ValueError(f"Could not parse YYYYMMDD from case date {case_date!r}")
    return (
        datetime.strptime(match.group(1), "%Y%m%d")
        + timedelta(days=DAY2_RAP_INIT_DAY_OFFSET)
    ).strftime("%Y%m%d")


def exp_tag(radius: int) -> str:
    return (
        f"{RUN_VERSION}_r{int(radius)}km_singletarget_radiusstats_"
        "mse_apcp13p7cv_domain"
    )


def target_output_tag(radius: int) -> str:
    return (
        f"r{int(radius)}km_singletarget_radiusstats_target_"
        f"{RUN_VERSION}_apcp13p7cv_domain"
    )


def target_column(radius: int) -> str:
    return f"Target_Day2_MRMS_FFG_Exceeded_R{int(radius)}km"


def local_fallback_dir(radius: int) -> Path:
    return LOCAL_FALLBACK_ROOT / f"prob_flood_models_{exp_tag(radius)}"


def _replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"Day-2 patch {label!r} expected one exact source match, found {count}."
        )
    return text.replace(old, new, 1)


def _sub_once(text: str, pattern: str, replacement: str, label: str) -> str:
    out, count = re.subn(pattern, replacement, text, count=1, flags=re.S)
    if count != 1:
        raise RuntimeError(
            f"Day-2 patch {label!r} expected one regex source match, found {count}."
        )
    return out


def _day2_forecast_summary_function() -> str:
    return r'''def add_forecast_hour_summary_features(table):
    """Keep the established 0-24 h summaries and add separate 24-48 h summaries."""
    hour_re = re.compile(
        r"^(?P<base>.+)_fhr_(?P<hour>00|06|12|18|24|30|36|42|48)h$"
    )
    groups = {}
    for col in list(table.columns if isinstance(table, pd.DataFrame) else table.keys()):
        match = hour_re.match(str(col))
        if not match:
            continue
        cstr = str(col)
        blocked = {
            "Date", "Lat", "Lon", "Year", TARGET_COLUMN, TRAIN_TARGET_COLUMN,
            R40KM_FRACTION_COLUMN, R40KM_TARGET_COUNT_COLUMN,
            R40KM_TARGET_EVENT_COUNT_COLUMN,
        }
        if cstr in blocked or any(
            cstr.startswith(prefix) for prefix in HYDRO_TARGET_PREFIXES
        ):
            continue
        groups.setdefault(match.group("base"), []).append(
            (int(match.group("hour")), col)
        )

    def _put(col, values):
        table[col] = np.asarray(values, dtype=np.float32)

    windows = (
        ("0_6_12_18_24h", {0, 6, 12, 18, 24}),
        ("24_30_36_42_48h", {24, 30, 36, 42, 48}),
    )
    for base, hour_cols in groups.items():
        by_hour = {hour: col for hour, col in hour_cols}
        for hour_tag, wanted in windows:
            if not wanted.issubset(by_hour):
                continue
            arrays = [
                np.asarray(table[by_hour[hour]], dtype=np.float32)
                for hour in sorted(wanted)
            ]
            for stat_name, stat in (
                ("Mean", "mean"), ("Min", "min"), ("Max", "max"), ("Std", "std")
            ):
                values = _nan_stat_no_warning(arrays, stat)
                if values is not None:
                    _put(f"{base}_{hour_tag}_{stat_name}", values)
    return table
'''


def _inject_day2_apcp_features(text: str) -> str:
    old_return = '''    return table


def add_forecast_hour_summary_features(table):'''
    late_apcp = '''    # Day-2 APCP summaries are distinct from the retained Day-1 summaries.
    late_interval_cols = [
        "APCP_Interval_24_30h", "APCP_Interval_30_36h",
        "APCP_Interval_36_42h", "APCP_Interval_42_48h",
    ]
    if all(_has(c) for c in late_interval_cols):
        late = [_arr(c) for c in late_interval_cols]
        for stat_name, stat in [("Mean", "mean"), ("Min", "min"), ("Max", "max"), ("Std", "std")]:
            vals = _nan_stat_no_warning(late, stat)
            if vals is not None:
                _put(f"Forecast_APCP_6hIntervals_24to48h_mm_{stat_name}", vals)
        _put("Forecast_APCP_Max_6h_Window_24to48h_mm", _nan_stat_no_warning(late, "max"))
        _put(
            "Forecast_APCP_Max_12h_Window_24to48h_mm",
            _nan_stat_no_warning(
                [late[0] + late[1], late[1] + late[2], late[2] + late[3]], "max"
            ),
        )
        _put("Forecast_APCP_Total_24to48h_mm", np.sum(np.stack(late, axis=0), axis=0))
    elif all(_has(c) for c in ["APCP_RunTotal_0_24h", "APCP_RunTotal_0_48h"]):
        _put(
            "Forecast_APCP_Total_24to48h_mm",
            np.clip(_arr("APCP_RunTotal_0_48h") - _arr("APCP_RunTotal_0_24h"), 0, None),
        )

    return table


def add_forecast_hour_summary_features(table):'''
    return _replace_once(
        text, old_return, late_apcp, "24-48 h APCP period summaries"
    )


def _inject_day2_ratio_features(text: str) -> str:
    marker = '''    return table


def _is_deprecated_or_raw_period_predictor(col):'''
    code = '''    # Separate Day-2 QPF/FFG ratios; do not mix them into Day-1 summaries.
    late_ratio_cols = []
    late_pairs = [
        ("Forecast_APCP_Max_6h_Window_24to48h_mm", "Guide_FFG_06h_mm",
         "Forecast_APCP_Max_6h_Window_24to48h_to_Guidance_FFG_06h_Ratio"),
        ("Forecast_APCP_Max_12h_Window_24to48h_mm", "Guide_FFG_12h_mm",
         "Forecast_APCP_Max_12h_Window_24to48h_to_Guidance_FFG_12h_Ratio"),
        ("Forecast_APCP_Total_24to48h_mm", "Guide_FFG_24h_mm",
         "Forecast_APCP_24to48h_Total_to_Guidance_FFG_24h_Ratio"),
    ]
    for apcp_col, ffg_col, out_col in late_pairs:
        if _has(apcp_col) and _has(ffg_col):
            _put(
                out_col,
                _arr(apcp_col) / np.maximum(_arr(ffg_col), np.float32(0.01)),
            )
            late_ratio_cols.append(out_col)
    if late_ratio_cols:
        late_arrays = [_arr(c) for c in late_ratio_cols]
        for stat_name, stat in [("Mean", "mean"), ("Min", "min"), ("Max", "max"), ("Std", "std")]:
            vals = _nan_stat_no_warning(late_arrays, stat)
            if vals is not None:
                _put(
                    f"Forecast_APCP_to_Guidance_FFG_Ratio_24to48h_Across_6h12h24h_{stat_name}",
                    vals,
                )

    return table


def _is_deprecated_or_raw_period_predictor(col):'''
    return _replace_once(text, marker, code, "24-48 h APCP/FFG ratios")


def apply_day2_contract(source: str, radius: int) -> str:
    """Convert one radius-specific v33 source string to the separate Day-2 contract."""
    radius = int(radius)
    text = source
    text = text.replace(
        '{"Date", "Lat", "Lon", "Year",',
        '{"Date", "RAP_Init_Date", "Lat", "Lon", "Year",',
    )
    text = text.replace(
        '"Date", "Lat", "Lon", "Year",\n        TARGET_COLUMN',
        '"Date", "RAP_Init_Date", "Lat", "Lon", "Year",\n        TARGET_COLUMN',
    )
    text = text.replace(
        '["Date", "Lat", "Lon", TARGET_COLUMN',
        '["Date", "RAP_Init_Date", "Lat", "Lon", TARGET_COLUMN',
    )

    text = _replace_once(
        text,
        "FORECAST_HOURS = [0, 6, 12, 18, 24]",
        f"FORECAST_HOURS = {DAY2_VALID_OFFSETS_H!r}",
        "forecast valid offsets",
    )
    text = _replace_once(
        text,
        "TARGET_VALID_WINDOW_HOURS = 24",
        (
            f"TARGET_VALID_WINDOW_HOURS = {DAY2_TARGET_WINDOW_H}\n"
            f"TARGET_WINDOW_START_OFFSET_HOURS = {DAY2_TARGET_START_OFFSET_H}\n"
            f"RAP_INIT_DAY_OFFSET_FROM_CASE = {DAY2_RAP_INIT_DAY_OFFSET}\n"
            f'DAY2_CASE_DATE_CONTRACT = "{DAY2_CASE_DATE_CONTRACT}"'
        ),
        "target window constants",
    )
    text = _replace_once(
        text,
        "def extract_unique_dates(csv_list):",
        '''def day2_event_date_to_rap_init_date(case_date):
    """Map event-valid case date V to the preceding Day-2 RAP initialization."""
    match = re.search(r"(20\\d{6}|19\\d{6})", str(case_date))
    if match is None:
        raise ValueError(f"Could not parse YYYYMMDD from case date {case_date!r}")
    event_date = datetime.strptime(match.group(1), "%Y%m%d")
    return (
        event_date + timedelta(days=int(RAP_INIT_DAY_OFFSET_FROM_CASE))
    ).strftime("%Y%m%d")


def extract_unique_dates(csv_list):''',
        "event date to RAP initialization helper",
    )
    text = _replace_once(
        text,
        '''    try:
        init_dt = datetime.strptime(str(date_str), "%Y%m%d") + timedelta(hours=int(GUIDANCE_FFG_INIT_RELATIVE_HOUR))
    except Exception:
        init_dt = datetime.strptime(str(date_str)[:8], "%Y%m%d") + timedelta(hours=int(GUIDANCE_FFG_INIT_RELATIVE_HOUR))
''',
        '''    rap_init_date = day2_event_date_to_rap_init_date(date_str)
    init_dt = datetime.strptime(rap_init_date, "%Y%m%d") + timedelta(
        hours=int(GUIDANCE_FFG_INIT_RELATIVE_HOUR)
    )
''',
        "guidance FFG uses preceding RAP initialization",
    )
    text = _replace_once(
        text,
        "REUSE_PRIOR_RAP_FEATURE_CHUNKS = True",
        (
            "# Day-1 chunks stop at valid+24 and cannot supply Day-2 predictors.\n"
            "REUSE_PRIOR_RAP_FEATURE_CHUNKS = False\n"
            "DAY2_RESUME_FROM_CACHED_CHUNKS = str(\n"
            '    os.environ.get("HAZARD_ML_DAY2_RESUME_CACHED_CHUNKS", "1")\n'
            ').strip().lower() in {"1", "true", "yes", "y"}'
        ),
        "disable prior Day-1 chunk reuse",
    )
    text = _replace_once(
        text,
        '''def process_daily_pixel_data(date_str, nam_dir, domain_vars, is_test_set=False):
    lats_2d, lons_2d, domain_mask, lats_1d, lons_1d = domain_vars
''',
        '''def process_daily_pixel_data(date_str, nam_dir, domain_vars, is_test_set=False):
    # date_str is event-valid date V. Day-2 predictors come from RAP V-1 09Z.
    rap_init_date_str = day2_event_date_to_rap_init_date(date_str)
    lats_2d, lons_2d, domain_mask, lats_1d, lons_1d = domain_vars
''',
        "separate event and RAP dates in daily extraction",
    )
    text = _replace_once(
        text,
        "    files_dict = download_nam_forecasts_osdf(date_str, nam_dir)",
        "    files_dict = download_nam_forecasts_osdf(rap_init_date_str, nam_dir)",
        "download RAP from preceding day",
    )
    text = text.replace(
        "validate_nam_forecast_file(req_path, date_str, req_fhr)",
        "validate_nam_forecast_file(req_path, rap_init_date_str, req_fhr)",
    )
    text = text.replace(
        'validate_nam_forecast_file(files_dict.get("fhr_00h", ""), date_str, 0)',
        'validate_nam_forecast_file(files_dict.get("fhr_00h", ""), rap_init_date_str, 0)',
    )
    text = _replace_once(
        text,
        '''    processed_dict = {
        "Date": np.full(num_keep, date_str),
''',
        '''    processed_dict = {
        "Date": np.full(num_keep, date_str),
        "RAP_Init_Date": np.full(num_keep, rap_init_date_str),
''',
        "persist event and RAP dates",
    )
    text = text.replace(
        '["Date", "Lat", "Lon", TARGET_COLUMN, TRAIN_TARGET_COLUMN,',
        '["Date", "RAP_Init_Date", "Lat", "Lon", TARGET_COLUMN, TRAIN_TARGET_COLUMN,',
    )
    text = _replace_once(
        text,
        "USE_PREV24H_FFG_EXCEEDANCE_FEATURES = True",
        (
            "# A previous Day-2 target would extend beyond the current RAP initialization.\n"
            "# Keep this optional base fallback disabled to prevent future-data leakage.\n"
            "USE_PREV24H_FFG_EXCEEDANCE_FEATURES = False"
        ),
        "disable previous-target leakage fallback",
    )
    text = _replace_once(
        text,
        "USE_RAY_FOR_DATA_EXTRACTION = True",
        (
            "USE_RAY_FOR_DATA_EXTRACTION = str(\n"
            '    os.environ.get("HAZARD_ML_USE_RAY_FOR_DATA_EXTRACTION", "0")\n'
            ').strip().lower() in {"1", "true", "yes", "y"}'
        ),
        "make extraction mode environment-controlled",
    )
    text = _sub_once(
        text,
        (
            r"def fetch_iem_flash_flood_reports_pixel"
            r"\(date_str, lats_1d, lons_1d, domain_mask_2d, lats_2d\):"
            r".*?\ndef _download_to_path"
        ),
        '''def fetch_iem_flash_flood_reports_pixel(
    date_str, lats_1d, lons_1d, domain_mask_2d, lats_2d
):
    """Not used by the Day-2 MRMS>FFG-only target; avoid an unrelated IEM request."""
    return np.zeros(len(lats_1d), dtype=np.int8)


def _download_to_path''',
        "disable unrelated LSR retrieval",
    )

    point_old = "Obs_MRMS_FFG_Exceeded_Point"
    point_new = "Obs_Day2_MRMS_FFG_Exceeded_Point"
    train_old = f"Target_MRMS_FFG_Exceeded_R{radius}km"
    train_new = target_column(radius)
    fraction_old = f"Obs_MRMS_FFG_Exceeded_R{radius}km_Fraction"
    fraction_new = f"Obs_Day2_MRMS_FFG_Exceeded_R{radius}km_Fraction"
    count_old = f"Obs_MRMS_FFG_R{radius}km_NeighborCount"
    count_new = f"Obs_Day2_MRMS_FFG_R{radius}km_NeighborCount"
    event_count_old = f"Obs_MRMS_FFG_Exceeded_R{radius}km_EventCount"
    event_count_new = f"Obs_Day2_MRMS_FFG_Exceeded_R{radius}km_EventCount"
    for old, new in (
        (train_old, train_new),
        (fraction_old, fraction_new),
        (event_count_old, event_count_new),
        (count_old, count_new),
        (point_old, point_new),
    ):
        text = text.replace(old, new)

    text = _sub_once(
        text,
        r"def add_forecast_hour_summary_features\(table\):.*?\n\n\ndef add_apcp_to_guidance_ffg_ratio_features\(table\):",
        _day2_forecast_summary_function()
        + "\n\ndef add_apcp_to_guidance_ffg_ratio_features(table):",
        "separate early/late instantaneous summaries",
    )
    text = _inject_day2_apcp_features(text)
    text = _inject_day2_ratio_features(text)

    extraction_anchor = '''    domain_vars = _prepare_domain_vars_for_extraction(all_dates)

    if USE_RAY_FOR_DATA_EXTRACTION:'''
    extraction_resume = '''    existing_chunk_paths = get_existing_daily_chunk_paths(all_dates)
    if len(existing_chunk_paths) == len(all_dates):
        print(
            "All requested daily chunks already exist; skipping domain preparation "
            "and feature extraction."
        )
        return combine_daily_chunks_to_master(all_dates)

    domain_vars = _prepare_domain_vars_for_extraction(all_dates)

    if USE_RAY_FOR_DATA_EXTRACTION:'''
    text = _replace_once(
        text,
        extraction_anchor,
        extraction_resume,
        "skip feature setup when all daily chunks are cached",
    )
    text = _replace_once(
        text,
        '''    sample_path = None
    for sample_date in tqdm(sample_candidate_dates, desc="Preparing RAP domain sample", unit="date"):
        sample_path = ensure_nam_forecast_file(sample_date, 0, RAP_DIR)
        if sample_path is not None and validate_nam_forecast_file(sample_path, sample_date, 0):
''',
        '''    sample_path = None
    for sample_date in tqdm(sample_candidate_dates, desc="Preparing RAP domain sample", unit="date"):
        rap_sample_date = day2_event_date_to_rap_init_date(sample_date)
        sample_path = ensure_nam_forecast_file(rap_sample_date, 0, RAP_DIR)
        if sample_path is not None and validate_nam_forecast_file(
            sample_path, rap_sample_date, 0
        ):
''',
        "domain sample uses preceding RAP date",
    )

    preflight_anchor = '''    all_dates = preflight_filter_rap_source_coverage(all_dates)

    if not all_dates:'''
    preflight_resume = '''    cached_chunk_dates = []
    if DAY2_RESUME_FROM_CACHED_CHUNKS and os.path.isdir(PIXEL_CHUNK_DIR):
        for chunk_name in os.listdir(PIXEL_CHUNK_DIR):
            match = re.fullmatch(r"pixel_features_(\\d{8})\\.parquet", chunk_name)
            if match and daily_chunk_exists(match.group(1)):
                cached_chunk_dates.append(match.group(1))
        cached_chunk_dates = sorted(set(cached_chunk_dates))

    if cached_chunk_dates:
        print(
            f"Resume mode: using {len(cached_chunk_dates)} completed daily chunk dates "
            "as the established Day-2 case set; skipping RAP source preflight."
        )
        all_dates = cached_chunk_dates
    else:
        all_dates = preflight_filter_rap_source_coverage(all_dates)

    if not all_dates:'''
    text = _replace_once(
        text,
        preflight_anchor,
        preflight_resume,
        "resume directly from cached Day-2 case dates",
    )
    text = _replace_once(
        text,
        '''    for d in tqdm(list(all_dates), desc="RAP source preflight", unit="date"):
        missing = []
        statuses = {}

        for fhr in MIN_REQUIRED_FORECAST_HOURS:
            ok, status, url = _rap_source_probe_ok(d, fhr)
''',
        '''    for d in tqdm(list(all_dates), desc="RAP source preflight", unit="date"):
        missing = []
        statuses = {}
        rap_init_date = day2_event_date_to_rap_init_date(d)

        for fhr in MIN_REQUIRED_FORECAST_HOURS:
            ok, status, url = _rap_source_probe_ok(rap_init_date, fhr)
''',
        "preflight preceding RAP date",
    )
    text = _replace_once(
        text,
        '''        rows.append({
            "date": str(d),
            "complete": complete,
''',
        '''        rows.append({
            "date": str(d),
            "event_valid_date": str(d),
            "rap_init_date": rap_init_date,
            "complete": complete,
''',
        "preflight date provenance",
    )
    text = _replace_once(
        text,
        '''    for d in all_dates:
        try:
            di = int(d)
            old_ok = int(RAP_THREDDS_OLD_START) <= di <= int(RAP_THREDDS_OLD_END)
            aws_ok = di >= int(RAP_AWS_PUBLIC_ARCHIVE_START)
''',
        '''    for d in all_dates:
        try:
            rap_init_date = day2_event_date_to_rap_init_date(d)
            di = int(rap_init_date)
            old_ok = int(RAP_THREDDS_OLD_START) <= di <= int(RAP_THREDDS_OLD_END)
            aws_ok = di >= int(RAP_AWS_PUBLIC_ARCHIVE_START)
''',
        "direct availability uses preceding RAP date",
    )

    # Ensure the new late-window engineered fields receive neighborhood statistics.
    smooth_anchor = '''        "Forecast_APCP_to_Guidance_FFG_Ratio_Across_6h12h24h_",
        "PrevDay_MRMS_FFG_",'''
    smooth_replacement = '''        "Forecast_APCP_to_Guidance_FFG_Ratio_Across_6h12h24h_",
        "Forecast_APCP_6hIntervals_24to48h_mm_",
        "Forecast_APCP_Max_6h_Window_24to48h_mm",
        "Forecast_APCP_Max_12h_Window_24to48h_mm",
        "Forecast_APCP_Total_24to48h_mm",
        "Forecast_APCP_Max_6h_Window_24to48h_to_Guidance_FFG_06h_Ratio",
        "Forecast_APCP_Max_12h_Window_24to48h_to_Guidance_FFG_12h_Ratio",
        "Forecast_APCP_24to48h_Total_to_Guidance_FFG_24h_Ratio",
        "Forecast_APCP_to_Guidance_FFG_Ratio_24to48h_",
        "PrevDay_MRMS_FFG_",'''
    text = _replace_once(
        text, smooth_anchor, smooth_replacement, "late-feature smoothing tokens"
    )
    text = _replace_once(
        text,
        'if re.search(r"_0_6_12_18_24h_(Mean|Min|Max|Std)$", c):',
        (
            'if re.search(\n'
            '        r"_(?:0_6_12_18_24|24_30_36_42_48)h_(Mean|Min|Max|Std)$", c\n'
            "    ):"
        ),
        "instantaneous summary smoothing regex",
    )

    # Add machine-readable horizon provenance to both feature metadata and manifest.
    provenance_anchor = '''        "target_radius_km": float(R40KM_TARGET_RADIUS_KM),'''
    provenance = '''        "target_radius_km": float(R40KM_TARGET_RADIUS_KM),
        "forecast_horizon": "day2",
        "case_date_definition": "V is event-valid window start date",
        "rap_case_date_definition": "RAP_Init_Date is V-1 at 09Z",
        "rap_init_day_offset_from_case": int(RAP_INIT_DAY_OFFSET_FROM_CASE),
        "case_date_contract": DAY2_CASE_DATE_CONTRACT,
        "rap_valid_offsets_h": list(FORECAST_HOURS),
        "rap_file_forecast_hours": [int(h) + int(RAP_FILE_FHR_OFFSET) for h in FORECAST_HOURS],
        "target_window_start_offset_h": int(TARGET_WINDOW_START_OFFSET_HOURS),
        "target_valid_window_h": int(TARGET_VALID_WINDOW_HOURS),'''
    text = text.replace(provenance_anchor, provenance)

    text = text.replace(
        "Build observed FFG-exceedance target for a 12z-to-12z case.",
        (
            "Build the Day-2 observed FFG-exceedance target from event date V "
            "12Z through V+1 12Z."
        ),
    )
    text = text.replace(
        "Uses hourly observed MRMS QPE ending at 13z..12z next day.",
        "Uses hourly observed MRMS QPE ending at V 13Z through V+1 12Z.",
    )

    old_header_end = (
        "# ======================================================================================\n\n"
    )
    day2_header = (
        "# DAY-2 CONTRACT: event case V; RAP init V-1 09Z; predictors valid "
        "V-1 12Z through V+1 12Z; target valid V 12Z through V+1 12Z.\n"
        "# Day-1 0-24 h summaries are retained and distinct 24-48 h summaries are added.\n"
        "# ======================================================================================\n\n"
    )
    text = text.replace(old_header_end, day2_header, 1)
    return text


def generated_script_path(radius: int) -> Path:
    return GEN_DIR / (
        f"hazard_ml_training_{RUN_VERSION}_r{int(radius)}km_"
        "singletarget_radiusstats_MEMSAFE.py"
    )


def generate_script(radius: int) -> Path:
    if not BASE_V28_RADIUSSTATS_SCRIPT.exists():
        raise FileNotFoundError(
            f"Base working v28 radiusstats script not found: {BASE_V28_RADIUSSTATS_SCRIPT}"
        )
    source = BASE_V28_RADIUSSTATS_SCRIPT.read_text(encoding="utf-8")
    if "TRAIN_TARGET_COLUMN" not in source or "FORECAST_HOURS" not in source:
        raise RuntimeError("Base file does not match the expected v28 radiusstats training source.")

    # Reuse the proven radius conversion but supply Day-2-unique version/path globals.
    basegen.RUN_VERSION = RUN_VERSION
    basegen.FEATURE_RADIUS_FOLLOWS_TARGET = FEATURE_RADIUS_FOLLOWS_TARGET
    radius_source = basegen.replace_working_v28_radiusstats_names(source, int(radius))
    radius_source = basegen.inject_slim_master_combine(radius_source)
    patched = apply_day2_contract(radius_source, int(radius))
    source_project_root = REMOTE_PROJECT_DIR.parent
    active_project_root = PROJECT_DIR.parent
    if active_project_root != source_project_root:
        patched = patched.replace(
            f"{source_project_root}/",
            f"{active_project_root}/",
        )
        patched = patched.replace(
            f'PROJECT_ROOT = "{source_project_root}"',
            f'PROJECT_ROOT = "{active_project_root}"',
        )

    out = generated_script_path(radius)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(patched, encoding="utf-8")
    return out


def candidate_paths_for_radius(radius: int) -> dict[str, list[Path]]:
    radius = int(radius)
    roots = [MODEL_DIR, local_fallback_dir(radius)]
    exp = exp_tag(radius)
    return {
        "manifest": [root / f"active_artifacts_{exp}.json" for root in roots],
        "model": [root / f"current_{RUN_VERSION}_r{radius}km_XGBoost_model.pkl" for root in roots],
        "scaler": [root / f"current_{RUN_VERSION}_r{radius}km_scaler.pkl" for root in roots],
        "features": [root / f"current_{RUN_VERSION}_r{radius}km_feature_names.json" for root in roots],
        "results": [
            root / f"prob_model_test_results_localoptuna_{exp}.csv" for root in roots
        ],
        "master": [
            PROJECT_DIR
            / (
                "pixel_domain_forecasts_rap09z_iem_mrms_ffg_"
                f"{target_output_tag(radius)}.parquet"
            )
        ],
    }


def _existing(paths: Iterable[Path]) -> list[Path]:
    return [path for path in paths if path.exists() and path.stat().st_size > 0]


def manifest_complete(path: Path, radius: int) -> bool:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if manifest.get("target_column") != target_column(radius):
        return False
    if manifest.get("forecast_horizon") != "day2":
        return False
    if manifest.get("case_date_contract") != DAY2_CASE_DATE_CONTRACT:
        return False
    if int(manifest.get("rap_init_day_offset_from_case", 999)) != -1:
        return False
    if list(manifest.get("rap_valid_offsets_h", [])) != DAY2_VALID_OFFSETS_H:
        return False
    if int(manifest.get("target_window_start_offset_h", -1)) != 24:
        return False
    try:
        if abs(float(manifest.get("target_radius_km")) - float(radius)) > 1e-6:
            return False
    except Exception:
        return False
    for key in ("model_path", "scaler_path", "feature_names_path", "results_path"):
        artifact = manifest.get(key)
        if not artifact:
            return False
        artifact_path = Path(artifact)
        relocated_path = path.parent / artifact_path.name
        if not artifact_path.is_file() and relocated_path.is_file():
            artifact_path = relocated_path
        if not artifact_path.is_file() or artifact_path.stat().st_size <= 0:
            return False
    return True


def daily_chunk_dir(radius: int) -> Path:
    return PROJECT_DIR / (
        "pixel_daily_chunks_rap09z_iem_mrms_ffg_"
        f"{target_output_tag(radius)}"
    )


def _directory_file_bytes(path: Path) -> int:
    if not path.is_dir():
        return 0
    return sum(
        item.stat().st_size
        for item in path.iterdir()
        if item.is_file()
    )


def _format_gib(value: int | float) -> str:
    return f"{float(value) / (1024 ** 3):.1f} GiB"


def estimate_remaining_disk(radius: int) -> dict[str, int | float]:
    """Estimate final-master growth from completed Day-2 radius peers."""
    radius = int(radius)
    chunk_bytes = _directory_file_bytes(daily_chunk_dir(radius))
    ratios = []
    peer_chunk_sizes = []
    peer_master_sizes = []
    for peer in (40, 60, 75, 100):
        peer_master = candidate_paths_for_radius(peer)["master"][0]
        peer_chunks = _directory_file_bytes(daily_chunk_dir(peer))
        if peer_chunks > 0:
            peer_chunk_sizes.append(peer_chunks)
        if peer_master.is_file() and peer_master.stat().st_size > 0:
            peer_size = peer_master.stat().st_size
            peer_master_sizes.append(peer_size)
            if peer_chunks > 0:
                ratios.append(peer_size / peer_chunks)

    if peer_chunk_sizes:
        estimated_final_chunks = max(
            chunk_bytes, int(statistics.median(peer_chunk_sizes))
        )
    else:
        estimated_final_chunks = max(
            chunk_bytes, int(DAY2_FRESH_CHUNK_ESTIMATE_BYTES)
        )

    if chunk_bytes > 0 and ratios:
        estimated_master = int(chunk_bytes * statistics.median(ratios))
    elif peer_master_sizes:
        estimated_master = int(statistics.median(peer_master_sizes))
    else:
        estimated_master = int(DAY2_FRESH_MASTER_ESTIMATE_BYTES)

    master = candidate_paths_for_radius(radius)["master"][0]
    partial = Path(f"{master}.tmp")
    partial_bytes = partial.stat().st_size if partial.is_file() else 0
    existing_master_bytes = master.stat().st_size if master.is_file() else 0
    remaining_chunks = max(0, estimated_final_chunks - chunk_bytes)
    remaining_master = max(
        0, estimated_master - max(partial_bytes, existing_master_bytes)
    )
    net_growth = remaining_chunks + remaining_master
    safety_bytes = 5 * 1024 ** 3
    recommended_free = net_growth + safety_bytes
    free_bytes = shutil.disk_usage(PROJECT_DIR).free
    return {
        "chunk_bytes": chunk_bytes,
        "estimated_final_chunk_bytes": estimated_final_chunks,
        "estimated_master_bytes": estimated_master,
        "partial_bytes": partial_bytes,
        "net_growth_bytes": net_growth,
        "recommended_free_bytes": recommended_free,
        "free_bytes": free_bytes,
    }


def print_resume_preflight(radius: int) -> None:
    radius = int(radius)
    chunk_dir = daily_chunk_dir(radius)
    chunk_count = (
        sum(1 for item in chunk_dir.iterdir() if item.is_file() and item.suffix == ".parquet")
        if chunk_dir.is_dir()
        else 0
    )
    estimate = estimate_remaining_disk(radius)
    print(f"r{radius}km resume preflight:")
    print(
        f"  cached daily feature chunks: {chunk_count} "
        f"({_format_gib(estimate['chunk_bytes'])})"
    )
    print(
        "  estimated completed daily chunks: "
        f"{_format_gib(estimate['estimated_final_chunk_bytes'])}"
    )
    print(
        "  estimated completed master: "
        f"{_format_gib(estimate['estimated_master_bytes'])}"
    )
    if estimate["partial_bytes"]:
        print(
            "  interrupted master temporary file: "
            f"{_format_gib(estimate['partial_bytes'])}"
        )
        print(
            "  note: daily features are reusable; the partial monolithic master "
            "cannot be appended safely and will be recombined from cached chunks"
        )
    print(
        "  estimated additional disk growth: "
        f"{_format_gib(estimate['net_growth_bytes'])}"
    )
    print(
        "  recommended free space (includes 5 GiB safety margin): "
        f"{_format_gib(estimate['recommended_free_bytes'])}"
    )
    print(f"  currently free: {_format_gib(estimate['free_bytes'])}")
    if estimate["free_bytes"] < estimate["recommended_free_bytes"]:
        raise RuntimeError(
            f"Insufficient free space for r{radius}km: "
            f"{_format_gib(estimate['free_bytes'])} available, "
            f"{_format_gib(estimate['recommended_free_bytes'])} recommended."
        )


def radius_is_complete(radius: int, verbose: bool = True) -> bool:
    candidates = candidate_paths_for_radius(radius)
    manifests = _existing(candidates["manifest"])
    complete = (
        bool(_existing(candidates["master"]))
        and any(manifest_complete(path, radius) for path in manifests)
    )
    if verbose:
        print(
            f"r{int(radius)}km Day-2 artifacts: "
            f"{'complete' if complete else 'incomplete'}"
        )
    return complete


def run_radius(radius: int) -> int:
    script = generate_script(radius)
    print("\n" + "=" * 100)
    print(f"Day-2 radius r{int(radius)}km | {exp_tag(radius)}")
    print(f"Generated script: {script}")
    print("=" * 100, flush=True)
    if not FORCE_RERUN and radius_is_complete(radius):
        print("Skipping complete radius. Set FORCE_RERUN=1 to retrain.", flush=True)
        return 0
    print_resume_preflight(radius)
    if DRY_RUN:
        print("DRY_RUN=1; generated and validated source, training was not started.")
        return 0

    env = os.environ.copy()
    env.setdefault("HAZARD_ML_FORCE_FRESH_RUN", "0")
    env.setdefault("HAZARD_ML_FORCE_RETRAIN_MODEL", "1")
    env.setdefault("HAZARD_ML_MASTER_COMBINE_BATCH_SIZE", "2")
    env.setdefault("HAZARD_ML_MASTER_COMBINE_SLIM_WRITE", "1")
    env.setdefault("HAZARD_ML_MASTER_SLIM_READ", "1")
    env.setdefault("HAZARD_ML_MASTER_STREAM_ROW_SAMPLE", "1")
    env.setdefault("HAZARD_ML_MASTER_MAX_TRAIN_ROWS", "800000")
    env.setdefault("HAZARD_ML_MASTER_MAX_TEST_ROWS", "200000")
    env.setdefault("HAZARD_ML_SKIP_MASTER_FEATURE_REENGINEERING", "1")
    env.setdefault("HAZARD_ML_LOCAL_MODEL_N_JOBS", "1")
    env.setdefault("HAZARD_ML_USE_RAY_FOR_DATA_EXTRACTION", "0")
    env.setdefault("HAZARD_ML_RAY_MAX_IN_FLIGHT", "2")
    # Preserve the base training default unless the runner/user supplies another budget.
    env.setdefault("HAZARD_ML_LOCAL_OPTUNA_XGB_TRIALS", "30")
    env.setdefault("HAZARD_ML_LOCAL_TUNE_MAX_TRAIN_ROWS_PER_FOLD", "300000")
    env.setdefault("HAZARD_ML_LOCAL_TUNE_MAX_VAL_ROWS_PER_FOLD", "100000")

    command = [PYTHON_EXE, str(script)]
    print("Command:", " ".join(shlex.quote(value) for value in command), flush=True)
    return int(subprocess.run(command, env=env).returncode)


def main() -> int:
    print("Separate v33 Day-2 four-radius XGBoost workflow")
    print(
        "Case date V is the event-valid start; "
        "RAP initialization is V-1 at 09Z"
    )
    print(f"RAP valid offsets: {DAY2_VALID_OFFSETS_H}")
    print("RAP file forecast hours: [3, 9, 15, 21, 27, 33, 39, 45, 51]")
    print("Target/verification window: case V 12Z through V+1 12Z")
    print(f"Radii: {RADIUS_LIST}")
    for radius in RADIUS_LIST:
        return_code = run_radius(radius)
        if return_code:
            return return_code
    print("\nAll requested Day-2 radii are complete or generated successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
