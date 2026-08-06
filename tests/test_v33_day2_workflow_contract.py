from __future__ import annotations

import ast
import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
GENERATOR_PATH = ROOT / (
    "make_run_hazard_ml_v33_day2_radius_sensitivity_from_WORKING_v28_"
    "radiusstats_SLIMMASTER_ROWSAMPLE.py"
)
VIEWER_PATH = ROOT / "hazard_ml_v33_day2_verification_viewer.ipynb"
MEMSAFE_PREDICTION_BUILDER = ROOT / "build_v33day2_test_predictions_memsafe.py"


def _load_module(name: str, path: Path):
    if path.suffix == ".ipynb":
        notebook = json.loads(path.read_text(encoding="utf-8"))
        module = types.ModuleType(name)
        module.__file__ = str(path)
        sys.modules[name] = module
        tagged_cells = [
            cell
            for cell in notebook.get("cells", [])
            if cell.get("cell_type") == "code"
            and "day2-routing-core"
            in cell.get("metadata", {}).get("tags", [])
        ]
        assert len(tagged_cells) == 1
        source = "".join(tagged_cells[0].get("source", []))
        exec(compile(source, str(path), "exec"), module.__dict__)
        return module
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _viewer_source() -> tuple[dict, str]:
    notebook = json.loads(VIEWER_PATH.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
    )
    return notebook, source


def test_day2_generator_contract(tmp_path, monkeypatch):
    monkeypatch.setenv("HAZARD_ML_V33_DAY2_GENERATED_SCRIPT_DIR", str(tmp_path))
    project_dir = tmp_path / "fall_2025_ml_proj"
    project_dir.mkdir()
    monkeypatch.setenv("HAZARD_ML_PROJECT_DIR", str(project_dir))
    generator = _load_module("day2_generator_test", GENERATOR_PATH)
    assert generator.rap_init_date_for_event("20240620") == "20240619"
    generated = generator.generate_script(40)
    source = generated.read_text(encoding="utf-8")
    ast.parse(source)

    assert "FORECAST_HOURS = [0, 6, 12, 18, 24, 30, 36, 42, 48]" in source
    assert "TARGET_WINDOW_START_OFFSET_HOURS = 24" in source
    assert "RAP_INIT_DAY_OFFSET_FROM_CASE = -1" in source
    assert 'DAY2_CASE_DATE_CONTRACT = "event_valid_start_v2"' in source
    assert "def day2_event_date_to_rap_init_date(case_date):" in source
    assert "rap_init_date_str = day2_event_date_to_rap_init_date(date_str)" in source
    assert "download_nam_forecasts_osdf(rap_init_date_str, nam_dir)" in source
    assert '"RAP_Init_Date": np.full(num_keep, rap_init_date_str)' in source
    assert (
        'base_dt = datetime.strptime(str(date_str), "%Y%m%d") '
        "+ timedelta(hours=int(TARGET_VALID_START_HOUR))"
    ) in source
    assert "REUSE_PRIOR_RAP_FEATURE_CHUNKS = False" in source
    assert 'os.environ.get("HAZARD_ML_DAY2_RESUME_CACHED_CHUNKS", "1")' in source
    assert "USE_PREV24H_FFG_EXCEEDANCE_FEATURES = False" in source
    assert 'os.environ.get("HAZARD_ML_USE_RAY_FOR_DATA_EXTRACTION", "0")' in source
    assert "All requested daily chunks already exist; skipping domain preparation" in source
    assert "skipping RAP source preflight" in source
    assert 'TRAIN_TARGET_COLUMN = "Target_Day2_MRMS_FFG_Exceeded_R40km"' in source
    assert '"forecast_horizon": "day2"' in source
    assert '"24_30_36_42_48h"' in source
    assert "Forecast_APCP_Total_24to48h_mm" in source
    assert "Not used by the Day-2 MRMS>FFG-only target" in source
    assert "Target_MRMS_FFG_Exceeded_R40km" not in source
    assert str(project_dir) in source
    assert "/home/tyreekfrazier/ISU_Research/fall_2025_ml_proj" not in source
    assert f'PROJECT_ROOT = "{project_dir.parent}"' in source


def test_relocated_manifest_artifacts_are_complete(tmp_path, monkeypatch):
    project_dir = tmp_path / "fall_2025_ml_proj"
    model_dir = project_dir / "prob_flood_models"
    model_dir.mkdir(parents=True)
    monkeypatch.setenv("HAZARD_ML_PROJECT_DIR", str(project_dir))
    monkeypatch.setenv(
        "HAZARD_ML_V33_DAY2_GENERATED_SCRIPT_DIR",
        str(tmp_path / "generated"),
    )
    generator = _load_module("day2_generator_relocation_test", GENERATOR_PATH)

    radius = 40
    master = generator.candidate_paths_for_radius(radius)["master"][0]
    master.write_bytes(b"master")
    artifact_names = {
        "model_path": "model.pkl",
        "scaler_path": "scaler.pkl",
        "feature_names_path": "features.json",
        "results_path": "results.csv",
    }
    for name in artifact_names.values():
        (model_dir / name).write_bytes(b"artifact")
    manifest = {
        "target_column": generator.target_column(radius),
        "forecast_horizon": "day2",
        "rap_valid_offsets_h": generator.DAY2_VALID_OFFSETS_H,
        "target_window_start_offset_h": 24,
        "case_date_contract": generator.DAY2_CASE_DATE_CONTRACT,
        "rap_init_day_offset_from_case": -1,
        "target_radius_km": radius,
        **{
            key: f"/old/relocated/project/prob_flood_models/{name}"
            for key, name in artifact_names.items()
        },
    }
    manifest_path = generator.candidate_paths_for_radius(radius)["manifest"][0]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    assert generator.radius_is_complete(radius, verbose=False)


def test_day2_window_and_latest_wpc_revision():
    viewer = _load_module("day2_viewer_window_test", VIEWER_PATH)
    start, end = viewer.day2_valid_window("20240715")
    assert str(start) == "2024-07-15 12:00:00+00:00"
    assert str(end) == "2024-07-16 12:00:00+00:00"
    assert viewer.day2_observation_date("20240715") == "20240715"
    viewer_start, viewer_end = viewer.day2_viewer_valid_window("20240715")
    assert viewer_start == start
    assert viewer_end == end

    historical = viewer._day2_apply_historical_viewer_date_convention(
        pd.DataFrame(
            {
                "Date": ["20240620"],
                "Year": ["2024"],
                "Lat": [40.0],
                "Lon": [-95.0],
            }
        )
    )
    assert historical.loc[0, "Date"] == "20240620"
    assert historical.loc[0, "RAP_Init_Date"] == "20240619"
    assert historical.loc[0, "Year"] == "2024"
    pd.testing.assert_frame_equal(
        historical,
        viewer._day2_apply_historical_viewer_date_convention(historical),
    )

    frame = pd.DataFrame(
        {
            "ISSUE": [
                "2024-07-15T12:00:00Z",
                "2024-07-15T12:00:00Z",
                "2024-07-15T12:00:00Z",
            ],
            "EXPIRE": [
                "2024-07-16T12:00:00Z",
                "2024-07-16T12:00:00Z",
                "2024-07-16T12:00:00Z",
            ],
            "PRODISS": [
                "2024-07-15T08:00:00Z",
                "2024-07-15T08:00:00Z",
                "2024-07-15T20:00:00Z",
            ],
            "DAY": [1, 2, 2],
            "CATEGORY": ["MRGL", "SLGT", "MDT"],
        }
    )
    selected = viewer._day2_filter_wpc_to_valid_window(frame, "20240715")
    assert len(selected) == 1
    assert selected.iloc[0]["DAY"] == 2
    assert selected.iloc[0]["CATEGORY"] == "MDT"


def test_ufvs_proxy_expansion_is_40_km_and_separate_from_pp():
    viewer = _load_module("day2_viewer_proxy_test", VIEWER_PATH)
    lat = np.array([40.0, 40.0, 40.0])
    lon = np.array([-100.0, -99.7, -99.0])
    raw = np.array([True, False, False])
    expanded = viewer._day2_expand_mask(raw, lat, lon, radius_km=40.0)
    assert expanded.tolist() == [True, True, False]

    notebook, source = _viewer_source()
    assert "compute_paper_pp_ets_rows" in source
    assert "compute_paper_proxy_ets_rows" in source
    assert "PAPER_PROXY_EXPANSION_RADIUS_KM = 40.0" in source
    assert "PAPER_PROXY_OPTIONS" in source
    assert "PP_Any flood proxy" in source


def test_wpc_is_radius_independent_and_ml_has_four_radius_members():
    viewer = _load_module("day2_viewer_radius_test", VIEWER_PATH)
    valid = pd.DataFrame(
        {
            "Source": ["WPC ERO", "ML r40km", "ML r60km"],
            "Source Radius": ["wpc", 40, 60],
        }
    )
    assert viewer.assert_wpc_radius_independent(valid)

    invalid = pd.DataFrame(
        {
            "Source": ["WPC ERO", "WPC ERO"],
            "Radius_km": [40, 60],
        }
    )
    try:
        viewer.assert_wpc_radius_independent(invalid)
    except AssertionError:
        pass
    else:
        raise AssertionError("Radius-specific WPC rows were not rejected")

    _, source = _viewer_source()
    expected_specs = """MODEL_SPECS = [
    {"label": "r40km", "radius_km": 40},
    {"label": "r60km", "radius_km": 60},
    {"label": "r75km", "radius_km": 75},
    {"label": "r100km", "radius_km": 100},
]"""
    assert expected_specs in source
    assert "WPC is yielded once, not once per radius" in source
    assert 'params = {"type": "E", "d": "2"' in source


def test_full_day1_viewer_feature_surface_is_preserved():
    notebook, source = _viewer_source()
    assert len(notebook["cells"]) >= 69
    assert sum(len(cell.get("outputs", [])) for cell in notebook["cells"]) == 0
    required_symbols = [
        "build_predict_verify_realtime_multi_radius",
        "plot_realtime_giant_all_members_wpc_pp_ufvs",
        "plot_case_ml_wpc_pp_proxy",
        "compute_paper_pp_ets_rows",
        "compute_paper_proxy_ets_rows",
        "plot_grouped_violin",
        "SHAP_MODEL_LABEL",
        "shap_feature_definitions",
        "case_level_risk_area_counts_pp_confusion",
        "df_risk_area_error_vs_pp_common",
        "plot_reliability_diagram",
        "run_final_bs_ets_verification_plots",
    ]
    missing = [symbol for symbol in required_symbols if symbol not in source]
    assert not missing

    assert 'RUN_VERSION_TAG = "v33day2valid"' in source
    assert "hazard_ml_training_v33day2valid_r" in source
    assert "generated_v33_day2_radius_sensitivity_slimmaster_rowsample" in source
    assert "Obs_Day2_MRMS_FFG_Exceeded_Point" in source
    assert 'SHAP_MODEL_LABEL = "r100km"' in source
    assert '"d": "1"' not in source
    assert "WPC ERO Day 1" not in source


def test_day2_test_predictions_are_streamed_and_resumable():
    _, source = _viewer_source()
    assert "DAY2_PREDICT_SOURCE_BATCH_ROWS" in source
    assert 'os.environ.get("XGBFFP_DAY2_PREDICT_BATCH_ROWS", "12000")' in source
    assert "parquet.iter_batches(" in source
    assert "_day2_write_prediction_fragment" in source
    assert "_day2_consolidate_prediction_fragments" in source
    assert "Resumable fragments:" in source
    assert "del radius_pred_parts" in source
    assert '"prediction_writer_version": 2' in source
    assert 'base_columns.append("RAP_Init_Date")' in source
    assert 'valid_dates = event_dates.dt.strftime("%Y%m%d").to_numpy()' in source
    assert "event_dates + pd.Timedelta(days=1)" not in source

    # The bounded-memory override must be the final active definition before the
    # notebook starts building radius predictions.
    definitions = [
        index
        for index in range(len(source))
        if source.startswith("def build_or_load_radius_predictions(", index)
    ]
    assert len(definitions) >= 2
    call_site = source.index("radius_pred_parts = []")
    assert definitions[-1] < call_site


def test_day2_viewer_has_only_the_four_trained_model_members():
    _, source = _viewer_source()
    expected_specs = '''MODEL_SPECS = [
    {"label": "r40km", "radius_km": 40},
    {"label": "r60km", "radius_km": 60},
    {"label": "r75km", "radius_km": 75},
    {"label": "r100km", "radius_km": 100},
]'''
    assert expected_specs in source

    nonexistent_member_tokens = [
        "r60kmV2",
        "r100kmV2",
        "R60KM_V2",
        "R100KM_V2",
        "ML_r60V2_Prob",
        "ML_r60kmV2_Prob",
        "ML_r100kmV2_Prob",
        "_rt_standardize_v2_probability_column",
        "PAPER_EXCLUDED_V2_MODEL_LABELS",
    ]
    assert not [token for token in nonexistent_member_tokens if token in source]
    assert 'SHAP_MODEL_LABEL = "r100km"' in source
    assert '''PAPER_RT_REQUIRED_ML_COLS = [
    "ML_r40_Prob",
    "ML_r60_Prob",
    "ML_r75_Prob",
    "ML_r100_Prob",
]''' in source


def test_standalone_memsafe_prediction_builder_contract():
    source = MEMSAFE_PREDICTION_BUILDER.read_text(encoding="utf-8")
    ast.parse(source)
    assert "--batch-rows" in source
    assert "default=6000" in source
    assert "CORE_CACHE_COLUMNS" in source
    assert '"RAP_Init_Date"' in source
    assert "build_or_load_radius_predictions" in source
    assert "_normalize_cache" in source
    assert "_normalize_verification_grid" in source
    assert "process peak RSS" in source


def test_all_regular_python_cells_compile():
    notebook, _ = _viewer_source()
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "code":
            continue
        source = "\n".join(
            line
            for line in "".join(cell.get("source", [])).splitlines()
            if not line.lstrip().startswith(("%", "!"))
        )
        compile(source, f"{VIEWER_PATH}#cell-{index}", "exec")
