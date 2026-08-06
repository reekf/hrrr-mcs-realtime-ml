import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VIEWER = ROOT / (
    "hazard_ml_v33_radiusstats_WORKING_BASELINE_PLUS_VERIFICATION_SHAP_REALTIME_"
    "MULTIRADIUS_ENSEMBLE_WPC_VALIDFIX_METRICS_PREDICTORS_v18_PP_EXCLUSIVE_"
    "PROXY_CUMULATIVE_VIOLINS.ipynb"
)


def _source() -> str:
    notebook = json.loads(VIEWER.read_text())
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") in {"code", "markdown"}
    )


def test_day1_viewer_does_not_route_r60km_v2():
    source = _source()
    forbidden = (
        "r60kmV2",
        "R60KM_V2",
        "ML_r60V2_Prob",
        "ML_r60kmV2_Prob",
        "_rt_standardize_v2_probability_column",
    )
    assert not [token for token in forbidden if token in source]
    assert '{"label": "r60km", "radius_km": 60}' in source
    assert 'SHAP_MODEL_LABEL = "r100km"' in source


def test_day1_viewer_uses_official_wpc_pp_reader():
    source = _source()
    assert "from wpc_practically_perfect import" in source
    assert "replace_pp_with_official_wpc" in source
    assert "WPC_PP_2P5KM_CACHE_DIR" in source
    assert "WPC_PP_RISK_THRESHOLDS" in source
    assert "wpc_pp_threshold_for_forecast_label" in source
    assert "official 5/10/20/40%" in source


def test_day1_viewer_has_only_the_official_all_proxy_pp_field():
    source = _source()
    forbidden_individual_pp_fields = (
        "PP_MRMS",
        "PP_Stage IV",
        "PP_LSR/USGS",
        "PP_USGS",
        "PP_Flash LSR",
        "PP_Flood LSR",
    )
    assert not [token for token in forbidden_individual_pp_fields if token in source]
    assert 'TRUTH_DEFINITIONS = [\n    "Any flood proxy",\n]' in source
    assert "REALTIME_GIANT_PP_COLUMNS = [WPC_PP_COLUMN]" in source
