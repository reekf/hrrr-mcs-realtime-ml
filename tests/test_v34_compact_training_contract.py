import ast
import hashlib
import json
from pathlib import Path

import numpy as np

from day1_compact_feature_contract import (
    REFERENCE_FEATURES_PATH,
    REFERENCE_FEATURES_SHA256,
    REFERENCE_MODEL_PATH,
    REFERENCE_MODEL_SHA256,
    SIMULATED_REFLECTIVITY_BASE,
    TOP_20_BASE_FAMILIES,
    TOP_20_REFERENCE_FEATURES,
    base_family,
    expected_model_features,
    source_summary_features,
)
from make_run_hazard_ml_v34_compact_thresholds import (
    EXPECTED_DOMAIN,
    SUPPORTED_RADII,
    SUPPORTED_RATIO_THRESHOLDS,
    Experiment,
    generate_source,
)


def test_reference_top20_collapses_to_expected_compact_contract():
    assert len(TOP_20_REFERENCE_FEATURES) == 20
    assert len(TOP_20_BASE_FAMILIES) == 10
    assert len(source_summary_features()) == 41
    assert base_family("MLCAPE_0_6_12_18_24h_Max_R100km_Std") == "MLCAPE"

    for radius in SUPPORTED_RADII:
        features = expected_model_features(radius)
        assert len(features) == 164
        assert len(set(features)) == 164
        assert all(f"_R{radius}km_" in feature for feature in features)
        reflectivity = [feature for feature in features if SIMULATED_REFLECTIVITY_BASE in feature]
        assert len(reflectivity) == 16


def test_frozen_top20_matches_authoritative_v33_artifacts_when_available():
    model_path = Path(REFERENCE_MODEL_PATH)
    features_path = Path(REFERENCE_FEATURES_PATH)
    if not (model_path.exists() and features_path.exists()):
        return

    assert hashlib.sha256(model_path.read_bytes()).hexdigest() == REFERENCE_MODEL_SHA256
    assert hashlib.sha256(features_path.read_bytes()).hexdigest() == REFERENCE_FEATURES_SHA256

    import joblib

    model = joblib.load(model_path)
    payload = json.loads(features_path.read_text())
    names = payload["feature_names"] if isinstance(payload, dict) else payload
    ranking = sorted(
        zip(names, model.feature_importances_), key=lambda item: item[1], reverse=True
    )[:20]
    assert [name for name, _value in ranking] == [name for name, _value in TOP_20_REFERENCE_FEATURES]
    np.testing.assert_allclose(
        [value for _name, value in ranking],
        [value for _name, value in TOP_20_REFERENCE_FEATURES],
        rtol=0,
        atol=5e-8,
    )


def test_generator_builds_only_the_six_requested_binary_experiments():
    experiments = [
        Experiment(radius, ratio)
        for ratio in SUPPORTED_RATIO_THRESHOLDS
        for radius in SUPPORTED_RADII
    ]
    assert len(experiments) == 6
    assert SUPPORTED_RADII == (75, 100)
    assert SUPPORTED_RATIO_THRESHOLDS == (1.0, 1.5, 2.0)
    assert len({experiment.target_column for experiment in experiments}) == 6
    assert len({experiment.experiment_tag for experiment in experiments}) == 6
    assert not any(experiment.radius_km in {40, 60} for experiment in experiments)


def test_every_generated_script_preserves_domain_target_and_feature_contract():
    for ratio in SUPPORTED_RATIO_THRESHOLDS:
        for radius in SUPPORTED_RADII:
            experiment = Experiment(radius, ratio)
            source = generate_source(experiment)
            ast.parse(source)

            assert f"TARGET_RATIO_THRESHOLD = {ratio:.1f}" in source
            assert f"R40KM_TARGET_RADIUS_KM = {radius:.1f}" in source
            assert f'TRAIN_TARGET_COLUMN = "{experiment.target_column}"' in source
            assert "exceeded = valid & (ratio >= float(TARGET_RATIO_THRESHOLD))" in source
            assert "XGBRegressor" not in source
            assert 'objective="binary:logistic"' in source
            assert "REUSE_PRIOR_RAP_FEATURE_CHUNKS = False" in source
            assert "USE_PREV24H_FFG_EXCEEDANCE_FEATURES = False" in source
            assert "c in COMPACT_MODEL_FEATURE_NAMES" in source
            assert "c not in COMPACT_MODEL_FEATURE_NAMES" in source
            assert "Required RAP REFC simulated composite reflectivity is missing" in source
            assert "if v_name == SIMULATED_REFLECTIVITY_BASE" in source
            for name, value in EXPECTED_DOMAIN.items():
                assert f"{name} = {value}" in source


def test_generated_neighborhood_geometry_distinguishes_75_from_100_km():
    from scipy.spatial import cKDTree

    event_counts = {}
    for radius in SUPPORTED_RADII:
        source = generate_source(Experiment(radius, 1.0))
        parsed = ast.parse(source)
        functions = [
            node
            for node in parsed.body
            if isinstance(node, ast.FunctionDef)
            and (
                node.name.startswith("_latlon_to_unit_xyz_for_r")
                or node.name.startswith("build_r")
                and node.name.endswith("km_fractional_target_arrays")
            )
        ]
        namespace = {
            "np": np,
            "cKDTree": cKDTree,
            "R40KM_TARGET_RADIUS_KM": float(radius),
        }
        exec(compile(ast.Module(body=functions, type_ignores=[]), "<geometry-test>", "exec"), namespace)
        build = next(
            value
            for name, value in namespace.items()
            if name.startswith("build_r") and name.endswith("km_fractional_target_arrays")
        )

        # At 40 N, 0.94 degrees longitude is about 80 km. The positive first
        # point must influence the second point only in the 100-km target.
        _fraction, _neighbors, events = build(
            np.asarray([1, 0]),
            np.asarray([40.0, 40.0]),
            np.asarray([-98.0, -97.06]),
        )
        event_counts[radius] = int(events[1])

    assert event_counts == {75: 0, 100: 1}
