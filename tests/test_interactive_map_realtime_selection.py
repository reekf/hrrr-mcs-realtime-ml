#!/usr/bin/env python3
"""Regression checks for realtime/verification map source selection."""

from pathlib import Path
import sys
import tempfile


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import generate_interactive_map_data as map_data


def test_verification_superset_is_preferred(tmp_path):
    date = "20260711"
    forecast = tmp_path / f"realtime_verified_v33_multiradius_r40_r60_r75_r100_{date}.parquet"
    verification = tmp_path / f"realtime_ufvs_verified_v33_multiradius_r40_r60_r75_r100_{date}.parquet"
    forecast.touch()
    verification.touch()

    original = map_data.REALTIME_DIR
    map_data.REALTIME_DIR = tmp_path
    try:
        assert map_data._preferred_realtime_forecast(date) == verification
    finally:
        map_data.REALTIME_DIR = original


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as directory:
        test_verification_superset_is_preferred(Path(directory))
    print("Interactive-map realtime source-selection regression check passed.")
