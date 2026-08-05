from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd
import xarray as xr


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wpc_practically_perfect import (  # noqa: E402
    WPC_PP_MAX_MATCH_DISTANCE_KM,
    WPC_PP_COLUMN,
    WPC_PP_PRODUCT,
    read_wpc_pp_netcdf,
    replace_pp_with_official_wpc,
    sample_wpc_pp_dataset,
    wpc_pp_file_info,
)


def test_filename_uses_month_of_ending_valid_time():
    info = wpc_pp_file_info("20240630")
    assert info["month"] == "202407"
    assert info["filename"] == "pp_co_2p5km_s2024063012_e2024070112.nc"
    assert info["url"].endswith("/202407/" + info["filename"])


def test_crop_and_nearest_sampling_preserve_official_values():
    source_lat = np.array([[35.0, 35.0], [35.025, 35.025]], dtype=np.float32)
    source_lon = np.array([[-97.0, -96.97], [-97.0, -96.97]], dtype=np.float32)
    source_pp = np.array([[0.05, 0.10], [0.20, np.nan]], dtype=np.float32)
    dataset = xr.Dataset(
        data_vars={"PP": (("y", "x"), source_pp)},
        coords={
            "lat": (("y", "x"), source_lat),
            "lon": (("y", "x"), source_lon),
        },
    )

    target_lat = np.array([35.0001, 35.0249, 40.0])
    target_lon = np.array([-96.9701, -97.0001, -110.0])
    values, distances, metadata = sample_wpc_pp_dataset(
        dataset,
        target_lat,
        target_lon,
        max_match_distance_km=WPC_PP_MAX_MATCH_DISTANCE_KM,
    )

    np.testing.assert_allclose(values[:2], [0.10, 0.20])
    assert np.isnan(values[2])
    assert np.all(distances[:2] < 0.1)
    assert np.isnan(distances[2])
    assert metadata["matched_points"] == 2


def test_noaa_reference_open_order_preserves_native_2d_orientation_and_scale(tmp_path):
    source_lat = np.array([[30.0, 30.1, 30.2], [31.0, 31.1, 31.2]], dtype=np.float32)
    source_lon = np.array([[-100.0, -99.9, -99.8], [-100.1, -100.0, -99.9]], dtype=np.float32)
    source_pp = np.array([[0.0, 0.05, 0.10], [0.20, 0.40, 0.70]], dtype=np.float32)
    dataset = xr.Dataset(
        data_vars={"PP": (("y", "x"), source_pp)},
        coords={
            "lat": (("y", "x"), source_lat),
            "lon": (("y", "x"), source_lon),
        },
        attrs={"d_km": 2.539703},
    )
    path = tmp_path / "pp_co_2p5km_s2024061012_e2024061112.nc"
    dataset.to_netcdf(path)

    latitude, longitude, probability, metadata = read_wpc_pp_netcdf(path)

    np.testing.assert_array_equal(latitude, source_lat)
    np.testing.assert_array_equal(longitude, source_lon)
    np.testing.assert_array_equal(probability, source_pp)
    np.testing.assert_allclose(probability * 100.0, source_pp * 100.0, rtol=1e-7)
    assert metadata["reader"] in {"netCDF4.Dataset", "xarray.open_dataset"}
    assert metadata["source_grid_spacing_km"] == 2.539703


def test_out_of_range_probability_is_rejected():
    dataset = xr.Dataset(
        data_vars={"PP": (("y", "x"), np.array([[5.0]], dtype=np.float32))},
        coords={
            "lat": (("y", "x"), np.array([[35.0]], dtype=np.float32)),
            "lon": (("y", "x"), np.array([[-97.0]], dtype=np.float32)),
        },
    )
    try:
        sample_wpc_pp_dataset(dataset, [35.0], [-97.0])
    except ValueError as exc:
        assert "0..1" in str(exc)
    else:
        raise AssertionError("Out-of-range official probabilities must be rejected")


def test_replacement_drops_every_legacy_pp_field_and_records_provenance(tmp_path):
    dataframe = pd.DataFrame(
        {
            "Date": ["20240610", "20240610"],
            "Lat": [35.0, 35.1],
            "Lon": [-97.0, -96.9],
            "PP_Any flood proxy": [0.9, 0.8],
            "PP_MRMS > FFG": [0.7, 0.6],
            "PP_ROI_old": [1.0, 1.0],
        }
    )
    metadata = {
        "date": "20240610",
        "url": "https://example.test/official.nc",
        "filename": "official.nc",
        "valid_start": "2024061012",
        "valid_end": "2024061112",
    }
    with patch(
        "wpc_practically_perfect.load_wpc_pp_for_grid",
        return_value=(
            np.array([0.05, np.nan], dtype=np.float32),
            np.array([1.0, np.nan], dtype=np.float32),
            metadata,
        ),
    ):
        output, rows = replace_pp_with_official_wpc(dataframe, cache_dir=tmp_path)

    assert [column for column in output if column.startswith("PP_")] == [WPC_PP_COLUMN]
    assert output[WPC_PP_COLUMN].iloc[0] == np.float32(0.05)
    assert np.isnan(output[WPC_PP_COLUMN].iloc[1])
    assert output["WPC_PP_Available"].tolist() == [True, False]
    assert output["WPC_PP_Product"].dropna().unique().tolist() == [WPC_PP_PRODUCT]
    assert rows[0]["available"] is True
