import json

import numpy as np
import pandas as pd
import xarray as xr

from mcs_lifetime_domains import (
    domain_from_robust_stats,
    domain_mask,
    filter_dataframe_to_domains,
    load_domains,
    save_domains,
)
from build_rap_mcs_lifetime_domains import _validate_viewer_grid_coverage


def _write_stats(path):
    times = pd.date_range("2024-06-16T12:00:00Z", periods=4, freq="h").tz_localize(None)
    ds = xr.Dataset(
        {
            "mcs_status": (("tracks", "times"), [[1, 1, 1, 1], [1, 1, 1, 0]]),
            "pf_mcsstatus": (("tracks", "times"), [[0, 0, 0, 0], [1, 1, 1, 0]]),
            "meanlat": (("tracks", "times"), [[35.0, 35.1, 35.2, 35.3], [44.0, 44.1, 44.2, np.nan]]),
            "meanlon": (("tracks", "times"), [[-80.0, -79.9, -79.8, -79.7], [-98.0, -97.9, -97.8, np.nan]]),
            "base_time": (("tracks", "times"), np.tile(times.values, (2, 1))),
            "ccs_area": (("tracks", "times"), np.full((2, 4), 50_000.0)),
        },
        coords={"tracks": [10, 20], "times": np.arange(4)},
    )
    ds.to_netcdf(path)


def test_anchor_selects_intended_mcs_and_uses_mcs_lifecycle(tmp_path):
    stats = tmp_path / "mcs_tracks_20240616.nc"
    _write_stats(stats)
    domain = domain_from_robust_stats(
        stats,
        date="20240616",
        anchor_lat=44.0,
        anchor_lon=-98.0,
        anchor_name="test anchor",
    )
    assert domain.track_number == 20
    assert domain.mcs_samples == 3
    assert domain.mcs_duration_hours == 3.0
    assert 44.0 < domain.center_lat < 44.2
    assert -98.0 < domain.center_lon < -97.8
    assert domain.selection_anchor == "test anchor"


def test_domain_mask_is_400_km_local_square(tmp_path):
    stats = tmp_path / "mcs_tracks_20240616.nc"
    _write_stats(stats)
    domain = domain_from_robust_stats(stats, date="20240616", anchor_lat=44.0, anchor_lon=-98.0)
    lat = np.array([domain.center_lat, domain.center_lat, domain.center_lat + 2.2])
    lon = np.array([domain.center_lon, domain.center_lon + 2.0, domain.center_lon])
    assert domain_mask(lat, lon, domain).tolist() == [True, True, False]

    # The map envelope must contain the projected square's corners as well as
    # its cardinal midpoints; otherwise Cartopy can clip the requested box.
    from pyproj import CRS, Transformer

    local = CRS.from_proj4(
        f"+proj=aeqd +lat_0={domain.center_lat} +lon_0={domain.center_lon} "
        "+datum=WGS84 +units=m +no_defs"
    )
    corner_lon, corner_lat = Transformer.from_crs(
        local, "EPSG:4326", always_xy=True
    ).transform(
        [-200_000.0, 200_000.0, 200_000.0, -200_000.0],
        [-200_000.0, -200_000.0, 200_000.0, 200_000.0],
    )
    assert min(corner_lon) >= domain.lon_min
    assert max(corner_lon) <= domain.lon_max
    assert min(corner_lat) >= domain.lat_min
    assert max(corner_lat) <= domain.lat_max


def test_dataframe_filter_and_json_round_trip(tmp_path):
    stats = tmp_path / "mcs_tracks_20240616.nc"
    _write_stats(stats)
    domain = domain_from_robust_stats(stats, date="20240616", anchor_lat=44.0, anchor_lon=-98.0)
    cache = save_domains([domain], tmp_path / "domains.json")
    domains = load_domains(cache)
    frame = pd.DataFrame(
        {
            "Date": ["20240616", "20240616"],
            "Lat": [domain.center_lat, domain.center_lat + 5.0],
            "Lon": [domain.center_lon, domain.center_lon],
            "value": [1, 2],
        }
    )
    selected = filter_dataframe_to_domains(frame, domains)
    assert selected["value"].tolist() == [1]
    assert selected["MCS_Domain_Box_km"].iloc[0] == 400.0
    payload = json.loads(cache.read_text())
    assert payload["schema_version"] == 2


def test_incomplete_historical_grid_is_explicitly_excluded(tmp_path):
    stats = tmp_path / "mcs_tracks_20240616.nc"
    _write_stats(stats)
    domain = domain_from_robust_stats(stats, date="20240616", anchor_lat=44.0, anchor_lon=-98.0)
    # Deliberately omit the western half of the exact MCS box.
    grid = pd.DataFrame(
        {
            "Date": ["20240616"] * 4,
            "Lat": [domain.center_lat - 1.0, domain.center_lat + 1.0] * 2,
            "Lon": [domain.center_lon, domain.center_lon + 1.0] * 2,
        }
    )
    path = tmp_path / "viewer.parquet"
    grid.to_parquet(path, index=False)
    accepted, excluded = _validate_viewer_grid_coverage(path, [domain])
    assert accepted == []
    assert excluded[0]["date"] == "20240616"
    assert "extends beyond" in excluded[0]["reason"]
