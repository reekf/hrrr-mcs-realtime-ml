#!/usr/bin/env python3
"""Lifetime-centered case domains derived from PyFLEXTRKR MCS tracks.

The verification contract is intentionally strict: a case is evaluated only
inside one 400 x 400 km azimuthal-equidistant box centered on the time-mean
centroid of its selected MCS.  The same row mask is applied after all
ML, WPC ERO, and official Practically Perfect fields have been mapped to the
common viewer grid.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import glob
import json
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np


DOMAIN_SCHEMA_VERSION = 2
DEFAULT_BOX_SIZE_KM = 400.0


def date8(value: object) -> str:
    text = "".join(ch for ch in str(value) if ch.isdigit())
    if len(text) < 8:
        raise ValueError(f"Could not parse YYYYMMDD from {value!r}")
    return text[:8]


@dataclass(frozen=True)
class MCSCaseDomain:
    date: str
    center_lat: float
    center_lon: float
    box_size_km: float
    track_index: int
    track_number: int
    mcs_samples: int
    mcs_duration_hours: float
    track_start_utc: str | None
    track_end_utc: str | None
    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float
    source_stats_path: str
    source_stats_mtime_utc: str
    center_method: str = "equal-time mean of PyFLEXTRKR MCS-stage feature centroids"
    selection_method: str = (
        "maximum MCS-stage duration; ties by integrated cold-cloud-shield area, "
        "then lowest track index"
    )
    schema_version: int = DOMAIN_SCHEMA_VERSION
    selection_anchor: str | None = None
    selection_anchor_lat: float | None = None
    selection_anchor_lon: float | None = None
    selection_anchor_distance_km: float | None = None

    @property
    def extent(self) -> list[float]:
        return [self.lon_min, self.lon_max, self.lat_min, self.lat_max]


def _finite_datetime_strings(values: np.ndarray) -> tuple[str | None, str | None]:
    times = np.asarray(values).reshape(-1)
    if not np.issubdtype(times.dtype, np.datetime64):
        return None, None
    good = times[~np.isnat(times)]
    if good.size == 0:
        return None, None
    start = np.min(good).astype("datetime64[s]").astype(str) + "Z"
    end = np.max(good).astype("datetime64[s]").astype(str) + "Z"
    return start, end


def _time_resolution_hours(base_time: np.ndarray) -> float:
    values = np.asarray(base_time)
    if not np.issubdtype(values.dtype, np.datetime64):
        return 1.0
    diffs = []
    for row in np.atleast_2d(values):
        valid = row[~np.isnat(row)]
        if valid.size > 1:
            hours = np.diff(valid).astype("timedelta64[s]").astype(float) / 3600.0
            diffs.extend(hours[np.isfinite(hours) & (hours > 0)].tolist())
    return float(np.median(diffs)) if diffs else 1.0


def _spherical_mean_longitude(lons: np.ndarray) -> float:
    radians = np.deg2rad(np.asarray(lons, dtype=float))
    angle = np.arctan2(np.mean(np.sin(radians)), np.mean(np.cos(radians)))
    lon = float(np.rad2deg(angle))
    return ((lon + 180.0) % 360.0) - 180.0


def _geographic_extent(center_lat: float, center_lon: float, box_size_km: float) -> list[float]:
    """Return the lon/lat envelope containing the exact local projected square."""
    from pyproj import CRS, Transformer

    half_m = float(box_size_km) * 500.0
    local = CRS.from_proj4(
        f"+proj=aeqd +lat_0={center_lat:.10f} +lon_0={center_lon:.10f} "
        "+datum=WGS84 +units=m +no_defs"
    )
    perimeter = np.linspace(-half_m, half_m, 101)
    x = np.concatenate(
        [perimeter, perimeter, np.full_like(perimeter, -half_m), np.full_like(perimeter, half_m)]
    )
    y = np.concatenate(
        [np.full_like(perimeter, -half_m), np.full_like(perimeter, half_m), perimeter, perimeter]
    )
    lon, lat = Transformer.from_crs(local, "EPSG:4326", always_xy=True).transform(x, y)
    return [float(np.min(lon)), float(np.max(lon)), float(np.min(lat)), float(np.max(lat))]


def domain_from_robust_stats(
    stats_path: str | Path,
    *,
    date: object | None = None,
    box_size_km: float = DEFAULT_BOX_SIZE_KM,
    anchor_lat: float | None = None,
    anchor_lon: float | None = None,
    anchor_name: str | None = None,
) -> MCSCaseDomain:
    """Select the primary PyFLEXTRKR MCS and return its lifetime-centered domain.

    ``mcs_status`` is the centering lifecycle.  It exists in both Tb-defined
    and robust PF track files and avoids losing a valid cloud track solely
    because model reflectivity misses the separate robust-MCS threshold.
    """
    import xarray as xr

    path = Path(stats_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    with xr.open_dataset(path, decode_times=True) as ds:
        status_name = "mcs_status" if "mcs_status" in ds else "pf_mcsstatus"
        required = [status_name, "meanlat", "meanlon", "base_time"]
        missing = [name for name in required if name not in ds]
        if missing:
            raise RuntimeError(f"{path} is missing PyFLEXTRKR variables: {missing}")
        status = np.asarray(ds[status_name].values)
        meanlat = np.asarray(ds["meanlat"].values, dtype=float)
        meanlon = np.asarray(ds["meanlon"].values, dtype=float)
        base_time = np.asarray(ds["base_time"].values)
        if status.ndim != 2 or meanlat.shape != status.shape or meanlon.shape != status.shape:
            raise RuntimeError(f"Unexpected MCS-track array shapes in {path}")
        qualifying = (status == 1) & np.isfinite(meanlat) & np.isfinite(meanlon)
        sample_counts = np.sum(qualifying, axis=1)
        if not np.any(sample_counts > 0):
            raise RuntimeError(f"No PyFLEXTRKR MCS-stage centroid samples found in {path}")
        if "ccs_area" in ds:
            area = np.asarray(ds["ccs_area"].values, dtype=float)
        else:
            area = np.asarray(ds.get("area", xr.zeros_like(ds["meanlat"])).values, dtype=float)
        integrated_area = np.nansum(np.where(qualifying, area, np.nan), axis=1)
        candidates = np.flatnonzero(sample_counts > 0)
        anchor_distances = np.full(status.shape[0], np.nan, dtype=float)
        if anchor_lat is not None and anchor_lon is not None:
            from pyproj import Geod

            geod = Geod(ellps="WGS84")
            for index in candidates:
                candidate_lat = float(np.mean(meanlat[index, qualifying[index]]))
                candidate_lon = _spherical_mean_longitude(meanlon[index, qualifying[index]])
                _, _, distance_m = geod.inv(float(anchor_lon), float(anchor_lat), candidate_lon, candidate_lat)
                anchor_distances[index] = float(distance_m / 1000.0)
            track_index = int(
                sorted(
                    candidates.tolist(),
                    key=lambda index: (
                        float(anchor_distances[index]),
                        -int(sample_counts[index]),
                        -float(integrated_area[index]),
                        int(index),
                    ),
                )[0]
            )
        else:
            track_index = int(
                sorted(
                    candidates.tolist(),
                    key=lambda index: (-int(sample_counts[index]), -float(integrated_area[index]), int(index)),
                )[0]
            )
        mask = qualifying[track_index]
        lats = meanlat[track_index, mask]
        lons = meanlon[track_index, mask]
        center_lat = float(np.mean(lats))
        center_lon = _spherical_mean_longitude(lons)
        dt_hours = _time_resolution_hours(base_time)
        mcs_samples = int(np.sum(mask))
        start_utc, end_utc = _finite_datetime_strings(base_time[track_index, mask])
        track_number = int(ds["tracks"].values[track_index]) if "tracks" in ds.coords else track_index + 1

    extent = _geographic_extent(center_lat, center_lon, box_size_km)
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
    return MCSCaseDomain(
        date=date8(date if date is not None else path.name),
        center_lat=center_lat,
        center_lon=center_lon,
        box_size_km=float(box_size_km),
        track_index=track_index,
        track_number=track_number,
        mcs_samples=mcs_samples,
        mcs_duration_hours=float(mcs_samples * dt_hours),
        track_start_utc=start_utc,
        track_end_utc=end_utc,
        lon_min=extent[0],
        lon_max=extent[1],
        lat_min=extent[2],
        lat_max=extent[3],
        source_stats_path=str(path),
        source_stats_mtime_utc=mtime,
        selection_method=(
            f"nearest lifetime-mean MCS centroid to {anchor_name or 'the supplied case anchor'}; "
            "ties by MCS-stage duration and integrated cold-cloud-shield area"
            if anchor_lat is not None and anchor_lon is not None
            else MCSCaseDomain.__dataclass_fields__["selection_method"].default
        ),
        selection_anchor=anchor_name,
        selection_anchor_lat=float(anchor_lat) if anchor_lat is not None else None,
        selection_anchor_lon=float(anchor_lon) if anchor_lon is not None else None,
        selection_anchor_distance_km=(
            float(anchor_distances[track_index]) if np.isfinite(anchor_distances[track_index]) else None
        ),
    )


def domain_mask(lat: Iterable[float], lon: Iterable[float], domain: MCSCaseDomain) -> np.ndarray:
    """Return the exact square mask in a local azimuthal-equidistant CRS."""
    from pyproj import CRS, Transformer

    lat_values = np.asarray(lat, dtype=float)
    lon_values = np.asarray(lon, dtype=float)
    if lat_values.shape != lon_values.shape:
        raise ValueError("Latitude and longitude arrays must have the same shape")
    local = CRS.from_proj4(
        f"+proj=aeqd +lat_0={domain.center_lat:.10f} +lon_0={domain.center_lon:.10f} "
        "+datum=WGS84 +units=m +no_defs"
    )
    x, y = Transformer.from_crs("EPSG:4326", local, always_xy=True).transform(lon_values, lat_values)
    half_m = domain.box_size_km * 500.0
    return (
        np.isfinite(lat_values)
        & np.isfinite(lon_values)
        & np.isfinite(x)
        & np.isfinite(y)
        & (np.abs(x) <= half_m)
        & (np.abs(y) <= half_m)
    )


def save_domains(
    domains: Iterable[MCSCaseDomain],
    path: str | Path,
    *,
    excluded_cases: Iterable[Mapping[str, object]] = (),
) -> Path:
    output = Path(path).expanduser()
    payload = {
        "schema_version": DOMAIN_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "domain_contract": (
            "400 x 400 km local azimuthal-equidistant square; equal-time mean of the "
            "primary PyFLEXTRKR MCS-stage centroid series"
        ),
        "cases": {domain.date: asdict(domain) for domain in domains},
        "excluded_cases": list(excluded_cases),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return output


def load_domains(path: str | Path) -> dict[str, MCSCaseDomain]:
    payload = json.loads(Path(path).expanduser().read_text())
    if int(payload.get("schema_version", -1)) != DOMAIN_SCHEMA_VERSION:
        raise RuntimeError(f"Unsupported MCS-domain schema in {path}")
    return {date8(key): MCSCaseDomain(**value) for key, value in payload.get("cases", {}).items()}


def resize_domain(domain: MCSCaseDomain, box_size_km: float) -> MCSCaseDomain:
    """Return the same MCS-centered domain with a different square size.

    Domain manifests preserve the expensive PyFLEXTRKR track selection and
    lifetime-mean center.  Plotting/verification box size is intentionally a
    cheap viewer-time choice, so changing it never requires rebuilding the
    track manifest.
    """
    size = float(box_size_km)
    if not np.isfinite(size) or size <= 0:
        raise ValueError(f"box_size_km must be a positive finite value, got {box_size_km!r}")
    extent = _geographic_extent(domain.center_lat, domain.center_lon, size)
    return replace(
        domain,
        box_size_km=size,
        lon_min=extent[0],
        lon_max=extent[1],
        lat_min=extent[2],
        lat_max=extent[3],
    )


def resize_domains(
    domains: Mapping[str, MCSCaseDomain], box_size_km: float
) -> dict[str, MCSCaseDomain]:
    """Resize every cached MCS center to one viewer-selected box size."""
    return {date8(key): resize_domain(domain, box_size_km) for key, domain in domains.items()}


def domains_with_complete_grid_coverage(
    dataframe,
    domains: Mapping[str, MCSCaseDomain],
    *,
    date_col: str = "Date",
    lat_col: str = "Lat",
    lon_col: str = "Lon",
) -> tuple[dict[str, MCSCaseDomain], list[dict[str, object]]]:
    """Keep only domains fully contained by each case's available ML grid.

    An adjustable box must never turn a missing portion of the historical grid
    into an implicit correct negative.  This inexpensive bounds check mirrors
    the manifest builder's coverage contract and is rerun whenever the viewer
    box size changes.
    """
    dates = dataframe[date_col].astype(str).str.replace(r"\D", "", regex=True).str[:8]
    accepted: dict[str, MCSCaseDomain] = {}
    excluded: list[dict[str, object]] = []
    for key, domain in domains.items():
        case_date = date8(key)
        grid = dataframe.loc[dates == case_date, [lat_col, lon_col]]
        if grid.empty:
            excluded.append({"date": case_date, "reason": "no rows in the historical ML viewer grid"})
            continue
        bounds = {
            "lon_min": float(grid[lon_col].min()),
            "lon_max": float(grid[lon_col].max()),
            "lat_min": float(grid[lat_col].min()),
            "lat_max": float(grid[lat_col].max()),
        }
        complete = (
            domain.lon_min >= bounds["lon_min"]
            and domain.lon_max <= bounds["lon_max"]
            and domain.lat_min >= bounds["lat_min"]
            and domain.lat_max <= bounds["lat_max"]
        )
        if not complete:
            excluded.append(
                {
                    "date": case_date,
                    "reason": f"{domain.box_size_km:g}-km MCS box extends beyond the available historical ML grid",
                    "domain_extent": domain.extent,
                    "viewer_grid_bounds": [
                        bounds["lon_min"], bounds["lon_max"], bounds["lat_min"], bounds["lat_max"]
                    ],
                }
            )
            continue
        inside = domain_mask(
            grid[lat_col].to_numpy(float), grid[lon_col].to_numpy(float), domain
        )
        if not np.any(inside):
            excluded.append(
                {"date": case_date, "reason": f"{domain.box_size_km:g}-km MCS box contains no ML rows"}
            )
            continue
        accepted[case_date] = domain
    return accepted, excluded


def discover_robust_stats(case_root: str | Path, date: object) -> Path:
    d = date8(date)
    pattern = str(
        Path(case_root).expanduser()
        / f"{d}_12z"
        / "pyflextrkr"
        / "output"
        / "stats"
        / "mcs_tracks_robust_*.nc"
    )
    robust = [Path(value) for value in sorted(glob.glob(pattern))]
    if len(robust) == 1:
        return robust[0]
    stats_dir = Path(case_root).expanduser() / f"{d}_12z" / "pyflextrkr" / "output" / "stats"
    tb_only = [
        path for path in sorted(stats_dir.glob("mcs_tracks_*.nc"))
        if not path.name.startswith(("mcs_tracks_pf_", "mcs_tracks_robust_"))
    ]
    if len(tb_only) == 1:
        return tb_only[0]
    raise RuntimeError(
        f"Expected one robust or Tb-defined MCS stats file for {d}; "
        f"found robust={len(robust)} Tb={len(tb_only)} in {stats_dir}"
    )


def filter_dataframe_to_domains(
    dataframe,
    domains: Mapping[str, MCSCaseDomain],
    *,
    date_col: str = "Date",
    lat_col: str = "Lat",
    lon_col: str = "Lon",
    strict: bool = True,
):
    """Attach provenance columns and retain only rows in each case's MCS box."""
    import pandas as pd

    frame = dataframe.copy()
    dates = frame[date_col].astype(str).str.replace(r"\D", "", regex=True).str[:8]
    frame[date_col] = dates
    missing = sorted(set(dates.unique()) - set(domains))
    if missing and strict:
        raise RuntimeError(
            "Missing PyFLEXTRKR lifetime domains for viewer cases: " + ", ".join(missing)
        )
    parts = []
    for case_date, group in frame.groupby(date_col, sort=False):
        domain = domains.get(str(case_date))
        if domain is None:
            continue
        keep = domain_mask(group[lat_col].to_numpy(float), group[lon_col].to_numpy(float), domain)
        selected = group.loc[keep].copy()
        if selected.empty:
            raise RuntimeError(f"MCS domain for {case_date} contains no viewer-grid rows")
        selected["MCS_Domain_Center_Lat"] = domain.center_lat
        selected["MCS_Domain_Center_Lon"] = domain.center_lon
        selected["MCS_Domain_Box_km"] = domain.box_size_km
        selected["MCS_Domain_Track"] = domain.track_number
        selected["MCS_Domain_Duration_h"] = domain.mcs_duration_hours
        selected["MCS_Domain_Source"] = domain.source_stats_path
        parts.append(selected)
    if not parts:
        return frame.iloc[0:0].copy()
    return pd.concat(parts, ignore_index=True)


def extent_for_case(date: object, domains: Mapping[str, MCSCaseDomain]) -> list[float]:
    d = date8(date)
    if d not in domains:
        raise KeyError(f"No MCS lifetime domain for {d}")
    return domains[d].extent
