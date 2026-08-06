"""Download and map the official WPC 2.5-km Practically Perfect analysis.

The WPC files cover CONUS on a curvilinear Lambert grid.  This module preserves
the published PP probabilities by using nearest-neighbor sampling onto a viewer
grid after cropping the source arrays to the requested geographic extent.  The
official PP risk categories use 5/10/20/40-percent breaks, matching WPC's NOAA
reference plotter; they are intentionally different from the Day-1 ERO forecast
breaks of 5/15/40/70 percent.
"""

from __future__ import annotations

from datetime import datetime, timedelta
import hashlib
import json
import os
from pathlib import Path
import tempfile
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import warnings

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import xarray as xr


WPC_PP_BASE_URL = "https://ftp-wpc.ncep.noaa.gov/ffair/PP_2p5km"
WPC_PP_COLUMN = "PP_Any flood proxy"
WPC_PP_PRODUCT = "WPC official PP_2p5km"
WPC_PP_MAPPING_VERSION = "noaa-reference-reader-v3"
WPC_PP_NATIVE_SPACING_KM = 2.539703
WPC_PP_MAX_MATCH_DISTANCE_KM = 1.8 * WPC_PP_NATIVE_SPACING_KM
WPC_PP_RISK_THRESHOLDS = (
    (0.05, "Marginal"),
    (0.10, "Slight"),
    (0.20, "Moderate"),
    (0.40, "High"),
)
# Forecast-category labels retained by the viewer.  Each label maps to the
# corresponding official PP category threshold, rather than reusing the
# forecast probability threshold on the PP field.
WPC_PP_THRESHOLD_BY_FORECAST_LABEL = {
    ">5%": 0.05,
    ">15%": 0.10,
    ">40%": 0.20,
    ">70%": 0.40,
}
EARTH_RADIUS_KM = 6371.2


class WPCPracticallyPerfectUnavailable(RuntimeError):
    """Raised when WPC has no official PP file for a requested valid period."""


def wpc_pp_category_ids(values) -> np.ndarray:
    """Classify official PP fractions using NOAA's 5/10/20/40% breaks."""
    probability = np.asarray(values, dtype=np.float64)
    categories = np.zeros(probability.shape, dtype=np.int16)
    finite = np.isfinite(probability)
    for category, (threshold, _label) in enumerate(WPC_PP_RISK_THRESHOLDS, start=1):
        categories[finite & (probability >= threshold)] = category
    return categories


def wpc_pp_threshold_for_forecast_label(label: str) -> float:
    """Return the official PP threshold corresponding to a forecast category."""
    try:
        return float(WPC_PP_THRESHOLD_BY_FORECAST_LABEL[str(label)])
    except KeyError as exc:
        raise ValueError(
            f"Unknown forecast category label {label!r}; expected one of "
            f"{list(WPC_PP_THRESHOLD_BY_FORECAST_LABEL)}"
        ) from exc


def _date8(value) -> str:
    text = str(value).strip().replace("-", "")
    if len(text) < 8 or not text[:8].isdigit():
        raise ValueError(f"Expected a YYYYMMDD-like date, got {value!r}")
    date = text[:8]
    datetime.strptime(date, "%Y%m%d")
    return date


def wpc_pp_file_info(date) -> dict[str, str]:
    """Return the official filename and URL for a case's 12Z-to-12Z window.

    WPC stores a file under the month containing the *ending* valid time.  Thus
    the 2024-06-30 case is stored in ``202407/``.
    """
    start_date = _date8(date)
    start = datetime.strptime(start_date, "%Y%m%d")
    end = start + timedelta(days=1)
    start_stamp = start.strftime("%Y%m%d") + "12"
    end_stamp = end.strftime("%Y%m%d") + "12"
    filename = f"pp_co_2p5km_s{start_stamp}_e{end_stamp}.nc"
    month = end.strftime("%Y%m")
    return {
        "date": start_date,
        "valid_start": start_stamp,
        "valid_end": end_stamp,
        "month": month,
        "filename": filename,
        "url": f"{WPC_PP_BASE_URL}/{month}/{filename}",
    }


def _looks_like_netcdf(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 1024:
        return False
    with path.open("rb") as handle:
        signature = handle.read(8)
    return signature.startswith(b"CDF") or signature == b"\x89HDF\r\n\x1a\n"


def download_wpc_pp_netcdf(date, cache_dir, force: bool = False, timeout: int = 60) -> tuple[Path, dict[str, str]]:
    """Download one official WPC file atomically, or return its local cache."""
    info = wpc_pp_file_info(date)
    target_dir = Path(cache_dir).expanduser() / "netcdf" / info["month"]
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / info["filename"]
    if not force and _looks_like_netcdf(target):
        return target, info

    request = Request(info["url"], headers={"User-Agent": "XGBFFP-Day1-viewer/1.0"})
    temp_name = None
    try:
        with urlopen(request, timeout=int(timeout)) as response:
            with tempfile.NamedTemporaryFile(
                mode="wb", prefix=target.name + ".", suffix=".tmp", dir=target_dir, delete=False
            ) as temp:
                temp_name = temp.name
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    temp.write(block)
        temp_path = Path(temp_name)
        if not _looks_like_netcdf(temp_path):
            raise RuntimeError(
                f"Downloaded response is not a usable NetCDF file: {info['url']}"
            )
        os.replace(temp_path, target)
        return target, info
    except HTTPError as exc:
        if exc.code == 404:
            raise WPCPracticallyPerfectUnavailable(
                f"WPC has no PP_2p5km file for {info['date']}: {info['url']}"
            ) from exc
        raise RuntimeError(
            f"WPC PP_2p5km download failed with HTTP {exc.code}: {info['url']}"
        ) from exc
    except URLError as exc:
        raise RuntimeError(
            f"WPC PP_2p5km download failed for {info['url']}: {exc.reason}"
        ) from exc
    finally:
        if temp_name and Path(temp_name).exists():
            Path(temp_name).unlink()


def _unit_xyz(lat, lon) -> np.ndarray:
    lat_rad = np.deg2rad(np.asarray(lat, dtype=np.float64))
    lon_rad = np.deg2rad(np.asarray(lon, dtype=np.float64))
    cos_lat = np.cos(lat_rad)
    return np.column_stack(
        (cos_lat * np.cos(lon_rad), cos_lat * np.sin(lon_rad), np.sin(lat_rad))
    )


def _grid_signature(lat, lon) -> str:
    digest = hashlib.sha256()
    for values in (lat, lon):
        array = np.ascontiguousarray(np.asarray(values, dtype="<f4"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()[:20]


def _plain_float_array(values) -> np.ndarray:
    """Convert NetCDF/xarray values to float, preserving native 2-D layout."""
    if np.ma.isMaskedArray(values):
        values = np.ma.filled(values, np.nan)
    return np.asarray(values, dtype=np.float64)


def read_wpc_pp_netcdf(path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Read native ``lat``, ``lon``, and ``PP`` arrays using NOAA's order.

    The supplied NOAA reference reader tries ``netCDF4.Dataset`` first and
    xarray second.  It reads all three native two-dimensional arrays directly;
    no transpose, flip, or longitude conversion is applied.  The reference
    multiplies the 2.5-km ``PP`` values by 100 only when plotting.  This reader
    deliberately retains the source 0..1 fractions used by the viewer.
    """
    source = Path(path)
    netcdf4_error = None
    try:
        import netCDF4 as nc

        with nc.Dataset(source) as dataset:
            missing = sorted({"lat", "lon", "PP"}.difference(dataset.variables))
            if missing:
                raise ValueError(f"Official WPC PP NetCDF is missing variables: {missing}")
            latitude = _plain_float_array(dataset.variables["lat"][:])
            longitude = _plain_float_array(dataset.variables["lon"][:])
            probability = _plain_float_array(dataset.variables["PP"][:])
            attrs = {
                "reader": "netCDF4.Dataset",
                "source_grid_spacing_km": float(
                    getattr(dataset, "d_km", WPC_PP_NATIVE_SPACING_KM)
                ),
            }
        return latitude, longitude, probability, attrs
    except Exception as exc:
        netcdf4_error = exc

    try:
        with xr.open_dataset(source, mask_and_scale=True, decode_times=False) as dataset:
            missing = sorted({"lat", "lon", "PP"}.difference(dataset.variables))
            if missing:
                raise ValueError(f"Official WPC PP NetCDF is missing variables: {missing}")
            latitude = _plain_float_array(dataset["lat"].values)
            longitude = _plain_float_array(dataset["lon"].values)
            probability = _plain_float_array(dataset["PP"].values)
            attrs = {
                "reader": "xarray.open_dataset",
                "source_grid_spacing_km": float(
                    dataset.attrs.get("d_km", WPC_PP_NATIVE_SPACING_KM)
                ),
            }
        return latitude, longitude, probability, attrs
    except Exception as xarray_error:
        raise RuntimeError(
            f"Could not open official WPC PP NetCDF {source}; "
            f"netCDF4 error={netcdf4_error!r}; xarray error={xarray_error!r}"
        ) from xarray_error


def sample_wpc_pp_arrays(
    source_lat,
    source_lon,
    source_pp,
    target_lat,
    target_lon,
    max_match_distance_km: float = WPC_PP_MAX_MATCH_DISTANCE_KM,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Crop and nearest-sample native NOAA PP arrays onto target points."""
    source_lat = _plain_float_array(source_lat)
    source_lon = _plain_float_array(source_lon)
    source_pp = _plain_float_array(source_pp)
    if source_lat.shape != source_lon.shape or source_lat.shape != source_pp.shape:
        raise ValueError(
            f"WPC lat/lon/PP shapes differ: {source_lat.shape}, {source_lon.shape}, {source_pp.shape}"
        )

    target_lat = np.asarray(target_lat, dtype=np.float64)
    target_lon = np.asarray(target_lon, dtype=np.float64)
    if target_lat.shape != target_lon.shape:
        raise ValueError(f"Target Lat/Lon shapes differ: {target_lat.shape} and {target_lon.shape}")

    output = np.full(target_lat.shape, np.nan, dtype=np.float32)
    distance_km = np.full(target_lat.shape, np.nan, dtype=np.float32)
    target_finite = np.isfinite(target_lat) & np.isfinite(target_lon)
    if not np.any(target_finite):
        return output, distance_km, {"source_points_after_crop": 0, "matched_points": 0}

    probability_finite = np.isfinite(source_pp)
    if np.any(probability_finite):
        source_min = float(np.nanmin(source_pp))
        source_max = float(np.nanmax(source_pp))
        if source_min < -1e-6 or source_max > 1.0 + 1e-6:
            raise ValueError(
                f"Expected official WPC PP probability in 0..1, found {source_min}..{source_max}"
            )

    # Crop before building the tree.  The degree padding safely exceeds the
    # maximum spherical match distance throughout the CONUS viewer domain.
    padding_deg = float(max_match_distance_km) / 80.0 + 0.02
    lat_min = float(np.nanmin(target_lat[target_finite])) - padding_deg
    lat_max = float(np.nanmax(target_lat[target_finite])) + padding_deg
    lon_min = float(np.nanmin(target_lon[target_finite])) - padding_deg
    lon_max = float(np.nanmax(target_lon[target_finite])) + padding_deg
    source_keep = (
        np.isfinite(source_lat)
        & np.isfinite(source_lon)
        & probability_finite
        & (source_lat >= lat_min)
        & (source_lat <= lat_max)
        & (source_lon >= lon_min)
        & (source_lon <= lon_max)
    )
    if not np.any(source_keep):
        return output, distance_km, {"source_points_after_crop": 0, "matched_points": 0}

    cropped_lat = source_lat[source_keep]
    cropped_lon = source_lon[source_keep]
    cropped_pp = source_pp[source_keep]
    tree = cKDTree(_unit_xyz(cropped_lat, cropped_lon))
    target_index = np.flatnonzero(target_finite)
    chord, nearest = tree.query(_unit_xyz(target_lat[target_finite], target_lon[target_finite]), k=1)
    chord = np.clip(np.asarray(chord, dtype=np.float64), 0.0, 2.0)
    nearest_km = EARTH_RADIUS_KM * (2.0 * np.arcsin(chord / 2.0))
    accepted = np.isfinite(nearest_km) & (nearest_km <= float(max_match_distance_km))
    accepted_index = target_index[accepted]
    output.flat[accepted_index] = cropped_pp[np.asarray(nearest)[accepted]].astype(np.float32)
    distance_km.flat[accepted_index] = nearest_km[accepted].astype(np.float32)

    metadata = {
        "source_points_after_crop": int(np.sum(source_keep)),
        "matched_points": int(np.sum(accepted)),
        "target_points": int(target_lat.size),
        "max_match_distance_km": float(max_match_distance_km),
        "source_probability_min": float(np.nanmin(cropped_pp)),
        "source_probability_max": float(np.nanmax(cropped_pp)),
        "source_percent_min": float(np.nanmin(cropped_pp) * 100.0),
        "source_percent_max": float(np.nanmax(cropped_pp) * 100.0),
    }
    return output, distance_km, metadata


def sample_wpc_pp_dataset(
    dataset: xr.Dataset,
    target_lat,
    target_lon,
    max_match_distance_km: float = WPC_PP_MAX_MATCH_DISTANCE_KM,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Crop and nearest-sample an open official PP dataset to target points.

    The nearest-neighbor method is deliberate: the official field contains
    published probability levels, and bilinear interpolation would manufacture
    intermediate probabilities.  Unmatched points remain NaN.
    """
    required = {"lat", "lon", "PP"}
    missing = sorted(required.difference(dataset.variables))
    if missing:
        raise ValueError(f"Official WPC PP NetCDF is missing variables: {missing}")

    return sample_wpc_pp_arrays(
        dataset["lat"].values,
        dataset["lon"].values,
        dataset["PP"].values,
        target_lat,
        target_lon,
        max_match_distance_km=max_match_distance_km,
    )


def load_wpc_pp_for_grid(
    date,
    target_lat,
    target_lon,
    cache_dir,
    force_download: bool = False,
    force_mapping: bool = False,
    max_match_distance_km: float = WPC_PP_MAX_MATCH_DISTANCE_KM,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Download/cache one date and map its official field to a target grid."""
    target_lat = np.asarray(target_lat, dtype=np.float64)
    target_lon = np.asarray(target_lon, dtype=np.float64)
    info = wpc_pp_file_info(date)
    signature = _grid_signature(target_lat, target_lon)
    distance_tag = f"{float(max_match_distance_km):.3f}".replace(".", "p")
    mapped_dir = Path(cache_dir).expanduser() / "mapped"
    mapped_dir.mkdir(parents=True, exist_ok=True)
    mapped_path = mapped_dir / (
        f"wpc_pp_2p5km_{info['date']}_{signature}_d{distance_tag}km_"
        f"{WPC_PP_MAPPING_VERSION}.npz"
    )

    if mapped_path.exists() and mapped_path.stat().st_size > 512 and not (force_mapping or force_download):
        try:
            with np.load(mapped_path, allow_pickle=False) as cached:
                values = np.asarray(cached["values"], dtype=np.float32)
                distances = np.asarray(cached["distance_km"], dtype=np.float32)
                metadata = json.loads(str(cached["metadata_json"].item()))
            if values.shape == target_lat.shape and distances.shape == target_lat.shape:
                metadata["mapped_cache"] = str(mapped_path)
                metadata["mapped_cache_hit"] = True
                return values, distances, metadata
        except Exception as exc:
            warnings.warn(f"Ignoring unreadable WPC PP mapped cache {mapped_path}: {exc}")

    netcdf_path, info = download_wpc_pp_netcdf(
        info["date"], cache_dir=cache_dir, force=force_download
    )
    source_lat, source_lon, source_pp, reader_metadata = read_wpc_pp_netcdf(netcdf_path)
    values, distances, metadata = sample_wpc_pp_arrays(
        source_lat,
        source_lon,
        source_pp,
        target_lat,
        target_lon,
        max_match_distance_km=max_match_distance_km,
    )
    metadata.update(reader_metadata)
    metadata["source_grid_shape"] = list(source_pp.shape)
    metadata["source_probability_units"] = (
        "fraction (0..1); multiply by 100 for percent display"
    )

    metadata.update(info)
    metadata.update(
        {
            "product": WPC_PP_PRODUCT,
            "mapping_version": WPC_PP_MAPPING_VERSION,
            "netcdf_path": str(netcdf_path),
            "mapped_cache": str(mapped_path),
            "mapped_cache_hit": False,
        }
    )
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=mapped_path.stem + ".", suffix=".npz", dir=mapped_dir, delete=False
        ) as temp:
            temp_name = temp.name
        np.savez_compressed(
            temp_name,
            values=values,
            distance_km=distances,
            metadata_json=json.dumps(metadata, sort_keys=True),
        )
        os.replace(temp_name, mapped_path)
    finally:
        if temp_name and Path(temp_name).exists():
            Path(temp_name).unlink()
    return values, distances, metadata


def replace_pp_with_official_wpc(
    dataframe: pd.DataFrame,
    cache_dir,
    *,
    strict: bool = False,
    force_download: bool = False,
    force_mapping: bool = False,
    max_match_distance_km: float = WPC_PP_MAX_MATCH_DISTANCE_KM,
) -> tuple[pd.DataFrame, list[dict]]:
    """Drop legacy/derived ``PP_*`` fields and add the official WPC field.

    Rows are retained here so realtime forecast grids remain intact.  Historical
    verification callers should filter on ``WPC_PP_Available``.
    """
    required = ["Date", "Lat", "Lon"]
    missing = [column for column in required if column not in dataframe.columns]
    if missing:
        raise ValueError(f"Viewer dataframe is missing required columns: {missing}")

    output = dataframe.copy().reset_index(drop=True)
    legacy_pp_columns = [column for column in output.columns if str(column).startswith("PP_")]
    output = output.drop(columns=legacy_pp_columns, errors="ignore")
    output["Date"] = output["Date"].astype(str).str.slice(0, 8)
    output[WPC_PP_COLUMN] = np.nan
    output["WPC_PP_Match_Distance_km"] = np.nan
    output["WPC_PP_Available"] = False
    output["WPC_PP_Product"] = None
    output["WPC_PP_Source_URL"] = None
    output["WPC_PP_Source_File"] = None
    output["WPC_PP_Valid_Start"] = None
    output["WPC_PP_Valid_End"] = None

    metadata_rows: list[dict] = []
    for date, index in output.groupby("Date", sort=True).groups.items():
        loc = np.asarray(list(index))
        try:
            values, distances, metadata = load_wpc_pp_for_grid(
                date,
                pd.to_numeric(output.loc[loc, "Lat"], errors="coerce").to_numpy(float),
                pd.to_numeric(output.loc[loc, "Lon"], errors="coerce").to_numpy(float),
                cache_dir=cache_dir,
                force_download=force_download,
                force_mapping=force_mapping,
                max_match_distance_km=max_match_distance_km,
            )
        except WPCPracticallyPerfectUnavailable as exc:
            unavailable = {"date": str(date), "available": False, "reason": str(exc)}
            metadata_rows.append(unavailable)
            if strict:
                raise
            warnings.warn(str(exc))
            continue

        available = np.isfinite(values)
        output.loc[loc, WPC_PP_COLUMN] = values
        output.loc[loc, "WPC_PP_Match_Distance_km"] = distances
        output.loc[loc, "WPC_PP_Available"] = available
        output.loc[loc, "WPC_PP_Product"] = WPC_PP_PRODUCT
        output.loc[loc, "WPC_PP_Source_URL"] = metadata["url"]
        output.loc[loc, "WPC_PP_Source_File"] = metadata["filename"]
        output.loc[loc, "WPC_PP_Valid_Start"] = metadata["valid_start"]
        output.loc[loc, "WPC_PP_Valid_End"] = metadata["valid_end"]
        metadata_rows.append({**metadata, "available": True})

    output[WPC_PP_COLUMN] = pd.to_numeric(output[WPC_PP_COLUMN], errors="coerce").astype(np.float32)
    output["WPC_PP_Match_Distance_km"] = pd.to_numeric(
        output["WPC_PP_Match_Distance_km"], errors="coerce"
    ).astype(np.float32)
    output["WPC_PP_Available"] = output["WPC_PP_Available"].fillna(False).astype(bool)
    return output, metadata_rows


def has_official_wpc_pp(dataframe: pd.DataFrame) -> bool:
    """Return whether a dataframe contains usable, provenance-tagged official PP."""
    if WPC_PP_COLUMN not in dataframe.columns or "WPC_PP_Product" not in dataframe.columns:
        return False
    product = dataframe["WPC_PP_Product"].dropna().astype(str)
    values = pd.to_numeric(dataframe[WPC_PP_COLUMN], errors="coerce")
    return bool(product.eq(WPC_PP_PRODUCT).any() and values.notna().any())
