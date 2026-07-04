#!/usr/bin/env python3
"""Localized probability-matched mean for aligned gridded probability members."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


EARTH_RADIUS_KM = 6371.0


def _latlon_to_unit_xyz(lat, lon) -> np.ndarray:
    lat_rad = np.deg2rad(np.asarray(lat, dtype=float))
    lon_rad = np.deg2rad(np.asarray(lon, dtype=float))
    cos_lat = np.cos(lat_rad)
    return np.column_stack(
        [cos_lat * np.cos(lon_rad), cos_lat * np.sin(lon_rad), np.sin(lat_rad)]
    )


def localized_probability_matched_mean(
    member_stack,
    lat,
    lon,
    radius_km: float = 100.0,
) -> np.ndarray:
    """Return a local PMM field using the v33 viewer's rank-matching method.

    ``member_stack`` has shape ``(n_members, n_grid_points)``. At each point,
    the local rank of the member-mean field is matched to the pooled probability
    distribution of all members within ``radius_km``.
    """
    members = np.asarray(member_stack, dtype=float)
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    if members.ndim != 2:
        raise ValueError(f"Expected member_stack to be 2D, got {members.shape}")
    if members.shape[1] != len(lat) or len(lat) != len(lon):
        raise ValueError("Member fields and coordinate arrays must have matching grid lengths")
    if float(radius_km) <= 0:
        raise ValueError("radius_km must be positive")

    members = np.clip(members, 0.0, 1.0)
    n_grid = members.shape[1]
    if n_grid == 0:
        return np.asarray([], dtype=np.float32)
    with np.errstate(invalid="ignore"):
        member_mean = np.nanmean(members, axis=0)

    xyz = _latlon_to_unit_xyz(lat, lon)
    chord_radius = 2.0 * np.sin(float(radius_km) / (2.0 * EARTH_RADIUS_KM))
    neighborhoods = cKDTree(xyz).query_ball_point(xyz, r=chord_radius)
    out = np.full(n_grid, np.nan, dtype=np.float32)

    for index, neighborhood in enumerate(neighborhoods):
        target = member_mean[index]
        if not np.isfinite(target):
            continue
        neighbor_index = np.asarray(neighborhood, dtype=int)
        local_mean = member_mean[neighbor_index]
        local_mean = local_mean[np.isfinite(local_mean)]
        pooled = members[:, neighbor_index].reshape(-1)
        pooled = pooled[np.isfinite(pooled)]
        if pooled.size == 0:
            out[index] = target
        elif local_mean.size <= 1:
            out[index] = float(np.nanmedian(pooled))
        else:
            sorted_mean = np.sort(local_mean)
            sorted_pool = np.sort(pooled)
            rank = np.searchsorted(sorted_mean, target, side="left")
            quantile = float(np.clip(rank / max(local_mean.size - 1, 1), 0.0, 1.0))
            pool_index = int(np.clip(round(quantile * (sorted_pool.size - 1)), 0, sorted_pool.size - 1))
            out[index] = sorted_pool[pool_index]
    return np.clip(out, 0.0, 1.0)
