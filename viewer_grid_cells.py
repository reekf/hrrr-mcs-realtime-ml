#!/usr/bin/env python3
"""Filled, zoom-stable grid-cell rendering for XGBFFP analysis maps."""

from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np


DEFAULT_CELL_FILL_FACTOR = 1.06


def grid_cell_vertices(
    lon: Iterable[float],
    lat: Iterable[float],
    *,
    fill_factor: float = DEFAULT_CELL_FILL_FACTOR,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Build projected square cells centered on an irregular lon/lat grid.

    Matplotlib scatter markers have a fixed screen size, so zooming exposes
    increasingly large gaps.  These cells are real map polygons whose size is
    based on the grid's median nearest-neighbor spacing, and therefore remain
    contiguous at every zoom level.

    Returns ``(vertices, coordinate_mask, nominal_spacing_m)``.  ``vertices``
    has shape ``(n_valid, 4, 2)`` in longitude/latitude coordinates.
    """
    from pyproj import CRS, Transformer
    from scipy.spatial import cKDTree

    lon_values = np.asarray(lon, dtype=float).reshape(-1)
    lat_values = np.asarray(lat, dtype=float).reshape(-1)
    if lon_values.shape != lat_values.shape:
        raise ValueError("Longitude and latitude arrays must have the same shape")
    factor = float(fill_factor)
    if not np.isfinite(factor) or factor <= 0:
        raise ValueError(f"fill_factor must be positive and finite, got {fill_factor!r}")

    coordinate_mask = np.isfinite(lon_values) & np.isfinite(lat_values)
    if np.count_nonzero(coordinate_mask) < 2:
        raise ValueError("At least two finite grid coordinates are required")
    valid_lon = lon_values[coordinate_mask]
    valid_lat = lat_values[coordinate_mask]

    center_lat = float(np.mean(valid_lat))
    center_lon = float(np.rad2deg(np.angle(np.mean(np.exp(1j * np.deg2rad(valid_lon))))))
    local = CRS.from_proj4(
        f"+proj=aeqd +lat_0={center_lat:.10f} +lon_0={center_lon:.10f} "
        "+datum=WGS84 +units=m +no_defs"
    )
    to_local = Transformer.from_crs("EPSG:4326", local, always_xy=True)
    to_lonlat = Transformer.from_crs(local, "EPSG:4326", always_xy=True)
    x, y = to_local.transform(valid_lon, valid_lat)
    xy = np.column_stack([x, y])

    tree = cKDTree(xy)
    neighbor_count = min(8, len(xy))
    distances, _ = tree.query(xy, k=neighbor_count)
    if distances.ndim == 1:
        distances = distances[:, None]
    positive = np.where(distances > 1.0, distances, np.nan)
    nearest = np.nanmin(positive, axis=1)
    nominal_spacing_m = float(np.nanmedian(nearest))
    if not np.isfinite(nominal_spacing_m) or nominal_spacing_m <= 0:
        raise ValueError("Could not infer a positive grid spacing from the coordinates")

    half = nominal_spacing_m * factor * 0.5
    offsets = np.asarray(
        [[-half, -half], [half, -half], [half, half], [-half, half]], dtype=float
    )
    local_vertices = xy[:, None, :] + offsets[None, :, :]
    corner_lon, corner_lat = to_lonlat.transform(
        local_vertices[..., 0].reshape(-1), local_vertices[..., 1].reshape(-1)
    )
    vertices = np.stack([corner_lon, corner_lat], axis=1).reshape(-1, 4, 2)
    return vertices, coordinate_mask, nominal_spacing_m


def _collection_transform(ax, source_crs):
    if source_crs is None:
        return None
    converter = getattr(source_crs, "_as_mpl_transform", None)
    return converter(ax) if converter is not None else source_crs


def add_categorical_grid_cells(
    ax,
    lon: Iterable[float],
    lat: Iterable[float],
    labels: Iterable[object],
    *,
    colors: Mapping[object, object],
    order: Iterable[object] | None = None,
    hidden: Iterable[object] = (),
    alpha: float = 1.0,
    fill_factor: float = DEFAULT_CELL_FILL_FACTOR,
    transform=None,
    rasterized: bool = True,
) -> dict[object, object]:
    """Render categorical values as contiguous map cells."""
    from matplotlib.collections import PolyCollection

    label_values = np.asarray(labels, dtype=object).reshape(-1)
    vertices, coordinate_mask, _ = grid_cell_vertices(lon, lat, fill_factor=fill_factor)
    if label_values.shape != coordinate_mask.shape:
        raise ValueError("Labels and coordinates must have the same length")
    valid_labels = label_values[coordinate_mask]
    hidden_values = set(hidden)
    draw_order = list(order) if order is not None else list(dict.fromkeys(valid_labels.tolist()))
    collections = {}
    mpl_transform = _collection_transform(ax, transform)
    for label in draw_order:
        if label in hidden_values:
            continue
        keep = valid_labels == label
        if not np.any(keep):
            continue
        collection = PolyCollection(
            vertices[keep],
            facecolors=colors[label],
            edgecolors="none",
            linewidths=0.0,
            antialiased=False,
            alpha=float(alpha),
            label=str(label),
            rasterized=bool(rasterized),
        )
        if mpl_transform is not None:
            collection.set_transform(mpl_transform)
        ax.add_collection(collection, autolim=False)
        collections[label] = collection
    return collections


def add_continuous_grid_cells(
    ax,
    lon: Iterable[float],
    lat: Iterable[float],
    values: Iterable[float],
    *,
    cmap="viridis",
    norm=None,
    vmin=None,
    vmax=None,
    alpha: float = 1.0,
    fill_factor: float = DEFAULT_CELL_FILL_FACTOR,
    transform=None,
    rasterized: bool = True,
):
    """Render continuous values as contiguous map cells and return the collection."""
    from matplotlib.collections import PolyCollection

    numeric = np.asarray(values, dtype=float).reshape(-1)
    vertices, coordinate_mask, _ = grid_cell_vertices(lon, lat, fill_factor=fill_factor)
    if numeric.shape != coordinate_mask.shape:
        raise ValueError("Values and coordinates must have the same length")
    valid_values = numeric[coordinate_mask]
    finite = np.isfinite(valid_values)
    collection = PolyCollection(
        vertices[finite],
        array=valid_values[finite],
        cmap=cmap,
        norm=norm,
        edgecolors="none",
        linewidths=0.0,
        antialiased=False,
        alpha=float(alpha),
        rasterized=bool(rasterized),
    )
    if norm is None:
        collection.set_clim(vmin=vmin, vmax=vmax)
    mpl_transform = _collection_transform(ax, transform)
    if mpl_transform is not None:
        collection.set_transform(mpl_transform)
    ax.add_collection(collection, autolim=False)
    return collection
