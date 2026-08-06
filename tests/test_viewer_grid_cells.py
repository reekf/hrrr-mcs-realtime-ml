import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from pyproj import CRS, Transformer

from viewer_grid_cells import add_categorical_grid_cells, grid_cell_vertices


def test_grid_cells_are_real_overlapping_map_polygons():
    lon = np.asarray([-98.0, -97.9, -98.0, -97.9])
    lat = np.asarray([40.0, 40.0, 40.1, 40.1])
    vertices, good, spacing_m = grid_cell_vertices(lon, lat, fill_factor=1.06)

    assert good.tolist() == [True, True, True, True]
    assert vertices.shape == (4, 4, 2)
    assert spacing_m > 0

    local = CRS.from_proj4(
        "+proj=aeqd +lat_0=40.05 +lon_0=-97.95 +datum=WGS84 +units=m +no_defs"
    )
    to_local = Transformer.from_crs("EPSG:4326", local, always_xy=True)
    x, _ = to_local.transform(vertices[0, :, 0], vertices[0, :, 1])
    assert np.ptp(x) > spacing_m


def test_categorical_renderer_creates_one_filled_collection_per_category():
    fig, ax = plt.subplots()
    collections = add_categorical_grid_cells(
        ax,
        [-98.0, -97.9, -98.0, -97.9],
        [40.0, 40.0, 40.1, 40.1],
        ["low", "high", "low", "high"],
        colors={"low": "green", "high": "red"},
        order=["low", "high"],
    )
    assert set(collections) == {"low", "high"}
    assert all(len(collection.get_paths()) == 2 for collection in collections.values())
    plt.close(fig)
