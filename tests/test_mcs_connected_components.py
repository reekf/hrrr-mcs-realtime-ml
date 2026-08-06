#!/usr/bin/env python3
"""Regression checks for the HRRR cold-cloud connected-object trigger."""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from realtime_mcs_trigger_plot import largest_component


def test_separated_cold_blobs_are_not_summed_for_trigger_area():
    mask = np.zeros((8, 8), dtype=bool)
    mask[0, 0:5] = True
    mask[4:6, 2:7] = True

    comp = largest_component(mask, cell_area_km2=1.0)

    assert int(mask.sum()) == 15
    assert comp["n_components"] == 2
    assert comp["largest_pixel_count"] == 10
    assert comp["max_area_km2"] == 10.0
    assert not (comp["max_area_km2"] >= 15.0)


if __name__ == "__main__":
    test_separated_cold_blobs_are_not_summed_for_trigger_area()
    print("MCS connected-component regression check passed.")
