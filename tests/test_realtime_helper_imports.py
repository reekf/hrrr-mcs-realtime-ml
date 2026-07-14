#!/usr/bin/env python3
"""Regression checks for dynamically loaded realtime feature helpers."""

from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from realtime_mcs_trigger_plot import load_training_module_for_realtime


def test_dynamic_helper_can_import_sibling_module(tmp_path):
    sibling = tmp_path / "mode_case_catalog.py"
    sibling.write_text("CATALOG_SENTINEL = 'loaded-from-sibling'\n")
    helper = tmp_path / "generated_helper.py"
    helper.write_text("from mode_case_catalog import CATALOG_SENTINEL\n")

    parent = str(tmp_path.resolve())
    assert parent not in sys.path
    sys.modules.pop("mode_case_catalog", None)

    try:
        module = load_training_module_for_realtime(
            radius_km=60,
            script_dir=tmp_path,
            explicit_script=str(helper),
            original_root=Path("/original-root"),
            local_root=Path("/local-root"),
        )

        assert module.CATALOG_SENTINEL == "loaded-from-sibling"
        assert parent not in sys.path
    finally:
        sys.modules.pop("mode_case_catalog", None)


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        test_dynamic_helper_can_import_sibling_module(Path(directory))
    print("Realtime helper sibling-import regression check passed.")
