#!/usr/bin/env python3
"""Apply the lifetime-centered MCS-domain contract to the Day-1 viewer notebook."""

from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK = Path(
    "hazard_ml_v33_radiusstats_WORKING_BASELINE_PLUS_VERIFICATION_SHAP_REALTIME_"
    "MULTIRADIUS_ENSEMBLE_WPC_VALIDFIX_METRICS_PREDICTORS_v18_PP_EXCLUSIVE_"
    "PROXY_CUMULATIVE_VIOLINS.ipynb"
)
MARKER = "MCS lifetime-centered verification domains loaded"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Expected one {label} block, found {count}")
    return text.replace(old, new, 1)


def set_source(cell: dict, text: str) -> None:
    cell["source"] = text.splitlines(keepends=True)


def repair_dynamic_extent_helpers(notebook: dict) -> None:
    primary = "".join(notebook["cells"][1]["source"])
    primary = primary.replace(
        "MCS_LIFETIME_DOMAIN_JSON = os.path.join(PROJECT_DIR, \"mcs_lifetime_domains_400km.json\")",
        "MCS_LIFETIME_DOMAIN_JSON = os.environ.get(\n"
        "    \"XGBFFP_MCS_DOMAIN_JSON\", os.path.abspath(\"mcs_lifetime_domains_400km.json\")\n"
        ")",
    )
    primary = primary.replace(
        "MCS_LIFETIME_DOMAIN_JSON = os.path.join(SCRIPT_DIR, \"mcs_lifetime_domains_400km.json\")",
        "MCS_LIFETIME_DOMAIN_JSON = os.environ.get(\n"
        "    \"XGBFFP_MCS_DOMAIN_JSON\", os.path.abspath(\"mcs_lifetime_domains_400km.json\")\n"
        ")",
    )
    primary = primary.replace(
        "    strict=REQUIRE_MCS_LIFETIME_DOMAINS,\n",
        "    strict=False,\n",
    )
    retained_block = (
        "print(\n"
        "    f\"MCS lifetime-centered verification domains loaded: {len(mcs_case_domains)} cases; \"\n"
        "    f\"retained {len(df_radius_viewer):,}/{len(df_radius_viewer_full_domain):,} viewer rows\"\n"
        ")\n"
    )
    if "excluded_mcs_dates =" not in primary:
        primary = primary.replace(
            retained_block,
            retained_block
            + "excluded_mcs_dates = sorted(\n"
            + "    set(df_radius_viewer_full_domain[\"Date\"].astype(str).str[:8]) - set(mcs_case_domains)\n"
            + ")\n"
            + "if excluded_mcs_dates:\n"
            + "    print(f\"Excluded cases without a complete 400-km ML grid: {excluded_mcs_dates}\")\n",
        )
    primary = primary.replace(
        "    dates = sorted(base_keyed_all[\"Date\"].astype(str).unique().tolist())\n",
        "    dates = sorted(set(base_keyed_all[\"Date\"].astype(str).unique()) & set(mcs_case_domains))\n",
    )
    primary = primary.replace(
        "w_lon_min = widgets.FloatText(value=DEFAULT_EXTENT[0], description=\"Lon min\")\n"
        "w_lon_max = widgets.FloatText(value=DEFAULT_EXTENT[1], description=\"Lon max\")\n"
        "w_lat_min = widgets.FloatText(value=DEFAULT_EXTENT[2], description=\"Lat min\")\n"
        "w_lat_max = widgets.FloatText(value=DEFAULT_EXTENT[3], description=\"Lat max\")",
        "_initial_mcs_extent = extent_for_case(ranked_dates[0], mcs_case_domains)\n"
        "w_lon_min = widgets.FloatText(value=_initial_mcs_extent[0], description=\"Lon min\", disabled=True)\n"
        "w_lon_max = widgets.FloatText(value=_initial_mcs_extent[1], description=\"Lon max\", disabled=True)\n"
        "w_lat_min = widgets.FloatText(value=_initial_mcs_extent[2], description=\"Lat min\", disabled=True)\n"
        "w_lat_max = widgets.FloatText(value=_initial_mcs_extent[3], description=\"Lat max\", disabled=True)",
    )
    primary = primary.replace(
        "def _on_plot_clicked(_):\n"
        "    with out_plot:\n"
        "        clear_output(wait=True)\n"
        "        extent = [w_lon_min.value, w_lon_max.value, w_lat_min.value, w_lat_max.value]",
        "def _sync_mcs_extent_widgets(date):\n"
        "    extent = extent_for_case(date, mcs_case_domains)\n"
        "    w_lon_min.value, w_lon_max.value, w_lat_min.value, w_lat_max.value = extent\n"
        "    return extent\n\n"
        "def _on_viewer_date_change(change):\n"
        "    if change.get(\"name\") == \"value\" and change.get(\"new\"):\n"
        "        _sync_mcs_extent_widgets(change[\"new\"])\n\n"
        "w_date.observe(_on_viewer_date_change, names=\"value\")\n\n"
        "def _on_plot_clicked(_):\n"
        "    with out_plot:\n"
        "        clear_output(wait=True)\n"
        "        extent = _sync_mcs_extent_widgets(w_date.value)",
    )
    set_source(notebook["cells"][1], primary)

    cell10 = "".join(notebook["cells"][10]["source"])
    cell10 = cell10.replace(
        "def _setup_realtime_axis(ax, title):\n"
        "    _setup_map_ax(ax, _mcs_extent_for_df(sub), show_states=True, show_coastline=True)",
        "def _setup_realtime_axis(ax, title, df=None):\n"
        "    _setup_map_ax(ax, _mcs_extent_for_df(df), show_states=True, show_coastline=True)",
    )
    cell10 = cell10.replace("_setup_realtime_axis(ax, title)", "_setup_realtime_axis(ax, title, df)")
    set_source(notebook["cells"][10], cell10)

    cell20 = "".join(notebook["cells"][20]["source"])
    old = (
        "def _setup_predictor_browser_map(ax, title=None):\n"
        "    if \"_setup_map_ax\" in globals():\n"
        "        _setup_map_ax(ax, DEFAULT_EXTENT, show_states=True, show_countries=True, show_coastline=True)\n"
        "    else:\n"
        "        ax.set_xlim(DEFAULT_EXTENT[0], DEFAULT_EXTENT[1])\n"
        "        ax.set_ylim(DEFAULT_EXTENT[2], DEFAULT_EXTENT[3])"
    )
    new = (
        "def _setup_predictor_browser_map(ax, title=None, lon=None, lat=None):\n"
        "    extent = DEFAULT_EXTENT\n"
        "    if lon is not None and lat is not None:\n"
        "        lon_values = np.asarray(lon, dtype=float)\n"
        "        lat_values = np.asarray(lat, dtype=float)\n"
        "        good = np.isfinite(lon_values) & np.isfinite(lat_values)\n"
        "        if np.any(good):\n"
        "            extent = [float(np.min(lon_values[good])) - 0.05, float(np.max(lon_values[good])) + 0.05, "
        "float(np.min(lat_values[good])) - 0.05, float(np.max(lat_values[good])) + 0.05]\n"
        "    if \"_setup_map_ax\" in globals():\n"
        "        _setup_map_ax(ax, extent, show_states=True, show_countries=True, show_coastline=True)\n"
        "    else:\n"
        "        ax.set_xlim(extent[0], extent[1])\n"
        "        ax.set_ylim(extent[2], extent[3])"
    )
    cell20 = cell20.replace(old, new)
    cell20 = cell20.replace(
        "_setup_predictor_browser_map(ax, title=title)",
        "_setup_predictor_browser_map(ax, title=title, lon=lon, lat=lat)",
    )
    set_source(notebook["cells"][20], cell20)


def main() -> int:
    notebook = json.loads(NOTEBOOK.read_text())
    primary = "".join(notebook["cells"][1]["source"])
    if MARKER in primary:
        repair_dynamic_extent_helpers(notebook)
        NOTEBOOK.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
        print(f"Repaired dynamic extent helpers in {NOTEBOOK}")
        return 0

    primary = replace_once(
        primary,
        ")\n\ntry:\n    import joblib",
        ")\nfrom mcs_lifetime_domains import (\n"
        "    domain_mask,\n"
        "    extent_for_case,\n"
        "    filter_dataframe_to_domains,\n"
        "    load_domains,\n"
        ")\n\ntry:\n    import joblib",
        "domain imports",
    )
    primary = replace_once(
        primary,
        "WPC_PP_2P5KM_CACHE_DIR = os.path.join(PROJECT_DIR, \"wpc_pp_2p5km_cache_v33\")\n"
        "FORCE_REFRESH_WPC_PP_DOWNLOADS = False",
        "WPC_PP_2P5KM_CACHE_DIR = os.path.join(PROJECT_DIR, \"wpc_pp_2p5km_cache_v33\")\n"
        "MCS_LIFETIME_DOMAIN_JSON = os.environ.get(\n"
        "    \"XGBFFP_MCS_DOMAIN_JSON\", os.path.abspath(\"mcs_lifetime_domains_400km.json\")\n"
        ")\n"
        "MCS_LIFETIME_BOX_KM = 400.0\n"
        "REQUIRE_MCS_LIFETIME_DOMAINS = True\n"
        "FORCE_REFRESH_WPC_PP_DOWNLOADS = False",
        "domain settings",
    )
    merge_anchor = (
        "df_radius_viewer = df_radius_viewer.drop(columns=[\"__LatKey\", \"__LonKey\"], errors=\"ignore\")\n"
        "df_radius_viewer[\"ML_Target_Radius_km\"] = df_radius_viewer[\"ML_Target_Radius_km\"].astype(int)\n\n"
        "print(\n"
    )
    merge_insert = (
        "df_radius_viewer = df_radius_viewer.drop(columns=[\"__LatKey\", \"__LonKey\"], errors=\"ignore\")\n"
        "df_radius_viewer[\"ML_Target_Radius_km\"] = df_radius_viewer[\"ML_Target_Radius_km\"].astype(int)\n\n"
        "if not os.path.isfile(MCS_LIFETIME_DOMAIN_JSON):\n"
        "    raise RuntimeError(\n"
        "        f\"Missing required PyFLEXTRKR MCS-domain cache: {MCS_LIFETIME_DOMAIN_JSON}. \"\n"
        "        \"Run build_rap_mcs_lifetime_domains.py before the viewer.\"\n"
        "    )\n"
        "mcs_case_domains = load_domains(MCS_LIFETIME_DOMAIN_JSON)\n"
        "df_radius_viewer_full_domain = df_radius_viewer\n"
        "df_radius_viewer = filter_dataframe_to_domains(\n"
        "    df_radius_viewer_full_domain,\n"
        "    mcs_case_domains,\n"
        "    strict=False,\n"
        ")\n"
        "print(\n"
        "    f\"MCS lifetime-centered verification domains loaded: {len(mcs_case_domains)} cases; \"\n"
        "    f\"retained {len(df_radius_viewer):,}/{len(df_radius_viewer_full_domain):,} viewer rows\"\n"
        ")\n"
        "excluded_mcs_dates = sorted(\n"
        "    set(df_radius_viewer_full_domain[\"Date\"].astype(str).str[:8]) - set(mcs_case_domains)\n"
        ")\n"
        "if excluded_mcs_dates:\n"
        "    print(f\"Excluded cases without a complete 400-km ML grid: {excluded_mcs_dates}\")\n\n"
        "def _mcs_extent_for_df(frame, fallback=DEFAULT_EXTENT):\n"
        "    if frame is not None and len(frame) and \"Date\" in frame.columns:\n"
        "        dates = frame[\"Date\"].astype(str).str.replace(r\"\\D\", \"\", regex=True).str[:8].unique()\n"
        "        if len(dates) == 1 and dates[0] in mcs_case_domains:\n"
        "            return extent_for_case(dates[0], mcs_case_domains)\n"
        "    return list(fallback)\n\n"
        "print(\n"
    )
    primary = replace_once(primary, merge_anchor, merge_insert, "post-merge domain filter")

    primary = primary.replace("    extent=DEFAULT_EXTENT,\n", "    extent=None,\n", 2)
    primary = replace_once(
        primary,
        "    sub = selected_case_df(date, radius_km, model_label=model_label)\n"
        "    lon = sub[\"Lon\"]",
        "    sub = selected_case_df(date, radius_km, model_label=model_label)\n"
        "    extent = _mcs_extent_for_df(sub) if extent is None else extent\n"
        "    lon = sub[\"Lon\"]",
        "selected-radius extent",
    )
    primary = replace_once(
        primary,
        "    d = str(date)[:8]\n"
        "    models = df_radius_viewer",
        "    d = str(date)[:8]\n"
        "    extent = extent_for_case(d, mcs_case_domains) if extent is None else extent\n"
        "    models = df_radius_viewer",
        "radius-comparison extent",
    )
    primary = replace_once(
        primary,
        "tag = f\"v33_singletarget_models_{model_tags}_verifyROI{int(VERIFY_ROI_KM)}km\"",
        "tag = f\"v33_singletarget_models_{model_tags}_mcs{int(MCS_LIFETIME_BOX_KM)}km_verifyROI{int(VERIFY_ROI_KM)}km\"",
        "metric cache tag",
    )
    primary = replace_once(
        primary,
        "        base = base.sort_values([\"__LatKey\", \"__LonKey\"]).reset_index(drop=True)\n"
        "        lat = base[\"Lat\"].to_numpy(dtype=float)\n"
        "        lon = base[\"Lon\"].to_numpy(dtype=float)\n"
        "        xyz = latlon_to_unit_xyz(lat, lon)",
        "        base = base.sort_values([\"__LatKey\", \"__LonKey\"]).reset_index(drop=True)\n"
        "        lat = base[\"Lat\"].to_numpy(dtype=float)\n"
        "        lon = base[\"Lon\"].to_numpy(dtype=float)\n"
        "        if str(date) not in mcs_case_domains:\n"
        "            raise RuntimeError(f\"Missing MCS lifetime domain for metric case {date}\")\n"
        "        evaluation_keep = domain_mask(lat, lon, mcs_case_domains[str(date)])\n"
        "        if not np.any(evaluation_keep):\n"
        "            raise RuntimeError(f\"MCS lifetime domain for {date} contains no metric-grid rows\")\n"
        "        xyz = latlon_to_unit_xyz(lat, lon)",
        "metric evaluation mask",
    )
    primary = replace_once(
        primary,
        "        wpc_masks = expanded_threshold_masks_from_values(base[WPC_COL].to_numpy(float), tree, xyz, radius_km=VERIFY_ROI_KM)\n"
        "        wpc_cat = category_from_expanded_masks(wpc_masks)",
        "        wpc_masks_full = expanded_threshold_masks_from_values(base[WPC_COL].to_numpy(float), tree, xyz, radius_km=VERIFY_ROI_KM)\n"
        "        wpc_masks = {label: values[evaluation_keep] for label, values in wpc_masks_full.items()}\n"
        "        wpc_cat = category_from_expanded_masks(wpc_masks)",
        "WPC metric crop",
    )
    primary = replace_once(
        primary,
        "            pp_masks = expanded_threshold_masks_from_values(base[pp_col].to_numpy(float), tree, xyz, radius_km=VERIFY_ROI_KM, official_pp=True)\n"
        "            pp_masks_by_truth[truth_def] = pp_masks",
        "            pp_masks_full = expanded_threshold_masks_from_values(base[pp_col].to_numpy(float), tree, xyz, radius_km=VERIFY_ROI_KM, official_pp=True)\n"
        "            pp_masks = {label: values[evaluation_keep] for label, values in pp_masks_full.items()}\n"
        "            pp_masks_by_truth[truth_def] = pp_masks",
        "PP metric crop",
    )
    primary = replace_once(
        primary,
        "            ml_masks = expanded_threshold_masks_from_values(merged[\"ML_Forecast_Prob\"].to_numpy(float), tree, xyz, radius_km=VERIFY_ROI_KM)\n"
        "            ml_cat = category_from_expanded_masks(ml_masks)",
        "            ml_masks_full = expanded_threshold_masks_from_values(merged[\"ML_Forecast_Prob\"].to_numpy(float), tree, xyz, radius_km=VERIFY_ROI_KM)\n"
        "            ml_masks = {label: values[evaluation_keep] for label, values in ml_masks_full.items()}\n"
        "            ml_cat = category_from_expanded_masks(ml_masks)",
        "ML metric crop",
    )
    primary = primary.replace(
        "    dates = sorted(base_keyed_all[\"Date\"].astype(str).unique().tolist())\n",
        "    dates = sorted(set(base_keyed_all[\"Date\"].astype(str).unique()) & set(mcs_case_domains))\n",
    )
    set_source(notebook["cells"][1], primary)

    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        source = source.replace(
            "_setup_map_ax(ax, DEFAULT_EXTENT, show_states=True, show_coastline=True)",
            "_setup_map_ax(ax, _mcs_extent_for_df(sub), show_states=True, show_coastline=True)",
        )
        source = source.replace(
            "    if \"DEFAULT_EXTENT\" in globals():\n        return DEFAULT_EXTENT\n",
            "    if \"_mcs_extent_for_df\" in globals():\n        return _mcs_extent_for_df(df)\n"
            "    if \"DEFAULT_EXTENT\" in globals():\n        return DEFAULT_EXTENT\n",
        )
        set_source(cell, source)

    repair_dynamic_extent_helpers(notebook)

    NOTEBOOK.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
    print(f"Updated {NOTEBOOK}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
