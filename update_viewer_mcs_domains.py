#!/usr/bin/env python3
"""Apply the lifetime-centered MCS-domain contract to the Day-1 viewer notebook."""

from __future__ import annotations

import json
import re
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

    # The manifest stores PyFLEXTRKR-selected centers.  Its original 400-km
    # size must not force users to build a second manifest just to zoom out.
    primary = primary.replace("mcs_lifetime_domains_800km.json", "mcs_lifetime_domains_400km.json")
    primary = primary.replace(
        "    filter_dataframe_to_domains,\n    load_domains,\n)",
        "    domains_with_complete_grid_coverage,\n    filter_dataframe_to_domains,\n"
        "    load_domains,\n    resize_domains,\n)",
    )
    if "domains_with_complete_grid_coverage" not in primary.split("try:\n    import joblib", 1)[0]:
        primary = primary.replace(
            "    domain_mask,\n",
            "    domain_mask,\n    domains_with_complete_grid_coverage,\n",
        )
    if "from viewer_grid_cells import (" not in primary:
        primary = primary.replace(
            "    resize_domains,\n)\n\ntry:\n    import joblib",
            "    resize_domains,\n)\nfrom viewer_grid_cells import (\n"
            "    DEFAULT_CELL_FILL_FACTOR,\n"
            "    add_categorical_grid_cells,\n"
            "    add_continuous_grid_cells,\n"
            ")\n\ntry:\n    import joblib",
        )
    elif "add_continuous_grid_cells" not in primary.split("try:\n    import joblib", 1)[0]:
        primary = primary.replace(
            "    add_categorical_grid_cells,\n)",
            "    add_categorical_grid_cells,\n    add_continuous_grid_cells,\n)",
        )
    primary = primary.replace(
        "# Map settings.\nPOINT_SIZE_DEFAULT = 8.0\nPOINT_ALPHA_DEFAULT = 0.90",
        "# Analysis-map settings.  Filled cells are real map polygons, so they remain\n"
        "# contiguous when the MCS domain is zoomed.  Values >1 add slight overlap to\n"
        "# suppress hairline seams from raster/vector rendering.\n"
        "GRID_CELL_FILL_FACTOR = DEFAULT_CELL_FILL_FACTOR\n"
        "POINT_SIZE_DEFAULT = 8.0  # Legacy plotting API compatibility; cells ignore marker size.\n"
        "POINT_ALPHA_DEFAULT = 0.90",
    )
    primary = primary.replace(
        "mcs_case_domains = load_domains(MCS_LIFETIME_DOMAIN_JSON)\n"
        "df_radius_viewer_full_domain = df_radius_viewer",
        "mcs_case_domain_centers = load_domains(MCS_LIFETIME_DOMAIN_JSON)\n"
        "mcs_case_domains_all = resize_domains(mcs_case_domain_centers, MCS_LIFETIME_BOX_KM)\n"
        "mcs_case_domains, mcs_domain_coverage_exclusions = domains_with_complete_grid_coverage(\n"
        "    df_radius_viewer, mcs_case_domains_all\n"
        ")\n"
        "df_radius_viewer_full_domain = df_radius_viewer",
    )
    primary = primary.replace(
        "mcs_case_domains = resize_domains(mcs_case_domain_centers, MCS_LIFETIME_BOX_KM)\n"
        "df_radius_viewer_full_domain = df_radius_viewer",
        "mcs_case_domains_all = resize_domains(mcs_case_domain_centers, MCS_LIFETIME_BOX_KM)\n"
        "mcs_case_domains, mcs_domain_coverage_exclusions = domains_with_complete_grid_coverage(\n"
        "    df_radius_viewer, mcs_case_domains_all\n"
        ")\n"
        "df_radius_viewer_full_domain = df_radius_viewer",
    )
    primary = primary.replace(
        "metric_tables = compute_or_load_radius_metrics(force=FORCE_REBUILD_RADIUS_METRICS)\n"
        "df_radius_metrics_by_case = metric_tables[\"case\"]",
        "metric_tables = compute_or_load_radius_metrics(force=FORCE_REBUILD_RADIUS_METRICS)\n"
        "METRIC_TABLE_BOX_KM = float(MCS_LIFETIME_BOX_KM)\n"
        "df_radius_metrics_by_case = metric_tables[\"case\"]",
    )
    primary = primary.replace(
        "print(f\"Excluded cases without a complete 400-km ML grid: {excluded_mcs_dates}\")",
        "print(f\"Excluded cases without a complete MCS-centered ML grid: {excluded_mcs_dates}\")",
    )
    primary = primary.replace(
        "    set(df_radius_viewer_full_domain[\"Date\"].astype(str).str[:8]) - set(mcs_case_domains)\n",
        "    set(df_radius_viewer_full_domain[\"Date\"].astype(str).str[:8]) - set(mcs_case_domain_centers)\n",
    )
    if "Excluded clipped cases at" not in primary:
        primary = primary.replace(
            "if excluded_mcs_dates:\n"
            "    print(f\"Excluded cases without a complete MCS-centered ML grid: {excluded_mcs_dates}\")\n",
            "if excluded_mcs_dates:\n"
            "    print(f\"Excluded cases without a complete MCS-centered ML grid: {excluded_mcs_dates}\")\n"
            "if mcs_domain_coverage_exclusions:\n"
            "    print(\n"
            "        f\"Excluded clipped cases at {MCS_LIFETIME_BOX_KM:g} km: \"\n"
            "        f\"{[row['date'] for row in mcs_domain_coverage_exclusions]}\"\n"
            "    )\n",
        )

    old_categorical = '''def _scatter_categorical(ax, lon, lat, labels, point_size=8.0, alpha=0.9, show_below_5=False, transform=None):
    labels = np.asarray(labels, dtype=object)
    for lab in RISK_LABELS:
        if lab == "<5%" and not show_below_5:
            continue
        m = labels == lab
        if not np.any(m):
            continue
        kwargs = dict(s=point_size, c=RISK_COLORS[lab], alpha=alpha, label=lab, linewidths=0.0)
        if transform is not None:
            kwargs["transform"] = transform
        ax.scatter(lon[m], lat[m], **kwargs)
'''
    new_categorical = '''def _scatter_categorical(ax, lon, lat, labels, point_size=8.0, alpha=0.9, show_below_5=False, transform=None):
    """Render analysis fields as contiguous, zoom-stable grid cells.

    ``point_size`` remains in the signature for older calls but is intentionally
    ignored: marker sizes are screen-relative and create gaps when zoomed.
    """
    hidden = () if show_below_5 else ("<5%",)
    return add_categorical_grid_cells(
        ax, lon, lat, labels,
        colors=RISK_COLORS,
        order=RISK_LABELS,
        hidden=hidden,
        alpha=alpha,
        fill_factor=GRID_CELL_FILL_FACTOR,
        transform=transform,
    )
'''
    primary = primary.replace(old_categorical, new_categorical)

    old_agreement = '''def _scatter_agreement(ax, lon, lat, labels, point_size=8.0, alpha=0.9, transform=None):
    labels = np.asarray(labels, dtype=object)
    for lab in ["Neither", "Hit", "Miss", "False Alarm"]:
        m = labels == lab
        if not np.any(m):
            continue
        if lab == "Neither":
            kwargs = dict(s=max(point_size * 0.6, 1.0), c=AGREEMENT_COLORS[lab], alpha=0.30, label=lab, linewidths=0.0)
        else:
            kwargs = dict(s=point_size, c=AGREEMENT_COLORS[lab], alpha=alpha, label=lab, linewidths=0.0)
        if transform is not None:
            kwargs["transform"] = transform
        ax.scatter(lon[m], lat[m], **kwargs)
'''
    new_agreement = '''def _scatter_agreement(ax, lon, lat, labels, point_size=8.0, alpha=0.9, transform=None):
    return add_categorical_grid_cells(
        ax, lon, lat, labels,
        colors=AGREEMENT_COLORS,
        order=["Neither", "Hit", "Miss", "False Alarm"],
        alpha=alpha,
        fill_factor=GRID_CELL_FILL_FACTOR,
        transform=transform,
    )
'''
    primary = primary.replace(old_agreement, new_agreement)
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
        "w_point_size = widgets.FloatSlider(value=POINT_SIZE_DEFAULT, min=1, max=40, step=1, description=\"Pt size\")",
        "w_box_km = widgets.BoundedFloatText(\n"
        "    value=MCS_LIFETIME_BOX_KM, min=100.0, max=2000.0, step=50.0, description=\"Box km\"\n"
        ")\n"
        "w_cell_fill = widgets.FloatSlider(\n"
        "    value=GRID_CELL_FILL_FACTOR, min=1.00, max=1.20, step=0.01, description=\"Cell fill\"\n"
        ")",
    )
    if "def _apply_viewer_domain_size(" not in primary:
        primary = primary.replace(
            "def _sync_mcs_extent_widgets(date):\n",
            "def _apply_viewer_domain_size(box_size_km):\n"
            "    global MCS_LIFETIME_BOX_KM, mcs_case_domains, mcs_domain_coverage_exclusions, df_radius_viewer\n"
            "    size = float(box_size_km)\n"
            "    changed = not np.isclose(size, MCS_LIFETIME_BOX_KM)\n"
            "    if changed:\n"
            "        resized = resize_domains(mcs_case_domain_centers, size)\n"
            "        candidate_domains, candidate_exclusions = domains_with_complete_grid_coverage(\n"
            "            df_radius_viewer_full_domain, resized\n"
            "        )\n"
            "        if not candidate_domains:\n"
            "            raise RuntimeError(f\"No cases have complete {size:g}-km ML-grid coverage\")\n"
            "        MCS_LIFETIME_BOX_KM = size\n"
            "        mcs_case_domains = candidate_domains\n"
            "        mcs_domain_coverage_exclusions = candidate_exclusions\n"
            "        df_radius_viewer = filter_dataframe_to_domains(\n"
            "            df_radius_viewer_full_domain, mcs_case_domains, strict=False\n"
            "        )\n"
            "        print(\n"
            "            f\"Viewer domain resized to {size:g} x {size:g} km; \"\n"
            "            f\"retained {len(df_radius_viewer):,} rows and \"\n"
            "            f\"excluded {len(mcs_domain_coverage_exclusions)} clipped cases.\"\n"
            "        )\n"
            "        options = [date for date in all_ranked_dates if date in mcs_case_domains]\n"
            "        previous = w_date.value\n"
            "        w_date.options = options\n"
            "        w_date.value = previous if previous in options else options[0]\n"
            "    return changed\n\n"
            "def _refresh_metric_globals_for_domain():\n"
            "    global metric_tables, METRIC_TABLE_BOX_KM\n"
            "    global df_radius_metrics_by_case, df_radius_metrics_case_mean, df_radius_metrics_pooled\n"
            "    global df_radius_rpss_by_case, df_radius_rpss_case_mean, df_radius_rpss_pooled\n"
            "    metric_tables = compute_or_load_radius_metrics(force=False)\n"
            "    METRIC_TABLE_BOX_KM = float(MCS_LIFETIME_BOX_KM)\n"
            "    df_radius_metrics_by_case = metric_tables[\"case\"]\n"
            "    df_radius_metrics_case_mean = metric_tables[\"case_mean\"]\n"
            "    df_radius_metrics_pooled = metric_tables[\"pooled\"]\n"
            "    df_radius_rpss_by_case = metric_tables[\"rpss_case\"]\n"
            "    df_radius_rpss_case_mean = metric_tables[\"rpss_case_mean\"]\n"
            "    df_radius_rpss_pooled = metric_tables[\"rpss_pooled\"]\n\n"
            "def _sync_mcs_extent_widgets(date):\n",
        )
    primary = primary.replace(
        "    global MCS_LIFETIME_BOX_KM, mcs_case_domains, df_radius_viewer",
        "    global MCS_LIFETIME_BOX_KM, mcs_case_domains, mcs_domain_coverage_exclusions, df_radius_viewer",
    )
    primary = primary.replace(
        "        mcs_case_domains = resize_domains(mcs_case_domain_centers, size)\n"
        "        df_radius_viewer = filter_dataframe_to_domains(",
        "        resized = resize_domains(mcs_case_domain_centers, size)\n"
        "        mcs_case_domains, mcs_domain_coverage_exclusions = domains_with_complete_grid_coverage(\n"
        "            df_radius_viewer_full_domain, resized\n"
        "        )\n"
        "        df_radius_viewer = filter_dataframe_to_domains(",
    )
    primary = primary.replace(
        "        MCS_LIFETIME_BOX_KM = size\n"
        "        resized = resize_domains(mcs_case_domain_centers, size)\n"
        "        mcs_case_domains, mcs_domain_coverage_exclusions = domains_with_complete_grid_coverage(\n"
        "            df_radius_viewer_full_domain, resized\n"
        "        )\n"
        "        df_radius_viewer = filter_dataframe_to_domains(",
        "        resized = resize_domains(mcs_case_domain_centers, size)\n"
        "        candidate_domains, candidate_exclusions = domains_with_complete_grid_coverage(\n"
        "            df_radius_viewer_full_domain, resized\n"
        "        )\n"
        "        if not candidate_domains:\n"
        "            raise RuntimeError(f\"No cases have complete {size:g}-km ML-grid coverage\")\n"
        "        MCS_LIFETIME_BOX_KM = size\n"
        "        mcs_case_domains = candidate_domains\n"
        "        mcs_domain_coverage_exclusions = candidate_exclusions\n"
        "        df_radius_viewer = filter_dataframe_to_domains(",
    )
    primary = primary.replace(
        "            f\"retained {len(df_radius_viewer):,} rows.\"\n"
        "        )\n"
        "    return changed",
        "            f\"retained {len(df_radius_viewer):,} rows and \"\n"
        "            f\"excluded {len(mcs_domain_coverage_exclusions)} clipped cases.\"\n"
        "        )\n"
        "        options = [date for date in all_ranked_dates if date in mcs_case_domains]\n"
        "        if not options:\n"
        "            raise RuntimeError(f\"No cases have complete {size:g}-km ML-grid coverage\")\n"
        "        previous = w_date.value\n"
        "        w_date.options = options\n"
        "        w_date.value = previous if previous in options else options[0]\n"
        "    return changed",
    )
    primary = primary.replace(
        "        if not options:\n"
        "            raise RuntimeError(f\"No cases have complete {size:g}-km ML-grid coverage\")\n"
        "        previous = w_date.value",
        "        previous = w_date.value",
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
    primary = primary.replace(
        "def _on_plot_clicked(_):\n"
        "    with out_plot:\n"
        "        clear_output(wait=True)\n"
        "        extent = _sync_mcs_extent_widgets(w_date.value)",
        "def _on_plot_clicked(_):\n"
        "    global GRID_CELL_FILL_FACTOR\n"
        "    with out_plot:\n"
        "        clear_output(wait=True)\n"
        "        _apply_viewer_domain_size(w_box_km.value)\n"
        "        GRID_CELL_FILL_FACTOR = float(w_cell_fill.value)\n"
        "        extent = _sync_mcs_extent_widgets(w_date.value)",
    )
    primary = primary.replace(
        "        elif w_view_mode.value == \"Metric dashboard\":\n"
        "            display_radius_metric_tables",
        "        elif w_view_mode.value == \"Metric dashboard\":\n"
        "            if not np.isclose(METRIC_TABLE_BOX_KM, MCS_LIFETIME_BOX_KM):\n"
        "                _refresh_metric_globals_for_domain()\n"
        "            display_radius_metric_tables",
    )
    primary = primary.replace(
        "    global metric_tables\n"
        "    global df_radius_metrics_by_case",
        "    global metric_tables, METRIC_TABLE_BOX_KM\n"
        "    global df_radius_metrics_by_case",
    )
    primary = primary.replace(
        "    metric_tables = compute_or_load_radius_metrics(force=False)\n"
        "    df_radius_metrics_by_case",
        "    metric_tables = compute_or_load_radius_metrics(force=False)\n"
        "    METRIC_TABLE_BOX_KM = float(MCS_LIFETIME_BOX_KM)\n"
        "    df_radius_metrics_by_case",
    )
    primary = primary.replace(
        "        domain_changed = _apply_viewer_domain_size(w_box_km.value)",
        "        _apply_viewer_domain_size(w_box_km.value)",
    )
    rank_block = (
        "all_ranked_dates = _rank_dates()\n"
        "ranked_dates = [date for date in all_ranked_dates if date in mcs_case_domains]"
    )
    primary, rank_repairs = re.subn(
        r"(?:all_)+ranked_dates = _rank_dates\(\)\n"
        r"(?:ranked_dates = \[date for date in all_ranked_dates if date in mcs_case_domains\]\n?)+",
        rank_block + "\n",
        primary,
    )
    if rank_repairs == 0:
        primary = replace_once(
            primary,
            "ranked_dates = _rank_dates()",
            rank_block,
            "ranked date initialization",
        )
    primary = primary.replace(
        "    tmp = df_radius_viewer[df_radius_viewer[\"ML_Target_Radius_km\"] == available_radii[0]].copy()",
        "    tmp = df_radius_viewer_full_domain[\n"
        "        df_radius_viewer_full_domain[\"ML_Target_Radius_km\"] == available_radii[0]\n"
        "    ].copy()",
    )
    primary = primary.replace(
        "            if domain_changed:\n                _refresh_metric_globals_for_domain()",
        "            if not np.isclose(METRIC_TABLE_BOX_KM, MCS_LIFETIME_BOX_KM):\n"
        "                _refresh_metric_globals_for_domain()",
    )
    primary = primary.replace(
        "widgets.HBox([w_point_size, w_alpha, w_show_low, w_states, w_countries, w_coast])",
        "widgets.HBox([w_box_km, w_cell_fill, w_alpha, w_show_low, w_states, w_countries, w_coast])",
    )
    primary = primary.replace("point_size=w_point_size.value", "point_size=POINT_SIZE_DEFAULT")
    set_source(notebook["cells"][1], primary)

    cell10 = "".join(notebook["cells"][10]["source"])
    cell10 = cell10.replace(
        "def _setup_realtime_axis(ax, title):\n"
        "    _setup_map_ax(ax, _mcs_extent_for_df(sub), show_states=True, show_coastline=True)",
        "def _setup_realtime_axis(ax, title, df=None):\n"
        "    _setup_map_ax(ax, _mcs_extent_for_df(df), show_states=True, show_coastline=True)",
    )
    cell10 = cell10.replace("_setup_realtime_axis(ax, title)", "_setup_realtime_axis(ax, title, df)")
    cell10 = cell10.replace(
        '''    transform = ccrs.PlateCarree() if HAS_CARTOPY else None
    if HAS_CARTOPY:
        sc = ax.scatter(
            lon[good], lat[good],
            c=vals[good],
            s=point_size,
            alpha=alpha,
            cmap=cmap,
            norm=norm,
            transform=transform,
            linewidths=0,
        )
    else:
        sc = ax.scatter(
            lon[good], lat[good],
            c=vals[good],
            s=point_size,
            alpha=alpha,
            cmap=cmap,
            norm=norm,
            linewidths=0,
        )
''',
        '''    transform = ccrs.PlateCarree() if HAS_CARTOPY else None
    sc = add_continuous_grid_cells(
        ax, lon[good], lat[good], vals[good],
        cmap=cmap,
        norm=norm,
        alpha=alpha,
        fill_factor=GRID_CELL_FILL_FACTOR,
        transform=transform,
    )
''',
    )
    set_source(notebook["cells"][10], cell10)

    paper_scatter = '''        scatter_kwargs = dict(
            c=np.clip(vals[good], PAPER_RT_MIN_PROB, 1.0),
            s=PAPER_RT_POINT_SIZE,
            alpha=PAPER_RT_ALPHA,
            cmap=PAPER_RT_CMAP,
            norm=(BoundaryNorm([0.05, 0.10, 0.20, 0.40, 1.01], PAPER_RT_CMAP.N, clip=True) if str(col) == WPC_PP_COLUMN else PAPER_RT_NORM),
            linewidths=0,
            rasterized=True,
        )

        if HAS_CARTOPY_RT_PAPER:
            scatter_kwargs["transform"] = ccrs.PlateCarree()

        ax.scatter(lon[good], lat[good], **scatter_kwargs)
'''
    paper_cells = '''        add_continuous_grid_cells(
            ax, lon[good], lat[good], np.clip(vals[good], PAPER_RT_MIN_PROB, 1.0),
            cmap=PAPER_RT_CMAP,
            norm=(BoundaryNorm([0.05, 0.10, 0.20, 0.40, 1.01], PAPER_RT_CMAP.N, clip=True) if str(col) == WPC_PP_COLUMN else PAPER_RT_NORM),
            alpha=PAPER_RT_ALPHA,
            fill_factor=GRID_CELL_FILL_FACTOR,
            transform=(ccrs.PlateCarree() if HAS_CARTOPY_RT_PAPER else None),
        )
'''
    for cell_index in (11, 13, 18):
        source = "".join(notebook["cells"][cell_index]["source"])
        source = source.replace(paper_scatter, paper_cells)
        set_source(notebook["cells"][cell_index], source)

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
    cell20 = cell20.replace(
        '''def _plot_continuous_scatter(ax, lon, lat, vals, title, cmap="viridis", vmin=None, vmax=None, point_size=7, alpha=0.85):
    transform = ccrs.PlateCarree() if HAS_CARTOPY else None
    _setup_predictor_browser_map(ax, title=title, lon=lon, lat=lat)
    kwargs = dict(c=vals, s=point_size, alpha=alpha, cmap=cmap, vmin=vmin, vmax=vmax)
    if HAS_CARTOPY:
        sc = ax.scatter(lon, lat, transform=transform, **kwargs)
    else:
        sc = ax.scatter(lon, lat, **kwargs)
    plt.colorbar(sc, ax=ax, shrink=0.85)
    return sc
''',
        '''def _plot_continuous_scatter(ax, lon, lat, vals, title, cmap="viridis", vmin=None, vmax=None, point_size=7, alpha=0.85):
    transform = ccrs.PlateCarree() if HAS_CARTOPY else None
    _setup_predictor_browser_map(ax, title=title, lon=lon, lat=lat)
    sc = add_continuous_grid_cells(
        ax, lon, lat, vals,
        cmap=cmap, vmin=vmin, vmax=vmax,
        alpha=alpha,
        fill_factor=GRID_CELL_FILL_FACTOR,
        transform=transform,
    )
    plt.colorbar(sc, ax=ax, shrink=0.85)
    return sc
''',
    )
    set_source(notebook["cells"][20], cell20)

    cell66 = "".join(notebook["cells"][66]["source"])
    cell66 = cell66.replace(
        '''            scatter_kwargs = dict(
                c=count_arr[good],
                s=HEATMAP_POINT_SIZE,
                cmap=row_cmap,
                norm=row_norm,
                alpha=HEATMAP_POINT_ALPHA,
                linewidths=0.0,
                marker="s",
                rasterized=True,
            )

            if transform is not None:
                scatter_kwargs["transform"] = transform

            last_scatter = ax.scatter(lon[good], lat[good], **scatter_kwargs)
''',
        '''            last_scatter = add_continuous_grid_cells(
                ax, lon[good], lat[good], count_arr[good],
                cmap=row_cmap,
                norm=row_norm,
                alpha=HEATMAP_POINT_ALPHA,
                fill_factor=GRID_CELL_FILL_FACTOR,
                transform=transform,
            )
''',
    )
    set_source(notebook["cells"][66], cell66)

    # Do not preserve a stale execution traceback from a manually invented
    # size-specific manifest name (for example, ..._800km.json).
    notebook["cells"][1]["outputs"] = []
    notebook["cells"][1]["execution_count"] = None


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
