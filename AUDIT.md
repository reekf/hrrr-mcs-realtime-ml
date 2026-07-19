# XGBFFP Website Extension Audit

Audited 2026-07-19 before implementation. The product name is **XGBoosted
Flash Flood Predictions (XGBFFP)**.

## Website entry points

- `docs/index.html` is the GitHub Pages entry point.
- `docs/app.js` owns the Leaflet 2D map, MapLibre/deck.gl 3D map, product and
  contour controls, predictor overlays, archive loading, radar, NWS alerts,
  Iowa Environmental Mesonet LSRs, and mPING reports.
- `docs/style.css` contains the desktop/mobile layouts.
- `docs/archive/index.json` indexes the forecast archive; each date uses
  `docs/archive/YYYYMMDD/map.json` and `status.json`.
- `publish_latest_ml_output.sh` publishes forecasts and
  `publish_verification_output.sh` adds post-event verification.

## Forecast map data

`generate_interactive_map_data.py` writes schema version 5:

- `grid.lat[]` and `grid.lon[]` are aligned one-dimensional grid coordinates.
- `layers.<product>.values[]` are aligned integer probabilities from 0 to
  1000; divide by 10 for percent.
- Available product keys can include `ml_r40`, `ml_r60`, `ml_r60v2`,
  `ml_r75`, `ml_r100`, `ml_mean`, `wpc`, and post-event `pp`.
- `contours.<product>.<threshold>` stores Leaflet-ready lines at 5, 15, 40,
  and 70 percent.
- Realtime maps can include `predictors.r<radius>.<predictor>` with a
  normalized 0–1000 value, `scale_min`, `scale_max`, units, rank, global mean
  absolute SHAP importance, and direction text.
- Verified maps can include observation point collections in `observations`.
- Older archives omit newer members, predictors, observations, or PP. The new
  UI must treat all of those as optional.

## Scientific source functions and outputs

The final viewer notebook is
`hazard_ml_v33_radiusstats_WORKING_BASELINE_PLUS_VERIFICATION_SHAP_REALTIME_MULTIRADIUS_ENSEMBLE_WPC_VALIDFIX_METRICS_PREDICTORS_v18_PP_EXCLUSIVE_PROXY_CUMULATIVE_VIOLINS.ipynb`.
Relevant final functions include:

- `compute_ets_pod_far` and `run_final_bs_ets_verification_plots`
- `plot_shap_global_summary`
- `plot_shap_dependence` and the selected dependence-plot block
- the realtime multi-radius verification and Practically Perfect helpers

Final 2024–2025 test-set assets and their case tables are already saved under:

- `fall_2025_ml_proj/paper_verification_bs_ets_final/`
- `fall_2025_ml_proj/paper_shap_figures/`
- `fall_2025_ml_proj/shap_dependence_r100/`

The website publisher copies the ETS, Practically Perfect ETS, and
including/excluding-Marginal Brier Score figures from
`paper_verification_bs_ets_final` into stable `docs/` locations. It does not
publish a Brier Skill Score or common-case risk-area figure, and it does not
retrain a model.

## Verification outputs

- Forecast-date `map.json` files containing `layers.pp` are the
  machine-readable realtime verification source available to the website.
- Existing `verification.png` files are presentation graphics, not a suitable
  source for numerical aggregation.
- Rolling categorical metrics will pool hits, misses, false alarms, and
  correct negatives across verified map files, then recalculate ETS, CSI, POD,
  FAR, and frequency bias.
- Brier Score will pool squared errors and sample counts for continuous ML
  probabilities against thresholded Practically Perfect truth at each
  categorical threshold. BSS will use the pooled observed climatology as the
  reference. Division-by-zero results are explicit `null`.
- Weekly means the latest seven verified forecasts. Monthly is the trailing
  30 calendar days ending on the latest verified forecast. Seasonal uses the
  latest forecast's meteorological season and handles December as DJF.
- Formal 2024–2025 test-set records are never included in these realtime
  rolling files.

## SHAP outputs

Global beeswarm and mean-absolute-importance figures exist for r60 and r100.
Selected dependence panels currently exist for r100. Local per-grid-point SHAP
values are not exported in map JSON and therefore will not be claimed or
displayed as local contributions.

## Existing tests and validation

- `tests/test_interactive_map_realtime_selection.py` checks realtime source
  selection.
- Existing publish validation checks schema version 5 and required product
  layers.
- Established static checks are `node --check docs/app.js`, `bash -n` on
  publishing scripts, `jq empty` for JSON, and `git diff --check`.

## Proposed data flow and files

1. `generate_dashboard_data.py` validates/copies final figures, creates skill
   and explainability manifests, and builds daily plus rolling realtime
   verification JSON from verified archive maps.
2. `docs/app.js` consumes the existing map schema for Location Briefing and
   consumes the new manifests/rolling JSON for dashboard views.
3. `publish_verification_output.sh` refreshes dashboard data after a verified
   forecast is published.
4. `docs/index.html` and `docs/style.css` add top-level navigation, dashboard
   views, and a responsive briefing panel without replacing the map.
5. `docs/DATA_SCHEMA.md`, `docs/METRICS.md`, `README.md`, and
   `BUILD_WEEK_CHANGELOG.md` document contracts and provenance.

## Backward-compatibility risks

- Archive schemas vary. Missing products and diagnostics must render as “Not
  available,” never as zero.
- The existing query parameter `view=3d` conflicts with requested top-level
  views. It will remain accepted for backward compatibility while new links
  use `view=forecast&map=3d`.
- External radar, alert, LSR, and mPING services can fail independently; the
  briefing will summarize only data already loaded and label unavailable
  sources.
- r60kmV2 remains an experimental variant and is excluded from the standard
  four-member agreement calculation.
- The archived map domain is an irregular point grid. A selected point farther
  than 100 km from the nearest valid grid point is treated as outside the
  forecast domain.
