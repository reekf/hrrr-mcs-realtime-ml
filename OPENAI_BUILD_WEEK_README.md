# OpenAI Build Week: GPT-5.6 and Codex Development Record

## Project

**XGBoosted Flash Flood Predictions (XGBFFP)** is an experimental website and
real-time pipeline for displaying XGBoost flash-flood guidance, WPC Excessive
Rainfall Outlook context, post-event Practically Perfect verification, and
model explainability information.

During OpenAI Build Week, GPT-5.6 through Codex was used as the primary
AI-assisted software-engineering environment for extending and hardening this
project. Codex inspected the existing repository, implemented changes, ran
checks, reviewed diffs, and helped publish the completed work to GitHub Pages.
Development was iterative: the human project lead reviewed the scientific
meaning and presentation of each result, supplied corrections, and retained
final authority over the product.

## Starting point and Build Week scope

This was an extension of an active human-led research project, not a project
created from an empty repository. The core meteorological research, XGBoost
modeling approach, saved model artifacts, forecast targets, and portions of the
forecast/verification workflow already existed. GPT-5.6/Codex was used to help
extend, integrate, optimize, test, document, and deploy that work. The
contributions below describe that AI-assisted engineering scope without
reassigning authorship of the underlying science.

## How GPT-5.6 and Codex were used

### Website development

GPT-5.6/Codex helped turn the existing forecast map into a broader
decision-support and evaluation website. Codex-assisted work included:

- auditing the existing Leaflet, MapLibre/deck.gl, archive, and publishing
  code before extending it;
- building direct-link Forecast, Model Skill, Running Verification,
  Explainability, and About views;
- adding a click-to-select Location Briefing with probabilities, risk
  categories, multi-model agreement, predictor diagnostics, nearby reports,
  alerts, verification context, and copy-ready text;
- adding and refining 2D/3D forecast layers, contour controls, radar, flood
  alerts, local storm reports, mPING reports, predictor overlays, and
  observation/report radius displays;
- connecting finalized 2024–2025 ETS, Practically Perfect ETS, Any Flood Proxy
  Brier Score, and SHAP figures to machine-readable manifests;
- improving figure legibility with full-width, high-resolution presentation
  and direct full-resolution links;
- creating responsive layouts and cache-versioned assets for reliable GitHub
  Pages updates; and
- adding automated JavaScript, Python, JSON, shell, schema, and unit checks.

Recent human-directed refinements implemented with Codex included:

- showing pooled Hits, False Alarms, and Misses against Practically Perfect
  truth;
- adding per-product running counts of verified cases that actually contained
  the selected ML/WPC risk area;
- separating selected-risk case counts from total verified cases; and
- defaulting verification threshold controls to Moderate-or-greater.

### Improving ML-code efficiency

GPT-5.6/Codex was also used to inspect and improve the large XGBoost training
workflows so they could operate more efficiently on multi-case, multi-radius
datasets. This work included:

- developing memory-safe master-Parquet assembly with incremental PyArrow
  row-group writing instead of concatenating every daily dataset in memory;
- creating variants that retain the identifiers, target fields,
  and predictor columns required for training while omitting unnecessary
  intermediate/debug columns;
- adding bounded train/test row sampling before full pandas materialization,
  while preserving full-domain feature engineering and formal evaluation;
- reducing dataframe memory with narrower numeric dtypes and explicit cleanup
  of large temporary objects;
- controlling XGBoost/Optuna/Ray concurrency and releasing Ray resources before
  local full-model fitting;
- generating consistent R40, R60, R75, R100, and experimental same-radius
  training-script variants from known working provenance;
- centralizing case-catalog parsing, date deduplication, and stable case IDs;
  and
- adding compile, artifact, feature-radius, target-contract, and synthetic
  validation checks before treating a training result as trustworthy.

These changes were intended to improve memory use, repeatability, and
iteration speed without silently changing the human-defined modeling
contract.

### Building the real-time prediction and verification pipeline

Codex-assisted engineering helped build out and harden the operational path
from model artifacts to the public website:

1. The real-time plotter loads current atmospheric inputs and the saved XGBoost
   radius models.
2. Forecast probabilities are exported to a validated machine-readable map
   schema with ML members, ensemble mean, WPC context, contours, and available
   predictor diagnostics.
3. `publish_latest_ml_output.sh` validates and publishes the forecast,
   refreshes the latest product, and maintains the date archive.
4. `publish_verification_output.sh` adds the post-event Practically Perfect
   layer and verification graphics after the valid period.
5. `generate_dashboard_data.py` rebuilds daily and pooled weekly, monthly, and
   seasonal issued-forecast verification.
6. Git commits and GitHub Pages deployment make the updated forecast and
   verification available publicly.

Codex helped add schema and required-layer gates, stale-output protection,
archive/status rebuilding, resilient Git synchronization behavior, public-data
sanitization, rolling-verification aggregation, selected-risk case counts, and
live deployment checks. These checks reduce the chance that an incomplete or
stale run is presented as the latest forecast.

## Human scientific and product responsibility

GPT-5.6/Codex supported software implementation, refactoring, testing,
documentation, and deployment. Human-authored project logic remained
authoritative for:

- flood-proxy and Practically Perfect definitions;
- XGBoost classification targets and predictor selection;
- 5%, 15%, 40%, and 70% risk thresholds;
- neighborhood and report-expansion radii;
- training/test case selection;
- finalized figure selection and scientific interpretation; and
- decisions about which products and statistics belong on the public site.

GPT-5.6/Codex did not replace scientific review, originate official NWS
guidance, or convert this experimental product into an official forecast.

## Development workflow

The Build Week collaboration followed a reviewable engineering loop:

1. inspect the existing code and saved scientific outputs;
2. restate the requested scientific or interface contract;
3. make focused edits without including unrelated working-tree changes;
4. regenerate affected data and figures;
5. run proportional static, unit, schema, and numerical checks;
6. review the staged diff;
7. commit and open a GitHub pull request; and
8. merge, monitor GitHub Pages, and verify the public files.

This workflow let the human project lead give rapid scientific and product
feedback while GPT-5.6/Codex handled much of the repository-scale inspection,
implementation, consistency checking, and deployment verification.

## Representative repository artifacts

| Area | Key artifacts |
| --- | --- |
| Website | `docs/index.html`, `docs/app.js`, `docs/style.css`, `docs/briefing.js` |
| Dashboard data | `generate_dashboard_data.py`, `docs/model-skill/`, `docs/verification/` |
| Real-time maps | `realtime_mcs_trigger_plot.py`, `generate_interactive_map_data.py` |
| Publishing | `publish_latest_ml_output.sh`, `publish_verification_output.sh`, `realtime_ml.crontab` |
| Efficient training | `hazard_ml_training_v28_r100km_singletarget_radiusstats_regression_MEMSAFE_V3.py` |
| Radius workflows | `run_hazard_ml_v33_radius_sensitivity_from_WORKING_v28_radiusstats_SLIMMASTER_ROWSAMPLE.sh` and its generator |
| Validation | `tests/test_dashboard_data.py`, `tests/test_briefing.js`, publisher schema checks |

## Representative development milestones

- `f481af6`: standalone HRRR MCS-triggered real-time ML plotter
- `647929d`: GitHub Pages site for real-time ML output
- `3ce5f3b`: stabilized real-time RAP prediction pipeline
- `e0d9d3a`: interactive ML forecast map
- `690a83d`: live radar and daily-publishing hardening
- `04d04fa`: SHAP predictor overlays and radar improvements
- `b74ef5a`: flood alerts and multi-radius predictor layers
- `f7a84ee`: XGBFFP Location Briefing and evaluation dashboard
- `03988de`: corrected authoritative model-skill figures
- `764030f`: added Hits and removed unneeded MRMS-over-FFG skill figures
- `cc1c2da`: added selected-risk case totals and removed Brier Skill Score

## Disclaimer

XGBFFP is experimental machine-learning guidance. It is not an official
National Weather Service forecast, watch, or warning. Users should evaluate it
alongside official forecasts, observations, and established operational
decision-support practices.
