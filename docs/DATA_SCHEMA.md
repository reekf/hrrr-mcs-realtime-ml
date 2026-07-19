# XGBFFP Website Data Schemas

All paths are relative to `docs/`. Missing optional fields must be rendered as
“Not available”; they must not be replaced with fabricated or zero values.

## Forecast map JSON — schema version 5

Path: `archive/YYYYMMDD/map.json` (and `latest/map.json`)

Core fields:

```text
schema_version
date
valid_period_label
generated_utc
source_class                 realtime | historical
probability_encoding         integer 0..1000; divide by 10 for percent
risk_threshold_percent[]     [5, 15, 40, 70]
grid.lat[]
grid.lon[]
layers.<product>.label
layers.<product>.kind
layers.<product>.values[]
contours.<product>.<threshold>[]
observations.<proxy>.points[]
predictors.r<radius>.<name>
```

Every layer/predictor value array aligns by index with `grid.lat` and
`grid.lon`. Predictor values are normalized 0–1000 positions between
`scale_min` and `scale_max`; metadata includes units, global SHAP rank, and
direction. They are raw predictor diagnostics, not local SHAP values.

Older archives can omit r60kmV2, ensemble mean, PP, predictors, or
observations. The consumer detects availability per date.

## Skill manifest — schema version 1

Path: `model-skill/manifest.json`

The top-level `dataset_class` is `formal-independent-test-set`. Each `figures`
entry records title, metric, target, threshold list, test period, model,
source script/function, generation timestamp, and repo-relative image `path`.
The publisher fails if a referenced path is missing.

`model-skill/risk-frequency.json` contains pooled false-alarm and miss grid-cell
counts derived from the final PP ETS contingency-count table. Local PMM,
ensemble maximum, and r100kmV2 are intentionally excluded from this comparison.
It remains labeled as formal test-set data.

## Explainability manifest — schema version 1

Path: `explainability/manifest.json`

The `dataset_class` is `formal-independent-test-set-explainability`. Figure
entries include model, kind (`beeswarm`, `importance`, or `dependence`), test
period, source function, timestamp, and path.

## Daily realtime verification — schema version 1

Path: `verification/daily/YYYYMMDD.json`

Required fields:

```text
dataset_class                 realtime-issued-verification
verification_target           Practically Perfect: Any flood proxy
date
valid_period_label
products.<product>.<threshold>
```

Threshold records contain non-negative contingency counts, sample count, truth
and forecast positive counts, squared-error sum, ETS, CSI, POD, FAR, frequency
bias, Brier Score, and BSS. Undefined metrics are `null`.

Only maps with `source_class == "realtime"` and an actual `layers.pp` array are
eligible.

## Rolling realtime verification — schema version 1

Paths:

```text
verification/rolling/latest.json
verification/rolling/weekly.json
verification/rolling/monthly.json
verification/rolling/seasonal.json
```

Each window records its definition, start/end dates, verified dates and count,
expected calendar days, missing-day count, completeness, target, and pooled
product/threshold metrics. `latest.json` embeds all three windows for one
browser request.

`verification/index.json` lists available daily dates and paths.

## Validation and versioning

`generate_dashboard_data.py` writes JSON with non-finite values disallowed,
validates aligned layer lengths, requires real finalized static figures, and
validates manifest paths. New incompatible contracts require a schema-version
increment; consumers must continue treating absent newer fields as optional for
archive compatibility.
