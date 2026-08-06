# V34 compact Day-1 XGBFFP training

This is a new training family. It does not overwrite or reinterpret the v33
40-km and 60-km experiments.

## Experiment matrix

| Target radius | MRMS QPE / FFG threshold | Role |
|---:|---:|---|
| 75 km | >= 1.0 | Default |
| 100 km | >= 1.0 | Default |
| 75 km | >= 1.5 | Sensitivity test |
| 100 km | >= 1.5 | Sensitivity test |
| 75 km | >= 2.0 | Sensitivity test |
| 100 km | >= 2.0 | Sensitivity test |

Every target is binary: a row is positive when at least one valid native RAP
grid point inside the stated radius has an observed MRMS QPE / FFG ratio at or
above the threshold during the 12Z-to-12Z verification window. The comparison
in the generated code is inclusive (`>=`). Flood/flash-flood LSRs remain
verification information and are not unioned into these targets.

All six models use `XGBClassifier` with `binary:logistic`, grouped-date cross
validation, and unweighted log loss. There is no regression model.

## Domain

The generator asserts and preserves the existing Day-1 bounds:

- latitude 30.0 to 50.0 degrees north
- longitude -105.0 to -80.5 degrees east

This is the same east-of-the-Rockies domain with the East Coast excluded. The
generator fails if these constants drift in the authoritative base script.

## Compact predictor contract

The authoritative v33 100-km Day-1 XGBoost model and feature list were read
from:

- `/home/tyreekfrazier/ISU_Research_LOCAL_RUN/fall_2025_ml_proj/prob_flood_models/current_v33_r100km_XGBoost_model.pkl`
- `/home/tyreekfrazier/ISU_Research_LOCAL_RUN/fall_2025_ml_proj/prob_flood_models/current_v33_r100km_feature_names.json`

The exact SHA-256 values and top-20 gain ranking are frozen in
`day1_compact_feature_contract.py`. The 20 final features collapse to these 10
base families:

1. maximum forecast 6-hour QPF / 6-hour guidance FFG ratio
2. SBCAPE
3. forecast QPF / guidance FFG ratio across 6-, 12-, and 24-hour durations
4. MLCIN
5. guidance FFG across 1-, 3-, 6-, 12-, and 24-hour durations
6. 10-m U wind
7. MLCAPE
8. forecast precipitation running totals through 24 hours
9. RAP-calculated MCS-maintenance probability
10. 10-m V wind

The code rebuilds all applicable temporal/duration mean, minimum, maximum, and
standard-deviation summaries for those families, followed by the same four
spatial summaries at the target radius. This produces 148 predictors.

RAP `REFC` simulated composite reflectivity is added at the five forecast
valid times. It receives temporal and spatial mean/min/max/std summaries,
adding 16 predictors. Reflectivity is kept in dBZ without the legacy linear
3-by-3 prefilter. The final contract is therefore exactly 164 predictors for
each model.

The run fails instead of silently dropping a compact predictor that is absent,
all-NaN, or constant. Prior RAP feature chunks are not reused because they do
not contain simulated reflectivity.

## Generate or run

Generate and syntax-check all six standalone training scripts without
starting training:

```bash
./run_hazard_ml_v34_compact_thresholds.sh --generate-only
```

Print the six training commands without executing them:

```bash
./run_hazard_ml_v34_compact_thresholds.sh --dry-run
```

Run the complete matrix sequentially:

```bash
./run_hazard_ml_v34_compact_thresholds.sh
```

Run only the default 1.0-ratio pair:

```bash
./run_hazard_ml_v34_compact_thresholds.sh --ratios 1.0
```

Run one sensitivity experiment and force retraining:

```bash
./run_hazard_ml_v34_compact_thresholds.sh --radii 75 --ratios 1.5 --force
```

Generated scripts are written under `generated_v34_compact_day1/`. Model,
scaler, feature-list, results, manifest, daily-chunk, and master-parquet names
contain both the radius and ratio tag, so the six experiments cannot overwrite
one another. A run is skipped only when its manifest, master parquet, target
radius, threshold, 164-feature count, and referenced artifacts are complete.
