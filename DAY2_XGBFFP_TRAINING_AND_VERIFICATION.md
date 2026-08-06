# XGBFFP Day-2 Training and Verification

This is a separate workflow for **XGBoosted Flash Flood Predictions (XGBFFP)**
at Day 2. It does not replace or reuse the Day-1 models, daily feature chunks,
master parquets, model aliases, WPC caches, or verification outputs.

## Forecast and target alignment

For an event case date **V**:

- Event and verification window: V at 12Z through V+1 at 12Z, exactly matching
  Day 1 for the same case.
- RAP initialization: V-1 at 09Z.
- Predictor valid offsets relative to V-1 at 12Z: 0, 6, 12, 18, 24, 30, 36,
  42, and 48 hours.
- RAP files: f03, f09, f15, f21, f27, f33, f39, f45, and f51.
- Retained feature family: the complete existing 0–24 h instantaneous and APCP
  summaries.
- New feature family: separate 24–48 h instantaneous summaries, 6-hour APCP
  intervals, maximum 6- and 12-hour APCP, 24–48 h total APCP, and corresponding
  guidance-FFG ratios.
- Guidance FFG predictors: latest FFG available at or before V-1 at 09Z.
- Target window: V at 12Z through V+1 at 12Z.
- Binary targets: any MRMS QPE/FFG exceedance within 40, 60, 75, or 100 km.
- Model family: one `XGBClassifier` per radius using `binary:logistic`.

Training, realtime prediction, and the historical viewer all use **V** as
`Date`. They separately retain `RAP_Init_Date=V-1`. For example, case
`20240620` uses RAP
initialized at 09Z `20240619` and is verified against WPC, UFVS, and MRMS for
12Z `20240620` through 12Z `20240621`.

Day-1 RAP feature chunks are explicitly disabled because they stop at valid
offset 24 and cannot contain the required f33–f51 predictors. The optional
previous-target feature fallback is disabled because a previous Day-2 target
would extend past the current RAP initialization and leak future observations.
Unrelated LSR retrieval is also disabled because these targets use only MRMS
exceedance of FFG.

## Train all four radii

```bash
./run_hazard_ml_v33_day2_radius_sensitivity_from_WORKING_v28_radiusstats_SLIMMASTER_ROWSAMPLE.sh
```

The default Optuna budget is 30 XGBoost trials per radius. Override only when
intended:

```bash
HAZARD_ML_LOCAL_OPTUNA_XGB_TRIALS=30 \
RADIUS_LIST="40 60 75 100" \
./run_hazard_ml_v33_day2_radius_sensitivity_from_WORKING_v28_radiusstats_SLIMMASTER_ROWSAMPLE.sh
```

The runner is resume-aware. It validates each radius manifest and master,
including artifacts relocated from `ISU_Research` to
`ISU_Research_LOCAL_RUN`, skips completed radii, and reuses every completed
daily feature chunk for an incomplete radius. Before continuing an incomplete
radius it reports cached feature storage, estimated final-master size, net
additional storage, current free space, and a recommended free-space value
with a 5 GiB safety margin. An interrupted monolithic `.parquet.tmp` master is
not appendable; only that combine is restarted from the already-built daily
chunks. Feature extraction and completed-radius training are not repeated.
Resume mode also treats the completed chunk filenames as the established Day-2
case set, so it does not repeat the historical RAP source-availability
preflight before recombining the master.

To generate and inspect all four training programs without starting extraction
or training:

```bash
DRY_RUN=1 \
./run_hazard_ml_v33_day2_radius_sensitivity_from_WORKING_v28_radiusstats_SLIMMASTER_ROWSAMPLE.sh
```

Generated programs go to
`generated_v33_day2_radius_sensitivity_slimmaster_rowsample/`. Models and
manifests use the unique `v33day2valid` tag. The earlier `v33day2` artifacts
used the incorrect case-date contract and were purged; they must not be reused.

## Run the verification viewer

After all requested models and masters exist:

```bash
jupyter lab hazard_ml_v33_day2_verification_viewer.ipynb
```

The Day-2 notebook is a full copy of the current v33 Day-1 viewer, with the
forecast and observation routing changed to Day 2. It is not the earlier
reduced HTML-only verifier. The copied notebook retains:

- historical test-set case prediction maps;
- realtime feature generation, model prediction, WPC retrieval, and
  observation matching;
- all four trained ML target-radius members: R40, R60, R75, and R100;
- SHAP feature definitions, summary plots, dependence plots, and paper figures;
- categorical occurrence and contingency tables;
- hits, misses, false alarms, correct negatives, CSI, ETS, POD, FAR, and bias;
- reliability and Brier diagnostics;
- PP and proxy-based verification tables and violin plots;
- risk-area, area-error, heatmap, centroid, and displacement diagnostics.

All copied caches and figure directories use `v33day2valid` names so running this
notebook cannot silently load or overwrite a Day-1 viewer product.

### Test-prediction memory and resume behavior

Historical prediction generation is bounded-memory. The notebook does not load
all test-year predictors into one dataframe. It reads the master parquet in
12,000-row batches, runs the saved scaler and XGBoost model on one batch, and
writes a narrow prediction fragment containing only date, location, model,
radius, and probability.

The batch size can be reduced before launching Jupyter on a smaller-memory
machine:

```bash
XGBFFP_DAY2_PREDICT_BATCH_ROWS=6000 jupyter lab \
  hazard_ml_v33_day2_verification_viewer.ipynb
```

Fragments are stored beside each consolidated prediction cache under a
`.parquet.parts/<artifact-signature>/` directory. If the kernel stops, rerun
the main viewer cell: valid completed fragments are reused and only unfinished
batches are predicted. The artifact signature includes the model, scaler,
feature list, master parquet, test years, and batch size, preventing fragments
from a different trained model or configuration from being mixed in.

The existing completed R40 consolidated cache remains reusable. The streaming
builder starts with the first radius whose consolidated cache is absent.

To build the caches independently of Jupyter or VS Code notebook state, run:

```bash
python build_v33day2_test_predictions_memsafe.py --batch-rows 6000
```

This is the preferred recovery path after an out-of-memory kernel termination.
It processes one radius at a time, normalizes older caches to the narrow
eight-column schema (including both valid and RAP initialization dates), and
reports process peak RSS. Once it finishes, reopening
the notebook only loads the four compact prediction caches and does not read
the 820-feature training masters during test-set setup.

### Verification truth branches

The notebook preserves both verification comparisons from the original v33
viewer. They are intentionally separate:

1. **Practically Perfect:** probability-like `PP_*` fields built for the Day-2
   valid window. These retain the original viewer's expansion and weighted
   smoothing procedure.
2. **UFVS flood proxies:** raw UFVS occurrence fields such as Stage IV/FFG,
   Stage IV ARI, USGS, flash-flood LSR, and the combined `UFVS_ANY` field. The
   raw occurrence mask is expanded to a **40-km radius** before proxy-truth
   scoring. It is not substituted with a practically-perfect field.

Pointwise Day-2 MRMS QPE/FFG exceedance is read from
`Obs_Day2_MRMS_FFG_Exceeded_Point` in the Day-2 master and is used when the
original viewer requests MRMS/FFG truth or builds the associated
practically-perfect field.

### WPC handling

The notebook requests the IEM WPC ERO archive with `type=E,d=2`, requires the
valid period to match V 12Z through V+1 12Z, and selects the latest product
revision for that exact valid window. Day-2 WPC polygon and raster caches use
the corrected `v33day2valid` namespace.

WPC has no ML target radius. It is rasterized once on the common Day-2
verification grid and yielded once by the verification-source iterator. Radius
comparisons therefore change only the selected ML model; with a fixed forecast
threshold and truth definition, the WPC values and skill are identical across
the R40/R60/R75/R100 comparison. The notebook includes an invariant check that
rejects verification tables assigning 40-, 60-, 75-, or 100-km labels to WPC.
If the IEM archive contains no polygon set for the exact Day-2 valid window,
that case is retained with WPC marked unavailable (`NaN`) and is omitted only
from WPC scores; the notebook does not invent a zero-risk WPC forecast.

### Realtime Day-2 routing

Realtime generation imports only the programs under
`generated_v33_day2_radius_sensitivity_slimmaster_rowsample/`. It uses RAP
V-1 f03-f51 predictors, fetches the same V-to-V+1 UFVS observations used for
Day 1, and fetches the matching WPC Day-2 ERO. Realtime prediction and
verification caches are written below
`$HAZARD_ML_PROJECT_DIR/v33day2valid_realtime_radiusstats_forecasts/`.

The notebook defaults SHAP to the trained Day-2 `r100km` member. Change
`SHAP_RADIUS_KM` and `SHAP_MODEL_LABEL` together to inspect another Day-2
member.
