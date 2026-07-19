# XGBFFP Metrics and Aggregation

## Risk thresholds

Categorical thresholds are inclusive:

- Marginal or greater: probability `>= 0.05`
- Slight or greater: probability `>= 0.15`
- Moderate or greater: probability `>= 0.40`
- High: probability `>= 0.70`

Map JSON encodes probability in tenths of a percent, so the exact encoded
boundaries are 50, 150, 400, and 700.

## Categorical metrics

Let `H` be hits, `M` misses, `F` false alarms, and `C` correct negatives.

- CSI: `H / (H + M + F)`. Higher is better.
- POD: `H / (H + M)`. Higher is better.
- FAR: `F / (H + F)`. Lower is better.
- Frequency bias: `(H + F) / (H + M)`. One is unbiased frequency; it does not
  measure spatial accuracy.
- ETS subtracts expected random hits from CSI's numerator and denominator.
  Higher is better; zero represents no improvement over random overlap.

Undefined divisions are stored as JSON `null`, never infinity.

## Probabilistic metrics

- Brier Score is the mean squared difference between a forecast probability and
  a binary outcome. Lower is better.
- RPSS is a ranked-probability skill score for multi-category forecasts. It is
  defined by the source notebook but is not published in the initial realtime
  JSON because the archive does not preserve the required ranked components.

## Practically Perfect

Practically Perfect (PP) is created after the valid period from observed
flood-proxy locations, spatial expansion, and smoothing. It is an idealized
observation-based reference, not an operational forecast. PP ETS compares
categorical spatial overlap independently at all four thresholds.

## UFVS flood proxies

Final test-set figures use the exact proxy labels saved by the final viewer:
MRMS rainfall exceeding FFG, Stage IV rainfall exceeding FFG, Stage IV ARI,
USGS flood points, flash-flood LSRs, flood LSRs, and their final combined
`UFVS_ANY` definition. Raw point proxies in the final plots are expanded 40 km
where recorded by the source table. The website does not reinterpret them.

## Realtime pooling

For ETS, CSI, POD, FAR, and frequency bias, XGBFFP sums `H`, `M`, `F`, and `C`
across verified dates and recalculates each metric. It does not average daily
scores.

For Brier Score, squared-error sums and sample counts are pooled. Every metric
reports verified forecast count and grid sample count. Each product/threshold
also reports the number of verified cases containing at least one forecast
grid cell at or above the selected risk threshold.

The verification target is `Practically Perfect: Any flood proxy`. Formal
2024–2025 test cases and realtime-issued forecasts remain separate datasets.

## Windows

- Weekly: latest seven verified forecasts. The calendar gaps between the first
  and last included forecast are reported.
- Monthly: trailing 30 calendar days ending at the latest verified forecast.
- Seasonal: latest meteorological season to date: DJF, MAM, JJA, or SON.
  December belongs to the DJF season ending the following February.

Missing-day count is expected calendar days minus verified forecast dates.
Limited-sample warnings are shown below ten verified forecasts.
