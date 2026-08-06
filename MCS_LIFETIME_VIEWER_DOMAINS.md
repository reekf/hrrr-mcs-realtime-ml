# MCS lifetime-centered Day-1 viewer domains

The Day-1 viewer uses one exact, adjustable local azimuthal-equidistant square
for every case. Its center is the equal-time mean of the selected PyFLEXTRKR
MCS-stage centroid series over the 12Z-to-12Z valid period. The committed
manifest was built with 400 x 400 km boxes, but its track centers can be
resized at viewer time without rerunning RAP or PyFLEXTRKR.

## RAP and PyFLEXTRKR inputs

- RAP 09Z forecast hours 003 through 027 provide hourly fields valid from
  12Z through 12Z.
- The cloud-top field is RAP `SBT124`, the GOES-12 channel-4 longwave infrared
  brightness temperature. `SBT123` is the water-vapor-like channel and is not
  suitable for the 241 K cold-cloud-shield threshold.
- Composite reflectivity (`REFC`) supplies precipitation-feature information.
- If full cached RAP GRIB2 files are unavailable, the builder reads the NOAA
  AWS `.idx` files and downloads only the `SBT124` and `REFC` byte ranges.
- The adapter records its configuration and source-field provenance beneath
  each case directory so cached results are reused only when inputs and
  thresholds match.

PyFLEXTRKR can identify several unrelated MCSs across CONUS in one valid
period. The builder disambiguates them by selecting the lifetime-mean MCS
centroid nearest the centroid of the highest WPC ERO risk category. WPC is
only a case-location anchor; it does not determine the MCS center. If a case
has no positive WPC risk, the existing viewer-grid center is the explicit
fallback. The selected center, track number, duration, anchor distance, source
NetCDF path, and source modification time are written to the domain manifest.

The historical ML feature/prediction grid is smaller than CONUS. After track
selection, the builder verifies that the complete projected box is inside that
grid. A case that would be clipped is listed under `excluded_cases` in the
manifest and is omitted from plots and scores. It is never evaluated using a
partial box; adding such a case requires rebuilding ML features and predictions
on a broader source grid.

## Build

Run with the environment that contains PyFLEXTRKR and its RAP dependencies:

```bash
/home/tyreekfrazier/.conda/envs/xgbffp-pyflextrkr/bin/python \
  build_rap_mcs_lifetime_domains.py --workers 4
```

The default output is:

```text
./mcs_lifetime_domains_400km.json
```

The manifest is committed beside the viewer so the notebook does not depend on
an untracked machine-local cache. Launch Jupyter from the repository directory,
or set `XGBFFP_MCS_DOMAIN_JSON` to the manifest's absolute path.

Use `--stats-only` to regenerate the JSON from existing track NetCDF files
without rerunning PyFLEXTRKR.

## Plotting and verification contract

The notebook requires the domain manifest and filters every case using the
same projected-square mask. ML forecasts, WPC ERO, and official NOAA 2.5-km
Practically Perfect fields are all displayed inside the matching case domain.
Change `Box km` in the notebook controls and click **Plot** to resize the box
around the same MCS lifetime center. The default can also be changed with the
single `MCS_LIFETIME_BOX_KM` setting near the top of the notebook. Keep
`MCS_LIFETIME_DOMAIN_JSON` pointed at `mcs_lifetime_domains_400km.json`; the
filename identifies the source manifest, not the active viewer size.

If a requested box would extend beyond the available ML grid for a case, that
case is removed from the date selector and verification metrics at that size.
The viewer prints the excluded dates so a clipped domain cannot be counted as
an implicit correct negative.

For neighborhood verification, each forecast/truth field is first expanded on
the complete available viewer grid. The resulting masks are then cropped to
the case's selected square. This order prevents an artificial loss of
neighborhood influence at the box boundary while ensuring every score uses
only the requested MCS-centered region. The domain size is included in the
metric-cache tag to prevent reuse of older fixed-domain scores.

Analysis maps render ML, WPC, Practically Perfect, and agreement fields as
filled grid-cell polygons rather than fixed-size scatter markers. The cells
scale with the map when zoomed, and the `Cell fill` control provides a small
adjustable overlap to eliminate renderer hairline gaps. This change is local
to the notebook analysis maps; it does not alter the website's circular-dot
visual design.
