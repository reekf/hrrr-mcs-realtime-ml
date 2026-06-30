#!/usr/bin/env python3
"""Report WPC ERO threshold counts in the v33 verification input grid."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_INPUT = Path(
    "/home/tyreekfrazier/ISU_Research_LOCAL_RUN/fall_2025_ml_proj/"
    "df_pp_viewer_with_wpc_ero_day1.parquet"
)
THRESHOLDS = (0.05, 0.15, 0.40, 0.70)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dates", nargs="+", help="Forecast valid-start dates (YYYYMMDD)")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="v33 PP/WPC parquet")
    args = parser.parse_args()

    path = Path(args.input).expanduser().resolve()
    df = pd.read_parquet(path, columns=["Date", "WPC_ERO_Risk"])
    df["Date"] = df["Date"].astype(str).str.slice(0, 8)
    df["WPC_ERO_Risk"] = pd.to_numeric(df["WPC_ERO_Risk"], errors="coerce").fillna(0.0)

    print(f"Input: {path}")
    missing = False
    for date in args.dates:
        date = str(date)[:8]
        values = df.loc[df["Date"] == date, "WPC_ERO_Risk"]
        if values.empty:
            print(f"{date}: MISSING")
            missing = True
            continue
        counts = " ".join(f"N >= {threshold:.2f} = {(values >= threshold).sum():,}" for threshold in THRESHOLDS)
        print(f"{date}: rows={len(values):,} Max WPC_ERO_Risk = {values.max():.2f}; {counts}")

    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
