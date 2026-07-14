"""Shared date loader for the deduplicated MODE 24-hour case catalog."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path


CASE_LIST_CSV = Path(__file__).resolve().with_name("MODE_24h_500cases.csv")


def load_unique_case_dates(path: str | Path = CASE_LIST_CSV) -> list[str]:
    path = Path(path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "Date" not in rows[0]:
        raise ValueError(f"{path} is empty or lacks a Date column")

    values = [str(row.get("Date", "")).strip() for row in rows]
    split = [value.split("/") for value in values]
    if any(len(parts) != 3 for parts in split):
        raise ValueError(f"Unparseable slash-formatted date in {path}")
    first = [int(parts[0]) for parts in split]
    second = [int(parts[1]) for parts in split]
    mdy_evidence = any(value > 12 for value in second)
    dmy_evidence = any(value > 12 for value in first)
    if mdy_evidence and dmy_evidence:
        raise ValueError(f"Mixed day/month orientation in {path}")
    fmt = "%m/%d/%Y" if mdy_evidence else "%d/%m/%Y"
    dates = sorted({datetime.strptime(value, fmt).strftime("%Y%m%d") for value in values})
    print(f"Case catalog {path}: {len(rows)} rows -> {len(dates)} unique dates; format={fmt}")
    return dates
