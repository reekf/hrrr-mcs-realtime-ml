#!/usr/bin/env python3
"""Fetch and sanitize mPING flood reports for one 12Z-to-12Z valid period."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path


API_URL = "https://mping.ou.edu/mping/api/v2/reports"
ALLOWED_HOST = "mping.ou.edu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True, help="Forecast start date in YYYYMMDD format")
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def get_json(url: str, token: str) -> dict:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"} or parsed.hostname != ALLOWED_HOST:
        raise RuntimeError(f"Refusing unexpected mPING pagination URL: {url}")
    if parsed.scheme == "http":
        url = urllib.parse.urlunparse(parsed._replace(scheme="https"))
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "Authorization": f"Token {token}",
            "User-Agent": "ISU-realtime-ML-flood-guidance/1.0",
        },
    )
    with urllib.request.urlopen(request, timeout=45) as response:
        return json.load(response)


def main() -> int:
    args = parse_args()
    token = os.environ.get("MPING_API_TOKEN", "").strip()
    token_path = Path(os.environ.get("MPING_API_TOKEN_FILE", "~/.config/realtime-ml/mping-token")).expanduser()
    if not token and token_path.is_file():
        token = token_path.read_text().strip()
    if not token:
        print(
            f"No mPING token found in MPING_API_TOKEN or {token_path}; "
            "leaving any existing public mPING file unchanged.",
            file=sys.stderr,
        )
        return 3

    start = datetime.strptime(args.date, "%Y%m%d").replace(tzinfo=timezone.utc, hour=12)
    end = start + timedelta(days=1)
    query = urllib.parse.urlencode(
        {
            "category": "Flood",
            "obtime_gte": start.strftime("%Y-%m-%d %H:%M:%S"),
            "obtime_lt": end.strftime("%Y-%m-%d %H:%M:%S"),
            "in_bbox": "-105.1,30,-80.4,50.1",
        }
    )
    url: str | None = f"{API_URL}?{query}"
    reports: list[dict] = []
    seen: set[str] = set()
    while url:
        payload = get_json(url, token)
        for item in payload.get("results", []):
            coordinates = (item.get("geom") or {}).get("coordinates") or []
            if len(coordinates) < 2:
                continue
            try:
                lon, lat = float(coordinates[0]), float(coordinates[1])
                valid = datetime.fromisoformat(str(item.get("obtime", "")).replace("Z", "+00:00"))
            except (TypeError, ValueError):
                continue
            if not start <= valid < end or not (-105.1 <= lon <= -80.4 and 30 <= lat <= 50.1):
                continue
            report_id = str(item.get("id", ""))
            dedupe_key = report_id or f"{valid.isoformat()}:{lat:.5f}:{lon:.5f}"
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            reports.append(
                {
                    "id": report_id or None,
                    "valid": valid.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
                    "description": str(item.get("description") or "Flood impact"),
                    "lat": round(lat, 5),
                    "lon": round(lon, 5),
                }
            )
        next_url = payload.get("next")
        url = urllib.parse.urljoin(API_URL, next_url) if next_url else None

    reports.sort(key=lambda report: (report["valid"], report["id"] or ""))
    public = {
        "date": args.date,
        "valid_start_utc": start.isoformat().replace("+00:00", "Z"),
        "valid_end_utc": end.isoformat().replace("+00:00", "Z"),
        "source": "mPING",
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "reports": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(public, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {len(reports)} sanitized mPING flood reports to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
