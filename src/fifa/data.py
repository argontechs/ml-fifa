"""Downloads, caching, cleaning, and team-name normalization."""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
UA = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
}

_BASE = "https://raw.githubusercontent.com/martj42/international_results/master/"
RESULTS_URL = _BASE + "results.csv"
SHOOTOUTS_URL = _BASE + "shootouts.csv"
FIXTURES_URL = "https://fixturedownload.com/feed/json/fifa-world-cup-2026"

# eloratings.net-style successor chains missing from upstream (USSR→Russia and
# West Germany→Germany are already merged in results.csv; German DR terminates).
SUCCESSORS = {"Yugoslavia": "Serbia", "Czechoslovakia": "Czech Republic"}


def download(url: str, dest: Path, max_age_hours: float = 12.0, force: bool = False) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    fresh = dest.exists() and (time.time() - dest.stat().st_mtime) < max_age_hours * 3600
    if fresh and not force:
        return dest
    try:
        resp = requests.get(url, headers=UA, timeout=30)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
    except requests.RequestException as exc:
        if dest.exists():
            age_h = (time.time() - dest.stat().st_mtime) / 3600
            print(f"WARNING: download failed ({exc}); using cache {dest.name} ({age_h:.0f}h old)")
        else:
            raise RuntimeError(f"Cannot download {url} and no cache at {dest}") from exc
    return dest
