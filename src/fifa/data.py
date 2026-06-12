"""Downloads, caching, cleaning, and team-name normalization."""
from __future__ import annotations

import re
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


def load_results(force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (played, upcoming). Played: scores int, chronological. Upcoming: NA-score rows."""
    path = download(RESULTS_URL, DATA_DIR / "results.csv", force=force)
    df = pd.read_csv(path, na_values=["NA"])
    df["date"] = pd.to_datetime(df["date"])
    df["neutral"] = df["neutral"].astype(bool)
    for col in ("home_team", "away_team"):
        df[col] = df[col].replace(SUCCESSORS)
    upcoming = df[df["home_score"].isna()].copy().reset_index(drop=True)
    played = df.dropna(subset=["home_score", "away_score"]).copy()
    played["home_score"] = played["home_score"].astype(int)
    played["away_score"] = played["away_score"].astype(int)
    played = played.sort_values("date", kind="stable").reset_index(drop=True)
    return played, upcoming


def load_shootouts(force: bool = False) -> pd.DataFrame:
    path = download(SHOOTOUTS_URL, DATA_DIR / "shootouts.csv", force=force)
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    for col in ("home_team", "away_team", "winner"):
        df[col] = df[col].replace(SUCCESSORS)
    return df


# FIFA/fixture-feed names → martj42 dataset names. Identity names omitted.
# The live completeness check (Task 4 Step 5 / tests) is the source of truth —
# extend this map if it reports unmapped names.
FIFA_ALIASES = {
    "Korea Republic": "South Korea",
    "Korea DPR": "North Korea",
    "Czechia": "Czech Republic",
    "IR Iran": "Iran",
    "Côte d'Ivoire": "Ivory Coast",
    "Cote d'Ivoire": "Ivory Coast",
    "Cabo Verde": "Cape Verde",
    "Türkiye": "Turkey",
    "Turkiye": "Turkey",
    "Congo DR": "DR Congo",
    "China PR": "China",
    "USA": "United States",
    "Curacao": "Curaçao",
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",  # The Odds API spelling
}
TBD_PLACEHOLDER = "To be announced"

# Knockout seed tokens in the fixture feed: "1A" = group A winner, "2B" = runner-up B,
# "3CDFGH" = third-placed team from one of those groups. Not team names.
_SEED_RE = re.compile(r"^[123][A-L]{1,6}$")


def is_placeholder(name: str) -> bool:
    return name == TBD_PLACEHOLDER or bool(_SEED_RE.match(name))


def normalize_team(name: str) -> str:
    return FIFA_ALIASES.get(name, name)


def assert_known(names, known_teams) -> None:
    unknown = sorted({n for n in names if n not in known_teams and not is_placeholder(n)})
    if unknown:
        raise ValueError(f"Unmapped team names (add to data.FIFA_ALIASES): {unknown}")
