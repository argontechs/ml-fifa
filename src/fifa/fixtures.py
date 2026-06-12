"""fixturedownload.com WC2026 feed client."""
from __future__ import annotations

import json

import pandas as pd

from . import data

# Exact feed Location string → host nation (martj42 team naming). All 16 WC2026 venues.
VENUE_HOST = {
    "Mexico City Stadium": "Mexico",
    "Guadalajara Stadium": "Mexico",
    "Monterrey Stadium": "Mexico",
    "Toronto Stadium": "Canada",
    "BC Place Vancouver": "Canada",
    "Atlanta Stadium": "United States",
    "Boston Stadium": "United States",
    "Dallas Stadium": "United States",
    "Houston Stadium": "United States",
    "Kansas City Stadium": "United States",
    "Los Angeles Stadium": "United States",
    "Miami Stadium": "United States",
    "New York/New Jersey Stadium": "United States",
    "Philadelphia Stadium": "United States",
    "San Francisco Bay Area Stadium": "United States",
    "Seattle Stadium": "United States",
}


def _host_of(location: str) -> str:
    if location not in VENUE_HOST:
        raise ValueError(f"Unmapped venue {location!r} — add to fixtures.VENUE_HOST")
    return VENUE_HOST[location]


def load_fixtures(force: bool = False) -> pd.DataFrame:
    path = data.download(data.FIXTURES_URL, data.DATA_DIR / "fixtures.json",
                         max_age_hours=6, force=force)
    feed = json.loads(path.read_text())
    rows = []
    for m in feed:
        home = data.normalize_team(m["HomeTeam"])
        away = data.normalize_team(m["AwayTeam"])
        tbd = data.is_placeholder(m["HomeTeam"]) or data.is_placeholder(m["AwayTeam"])
        played = m["HomeTeamScore"] is not None
        host = _host_of(m["Location"])
        rows.append({
            "match_number": m["MatchNumber"],
            "round": m["RoundNumber"],
            # naive UTC — the rest of the system (results.csv dates) is tz-naive
            "date": pd.to_datetime(m["DateUtc"]).tz_localize(None),
            "location": m["Location"],
            "host": host,
            "home": home,
            "away": away,
            "group": (m["Group"] or "").replace("Group ", "") or None,
            "home_score": m["HomeTeamScore"],
            "away_score": m["AwayTeamScore"],
            "status": "tbd" if tbd else ("played" if played else "upcoming"),
            "neutral": home != host,
        })
    return pd.DataFrame(rows).sort_values("match_number").reset_index(drop=True)
