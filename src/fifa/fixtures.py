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


def load_fixtures(force: bool = False, max_age_hours: float = 6.0) -> pd.DataFrame:
    path = data.download(data.FIXTURES_URL, data.DATA_DIR / "fixtures.json",
                         max_age_hours=max_age_hours, force=force,
                         validator=lambda b: json.loads(b))
    feed = json.loads(path.read_text())
    rows = []
    for m in feed:
        home = data.normalize_team(m["HomeTeam"])
        away = data.normalize_team(m["AwayTeam"])
        tbd = data.is_placeholder(m["HomeTeam"]) or data.is_placeholder(m["AwayTeam"])
        home_score, away_score = m["HomeTeamScore"], m["AwayTeamScore"]
        played = home_score is not None and away_score is not None
        host = _host_of(m["Location"])
        w_raw = m.get("Winner") or ""
        winner = data.normalize_team(w_raw) if w_raw and not data.is_placeholder(w_raw) else None
        if away == host and not tbd:
            # FIFA lists the host as the designated away side in a few fixtures; our
            # models grant home advantage to the home column — swap so the host gets
            # it and neutrality is correct (audit HIGH: 3 real group fixtures)
            home, away = away, home
            home_score, away_score = away_score, home_score
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
            "home_score": home_score,
            "away_score": away_score,
            "status": "tbd" if tbd else ("played" if played else "upcoming"),
            "winner": winner,
            "neutral": home != host,
        })
    fx = pd.DataFrame(rows).sort_values("match_number").reset_index(drop=True)
    # redundancy: fixturedownload lagged a final score 75+ min (2026-06-12) — fill
    # past-kickoff gaps from ESPN's real-time scoreboard
    from . import livescores

    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    dates = livescores.needed_dates(fx, now)
    if dates:
        fx = livescores.merge_scores(fx, livescores.fetch_scoreboard(dates), now)
    return fx
