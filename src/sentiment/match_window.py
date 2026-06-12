"""Which matches are live right now, and what keywords identify them in posts."""
from __future__ import annotations

import pandas as pd

from fifa import data as fdata

ACTIVE_BEFORE = pd.Timedelta(minutes=30)
ACTIVE_AFTER = pd.Timedelta(minutes=150)

# FIFA trigraph codes for all 48 WC2026 teams — verified against
# en.wikipedia.org/wiki/List_of_FIFA_country_codes on 2026-06-12.
TRIGRAPH = {
    "Mexico": "MEX", "South Africa": "RSA", "South Korea": "KOR", "Czech Republic": "CZE",
    "Canada": "CAN", "Bosnia and Herzegovina": "BIH", "Qatar": "QAT", "Switzerland": "SUI",
    "Brazil": "BRA", "Morocco": "MAR", "Haiti": "HAI", "Scotland": "SCO",
    "United States": "USA", "Paraguay": "PAR", "Australia": "AUS", "Turkey": "TUR",
    "Germany": "GER", "Curaçao": "CUW", "Ivory Coast": "CIV", "Ecuador": "ECU",
    "Netherlands": "NED", "Japan": "JPN", "Sweden": "SWE", "Tunisia": "TUN",
    "Belgium": "BEL", "Egypt": "EGY", "Iran": "IRN", "New Zealand": "NZL",
    "Spain": "ESP", "Cape Verde": "CPV", "Saudi Arabia": "KSA", "Uruguay": "URU",
    "France": "FRA", "Senegal": "SEN", "Iraq": "IRQ", "Norway": "NOR",
    "Argentina": "ARG", "Algeria": "ALG", "Austria": "AUT", "Jordan": "JOR",
    "Portugal": "POR", "DR Congo": "COD", "Uzbekistan": "UZB", "Colombia": "COL",
    "England": "ENG", "Croatia": "CRO", "Ghana": "GHA", "Panama": "PAN",
}

_REVERSE_ALIASES: dict[str, set[str]] = {}
for _fifa_name, _ds_name in fdata.FIFA_ALIASES.items():
    _REVERSE_ALIASES.setdefault(_ds_name, set()).add(_fifa_name)


def active_matches(fx: pd.DataFrame, now: pd.Timestamp) -> list[dict]:
    """Matches inside the live window: kickoff − 30 min → kickoff + 150 min."""
    out = []
    for r in fx.itertuples(index=False):
        if r.status == "tbd":
            continue
        if r.date - ACTIVE_BEFORE <= now <= r.date + ACTIVE_AFTER:
            out.append({"key": int(r.match_number), "home": r.home, "away": r.away,
                        "kickoff": r.date})
    return out


def _team_terms(team: str) -> set[str]:
    return {team.lower()} | {a.lower() for a in _REVERSE_ALIASES.get(team, set())}


def keywords_for(match: dict) -> dict[str, set[str]]:
    th, ta = TRIGRAPH[match["home"]], TRIGRAPH[match["away"]]
    return {
        "home": _team_terms(match["home"]),
        "away": _team_terms(match["away"]),
        "both": {f"#{th}{ta}".lower(), f"#{ta}{th}".lower()},
    }
