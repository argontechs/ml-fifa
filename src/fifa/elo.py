"""World Football Elo Ratings engine (eloratings.net formulas)."""
from __future__ import annotations

import pandas as pd

HOME_ADV = 100.0
INITIAL = 1500.0

_CONTINENTAL_50 = {
    "UEFA Euro", "Copa América", "African Cup of Nations", "Africa Cup of Nations",
    "AFC Asian Cup", "Gold Cup", "CONCACAF Championship", "Confederations Cup",
    "FIFA Confederations Cup", "Oceania Nations Cup", "OFC Nations Cup", "Finalissima",
}


def tournament_k(tournament: str) -> float:
    if tournament == "FIFA World Cup":
        return 60.0
    low = tournament.lower()
    if "qualification" in low or "nations league" in low:
        return 40.0
    if tournament in _CONTINENTAL_50:
        return 50.0
    if tournament == "Friendly":
        return 20.0
    return 30.0


def expected(r_home: float, r_away: float, neutral: bool) -> float:
    dr = r_home - r_away + (0.0 if neutral else HOME_ADV)
    return 1.0 / (1.0 + 10.0 ** (-dr / 400.0))


def goal_multiplier(margin: int) -> float:
    m = abs(margin)
    if m <= 1:
        return 1.0
    if m == 2:
        return 1.5
    if m == 3:
        return 1.75
    return 1.75 + (m - 3) / 8.0


def result_value(home_score: int, away_score: int) -> float:
    if home_score > away_score:
        return 1.0
    if home_score < away_score:
        return 0.0
    return 0.5  # includes shootout matches — dataset scores exclude shootouts by design
