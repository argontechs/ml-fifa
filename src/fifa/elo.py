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


def compute_elo(
    played: pd.DataFrame, initial: float = INITIAL
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Chronological pass. Returns (copy with elo_{home,away}_{pre,post} columns, final ratings)."""
    ratings: dict[str, float] = {}
    pre_h, pre_a, post_h, post_a = [], [], [], []
    for row in played.itertuples(index=False):
        rh = ratings.get(row.home_team, initial)
        ra = ratings.get(row.away_team, initial)
        we = expected(rh, ra, row.neutral)
        delta = (
            tournament_k(row.tournament)
            * goal_multiplier(row.home_score - row.away_score)
            * (result_value(row.home_score, row.away_score) - we)
        )
        ratings[row.home_team] = rh + delta
        ratings[row.away_team] = ra - delta
        pre_h.append(rh)
        pre_a.append(ra)
        post_h.append(rh + delta)
        post_a.append(ra - delta)
    out = played.copy()
    out["elo_home_pre"] = pre_h
    out["elo_away_pre"] = pre_a
    out["elo_home_post"] = post_h
    out["elo_away_post"] = post_a
    return out, ratings
