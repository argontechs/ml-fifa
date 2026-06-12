"""Strictly pre-match feature construction via single chronological pass.

Leakage-safe BY CONSTRUCTION: for every match we snapshot team state first,
then update state with the match result. The unit test enforces this.
"""
from __future__ import annotations

import math
from collections import deque

import pandas as pd

from . import elo

DECAY = 0.001          # form decay per day (half-life ≈ 1.9 years)
REST_CAP = 60.0        # days
PRIOR_GOALS = 1.4      # international per-team scoring prior
PRIOR_WEIGHT = 5.0

CONFED = {
    # UEFA
    "England": "UEFA", "France": "UEFA", "Spain": "UEFA", "Portugal": "UEFA",
    "Germany": "UEFA", "Netherlands": "UEFA", "Belgium": "UEFA", "Croatia": "UEFA",
    "Italy": "UEFA", "Switzerland": "UEFA", "Austria": "UEFA", "Norway": "UEFA",
    "Sweden": "UEFA", "Scotland": "UEFA", "Turkey": "UEFA", "Czech Republic": "UEFA",
    "Bosnia and Herzegovina": "UEFA", "Serbia": "UEFA", "Denmark": "UEFA",
    "Poland": "UEFA", "Ukraine": "UEFA", "Wales": "UEFA", "Russia": "UEFA",
    "Hungary": "UEFA", "Greece": "UEFA", "Romania": "UEFA", "Slovakia": "UEFA",
    "Slovenia": "UEFA", "Ireland": "UEFA", "Northern Ireland": "UEFA", "Iceland": "UEFA",
    # CONMEBOL
    "Argentina": "CONMEBOL", "Brazil": "CONMEBOL", "Uruguay": "CONMEBOL",
    "Colombia": "CONMEBOL", "Paraguay": "CONMEBOL", "Ecuador": "CONMEBOL",
    "Chile": "CONMEBOL", "Peru": "CONMEBOL", "Bolivia": "CONMEBOL", "Venezuela": "CONMEBOL",
    # CONCACAF
    "Mexico": "CONCACAF", "United States": "CONCACAF", "Canada": "CONCACAF",
    "Panama": "CONCACAF", "Haiti": "CONCACAF", "Curaçao": "CONCACAF",
    "Costa Rica": "CONCACAF", "Jamaica": "CONCACAF", "Honduras": "CONCACAF",
    # CAF
    "Morocco": "CAF", "Senegal": "CAF", "Egypt": "CAF", "Algeria": "CAF",
    "Tunisia": "CAF", "Ghana": "CAF", "Ivory Coast": "CAF", "Nigeria": "CAF",
    "Cameroon": "CAF", "South Africa": "CAF", "DR Congo": "CAF", "Cape Verde": "CAF",
    # AFC
    "Japan": "AFC", "South Korea": "AFC", "Iran": "AFC", "Saudi Arabia": "AFC",
    "Australia": "AFC", "Qatar": "AFC", "Iraq": "AFC", "Jordan": "AFC",
    "Uzbekistan": "AFC", "China": "AFC", "North Korea": "AFC",
    # OFC
    "New Zealand": "OFC",
}
CONFED_LEVELS = ["UEFA", "CONMEBOL", "CONCACAF", "CAF", "AFC", "OFC", "OTHER"]


class _TeamState:
    __slots__ = ("gf", "ga", "w", "last_date", "recent", "played")

    def __init__(self):
        self.gf = 0.0
        self.ga = 0.0
        self.w = 0.0
        self.last_date = None
        self.recent = deque(maxlen=10)
        self.played = 0

    def _decay_factor(self, date):
        if self.last_date is None:
            return 1.0
        return math.exp(-DECAY * (date - self.last_date).days)

    def snapshot(self, date):
        f = self._decay_factor(date)
        w = self.w * f + PRIOR_WEIGHT
        rest = REST_CAP if self.last_date is None else min((date - self.last_date).days, REST_CAP)
        return {
            "gf_rate": (self.gf * f + PRIOR_GOALS * PRIOR_WEIGHT) / w,
            "ga_rate": (self.ga * f + PRIOR_GOALS * PRIOR_WEIGHT) / w,
            "winrate": sum(self.recent) / len(self.recent) if self.recent else 0.5,
            "rest": float(rest),
            "played": self.played,
        }

    def update(self, date, goals_for, goals_against, result):
        f = self._decay_factor(date)
        self.gf = self.gf * f + goals_for
        self.ga = self.ga * f + goals_against
        self.w = self.w * f + 1.0
        self.recent.append(result)
        self.last_date = date
        self.played += 1


class FeatureBuilder:
    """fit_transform() replays played matches; features_for() queries the final state."""

    COLUMNS = [
        "elo_home", "elo_away", "elo_diff",
        "gf_home", "ga_home", "gf_away", "ga_away",
        "winrate_home", "winrate_away", "rest_home", "rest_away",
        "played_home", "played_away", "h2h_gd",
        "elo_trend1y_home", "elo_trend1y_away", "elo_trend2y_home", "elo_trend2y_away",
        "wc_exp_home", "wc_exp_away",
        "k_tier", "neutral", "confed_home", "confed_away",
    ]

    def __init__(self):
        self._teams: dict[str, _TeamState] = {}
        self._h2h: dict[tuple[str, str], float] = {}
        self._ratings: dict[str, float] = {}
        self._elo_hist: dict[str, list[tuple[pd.Timestamp, float]]] = {}
        self._wc_count: dict[str, int] = {}

    def _team(self, name):
        if name not in self._teams:
            self._teams[name] = _TeamState()
        return self._teams[name]

    def _trend(self, team, date, elo_now, days):
        """Elo momentum: rating now minus rating `days` ago (0 if no history that old)."""
        hist = self._elo_hist.get(team)
        if not hist:
            return 0.0
        cutoff = date - pd.Timedelta(days=days)
        for d, r in reversed(hist):
            if d <= cutoff:
                return elo_now - r
        return 0.0

    def _row(self, home, away, date, tournament, neutral, elo_home, elo_away):
        sh = self._team(home).snapshot(date)
        sa = self._team(away).snapshot(date)
        key = (min(home, away), max(home, away))
        h2h = self._h2h.get(key, 0.0) * (1 if home <= away else -1)
        return {
            "elo_home": elo_home,
            "elo_away": elo_away,
            "elo_diff": elo_home - elo_away + (0.0 if neutral else elo.HOME_ADV),
            "gf_home": sh["gf_rate"], "ga_home": sh["ga_rate"],
            "gf_away": sa["gf_rate"], "ga_away": sa["ga_rate"],
            "winrate_home": sh["winrate"], "winrate_away": sa["winrate"],
            "rest_home": sh["rest"], "rest_away": sa["rest"],
            "played_home": sh["played"], "played_away": sa["played"],
            "h2h_gd": h2h,
            "elo_trend1y_home": self._trend(home, date, elo_home, 365),
            "elo_trend1y_away": self._trend(away, date, elo_away, 365),
            "elo_trend2y_home": self._trend(home, date, elo_home, 730),
            "elo_trend2y_away": self._trend(away, date, elo_away, 730),
            "wc_exp_home": self._wc_count.get(home, 0),
            "wc_exp_away": self._wc_count.get(away, 0),
            "k_tier": elo.tournament_k(tournament),
            "neutral": int(neutral),
            "confed_home": CONFED.get(home, "OTHER"),
            "confed_away": CONFED.get(away, "OTHER"),
        }

    def fit_transform(self, elo_df: pd.DataFrame):
        """elo_df: output of elo.compute_elo() — chronological, with elo_*_pre columns."""
        rows = []
        for r in elo_df.itertuples(index=False):
            rows.append(
                self._row(r.home_team, r.away_team, r.date, r.tournament,
                          r.neutral, r.elo_home_pre, r.elo_away_pre)
            )
            res = elo.result_value(r.home_score, r.away_score)
            self._team(r.home_team).update(r.date, r.home_score, r.away_score, res)
            self._team(r.away_team).update(r.date, r.away_score, r.home_score, 1.0 - res)
            key = (min(r.home_team, r.away_team), max(r.home_team, r.away_team))
            gd = (r.home_score - r.away_score) * (1 if r.home_team <= r.away_team else -1)
            self._h2h[key] = self._h2h.get(key, 0.0) * 0.9 + gd
            self._ratings[r.home_team] = r.elo_home_post
            self._ratings[r.away_team] = r.elo_away_post
            self._elo_hist.setdefault(r.home_team, []).append((r.date, r.elo_home_post))
            self._elo_hist.setdefault(r.away_team, []).append((r.date, r.elo_away_post))
            if r.tournament == "FIFA World Cup":
                self._wc_count[r.home_team] = self._wc_count.get(r.home_team, 0) + 1
                self._wc_count[r.away_team] = self._wc_count.get(r.away_team, 0) + 1
        X = pd.DataFrame(rows, columns=self.COLUMNS)
        X = self._finalize(X)
        return X, elo_df["home_score"].to_numpy(), elo_df["away_score"].to_numpy()

    def features_for(self, home, away, date, tournament, neutral) -> pd.DataFrame:
        eh = self._ratings.get(home, elo.INITIAL)
        ea = self._ratings.get(away, elo.INITIAL)
        X = pd.DataFrame([self._row(home, away, date, tournament, neutral, eh, ea)],
                         columns=self.COLUMNS)
        return self._finalize(X)

    @staticmethod
    def _finalize(X: pd.DataFrame) -> pd.DataFrame:
        for col in ("confed_home", "confed_away"):
            X[col] = pd.Categorical(X[col], categories=CONFED_LEVELS)
        return X
