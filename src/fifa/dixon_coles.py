"""Time-decayed Poisson GLM baseline (Dixon-Coles family; tau applied at matrix stage)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import PoissonRegressor


class DixonColes:
    def __init__(self, xi: float = 0.0005, alpha: float = 1e-3, window_years: int = 15):
        self.xi = xi
        self.alpha = alpha
        self.window_years = window_years

    def fit(self, played: pd.DataFrame, ref_date: pd.Timestamp) -> "DixonColes":
        cutoff = ref_date - pd.DateOffset(years=self.window_years)
        df = played[(played["date"] <= ref_date) & (played["date"] >= cutoff)]
        teams = sorted(set(df["home_team"]) | set(df["away_team"]))
        self.idx_ = {t: i for i, t in enumerate(teams)}
        T = len(teams)
        n = len(df)

        h = df["home_team"].map(self.idx_).to_numpy()
        a = df["away_team"].map(self.idx_).to_numpy()
        not_neutral = (~df["neutral"]).to_numpy().astype(float)
        days = (ref_date - df["date"]).dt.days.to_numpy()
        w = np.exp(-self.xi * days)

        # Two observations per match: row 2i = home goals (attack=home, defense=away,
        # home_adv active when not neutral), row 2i+1 = away goals (attack=away,
        # defense=home). Columns: [home_adv | attack_0..T-1 | defense_0..T-1].
        rows = np.repeat(np.arange(2 * n), 3)
        cols = np.empty(6 * n, dtype=int)
        vals = np.empty(6 * n)
        # home-goal observations (entries 6i..6i+2 → row 2i)
        cols[0::6], vals[0::6] = 0, not_neutral
        cols[1::6], vals[1::6] = 1 + h, 1.0
        cols[2::6], vals[2::6] = 1 + T + a, 1.0
        # away-goal observations (entries 6i+3..6i+5 → row 2i+1)
        cols[3::6], vals[3::6] = 0, 0.0
        cols[4::6], vals[4::6] = 1 + a, 1.0
        cols[5::6], vals[5::6] = 1 + T + h, 1.0
        X = sparse.csr_matrix((vals, (rows, cols)), shape=(2 * n, 1 + 2 * T))
        y = np.empty(2 * n)
        y[0::2] = df["home_score"].to_numpy()
        y[1::2] = df["away_score"].to_numpy()

        self.model_ = PoissonRegressor(alpha=self.alpha, max_iter=300)
        self.model_.fit(X, y, sample_weight=np.repeat(w, 2))
        self.home_adv_ = float(self.model_.coef_[0])
        self._T = T
        return self

    def _linear(self, attack_team: str | None, defense_team: str | None, home_adv: bool) -> float:
        z = self.model_.intercept_ + (self.home_adv_ if home_adv else 0.0)
        if attack_team in self.idx_:
            z += self.model_.coef_[1 + self.idx_[attack_team]]
        if defense_team in self.idx_:
            z += self.model_.coef_[1 + self._T + self.idx_[defense_team]]
        return z

    def predict_lambdas(self, home: str, away: str, neutral: bool) -> tuple[float, float]:
        lh = float(np.exp(self._linear(home, away, home_adv=not neutral)))
        la = float(np.exp(self._linear(away, home, home_adv=False)))
        return float(np.clip(lh, 0.05, 8.0)), float(np.clip(la, 0.05, 8.0))
