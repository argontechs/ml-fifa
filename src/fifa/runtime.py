"""Build the production Predictor on all data; shared by predict/simulate/update."""
from __future__ import annotations

import json

import pandas as pd

from . import data, elo, features, matrix
from .dixon_coles import DixonColes
from .ensemble import Predictor
from .gbm import GoalModel

DEFAULT_RHO, DEFAULT_W = -0.05, 0.5


def tuned_params() -> tuple[float, float]:
    path = data.DATA_DIR / "backtest_report.json"
    if path.exists():
        rep = json.loads(path.read_text())
        return rep["rho"], rep["w_dc"]
    print("WARNING: no backtest_report.json — using default rho/w (run backtest.py)")
    return DEFAULT_RHO, DEFAULT_W


def build_predictor(force: bool = False) -> Predictor:
    """Fit DC + GBM on ALL played matches through yesterday."""
    played, _ = data.load_results(force=force)
    elo_df, _ = elo.compute_elo(played)
    fb = features.FeatureBuilder()
    X, y_home, y_away = fb.fit_transform(elo_df)
    today = played["date"].max()
    dc = DixonColes().fit(played, ref_date=today)
    gbm = GoalModel().fit(X, y_home, y_away, elo_df["date"], ref_date=today)
    rho, w_dc = tuned_params()
    return Predictor(dc, gbm, fb, rho=rho, w_dc=w_dc)


def predict_fixture(pred: Predictor, home: str, away: str, date, neutral: bool):
    return pred.matrix_for(home, away, pd.Timestamp(date), "FIFA World Cup", neutral)


def format_prediction(home, away, when, comp, p, top5) -> str:
    badge = matrix.tier(max(p))
    tops = ", ".join(f"{i}-{j} ({pr:.1%})" for (i, j), pr in top5)
    return (
        f"{home} vs {away} — {when} — {comp}\n"
        f"  W {p[0]:.1%} | D {p[1]:.1%} | L {p[2]:.1%}   [{badge}]\n"
        f"  Top scorelines: {tops}"
    )
