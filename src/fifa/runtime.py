"""Build the production Predictor on all data; shared by predict/simulate/update."""
from __future__ import annotations

import json

import pandas as pd

from . import data, elo, features, matrix
from .dixon_coles import DixonColes
from .ensemble import Predictor
from .gbm import GoalModel

DEFAULT_RHO, DEFAULT_W = -0.05, 0.5


def tuned_params(prefer_production: bool = True) -> tuple[float, float, "WDLCalibrator | None"]:
    from .calibrate import WDLCalibrator

    path = data.DATA_DIR / "backtest_report.json"
    if path.exists():
        rep = json.loads(path.read_text())
        # prefer the rolling production calibrator (fit on all out-of-sample predictions
        # through the last backtest run); fall back to the val-only one
        if prefer_production:
            cal_dict = rep.get("calibrator_production") or rep.get("calibrator")
        else:  # walk-forward consumers (retro): val-only window predates all targets
            cal_dict = rep.get("calibrator")
        cal = WDLCalibrator.from_dict(cal_dict) if cal_dict else None
        return rep["rho"], rep["w_dc"], cal
    print("WARNING: no backtest_report.json — using default rho/w (run backtest.py)")
    return DEFAULT_RHO, DEFAULT_W, None


def build_predictor(force: bool = False) -> Predictor:
    """Fit DC + GBM on ALL played matches through yesterday."""
    played, _ = data.load_results(force=force)
    elo_df, _ = elo.compute_elo(played)
    fb = features.FeatureBuilder()
    X, y_home, y_away = fb.fit_transform(elo_df)
    today = played["date"].max()
    dc = DixonColes().fit(played, ref_date=today)
    gbm = GoalModel().fit(X, y_home, y_away, elo_df["date"], ref_date=today)
    rho, w_dc, cal = tuned_params()
    from . import odds

    book = odds.fetch_book()  # 6h cache governs; force here burned ~96% of monthly quota (audit)
    if book:
        print(f"bookmaker odds loaded for {len(book)} fixtures (market-blended)")
    return Predictor(dc, gbm, fb, rho=rho, w_dc=w_dc, calibrator=cal, book=book)


def predict_fixture(pred: Predictor, home: str, away: str, date, neutral: bool,
                    country: str | None = None):
    return pred.matrix_for(home, away, pd.Timestamp(date), "FIFA World Cup", neutral,
                           country=country)


class MemoPredictor:
    """Same-pairing matrices are deterministic — cache across Monte Carlo runs."""

    def __init__(self, pred):
        self.pred, self.cache = pred, {}

    def matrix_for(self, home, away, date, tournament_name, neutral, country=None):
        key = (home, away, neutral, country, str(getattr(date, "date", lambda: date)()))
        if key not in self.cache:
            self.cache[key] = self.pred.matrix_for(home, away, date, tournament_name,
                                                   neutral, country=country)
        return self.cache[key]


def format_prediction(home, away, when, comp, p, top5) -> str:
    badge = matrix.tier(max(p))
    tops = ", ".join(f"{i}-{j} ({pr:.1%})" for (i, j), pr in top5)
    return (
        f"{home} vs {away} — {when} — {comp}\n"
        f"  W {p[0]:.1%} | D {p[1]:.1%} | L {p[2]:.1%}   [{badge}]\n"
        f"  Top scorelines: {tops}"
    )
