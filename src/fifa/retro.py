"""Retrospective predictions for matches played before the system existed.

A retro pick is an honest walk-forward prediction: the model is refit using ONLY
matches finished before the fixture's matchday, then asked to predict it — the
same protocol as the backtest. Retro rows are flagged and badged RETRO so they
are never conflated with predictions frozen before kickoff, and they never use
bookmaker odds (historical odds aren't available).
"""
from __future__ import annotations

import pandas as pd

from . import data, elo, features, ledger
from . import matrix as mx
from .dixon_coles import DixonColes
from .ensemble import Predictor
from .gbm import GoalModel


def _default_trainer(cutoff: pd.Timestamp) -> Predictor:
    from . import runtime

    played, _ = data.load_results()
    train = played[played["date"] < cutoff]
    elo_df, _ = elo.compute_elo(train)
    fb = features.FeatureBuilder()
    X, y_home, y_away = fb.fit_transform(elo_df)
    dc = DixonColes().fit(train, ref_date=cutoff)
    gbm = GoalModel().fit(X, y_home, y_away, elo_df["date"], ref_date=cutoff)
    rho, w_dc, cal = runtime.tuned_params()
    return Predictor(dc, gbm, fb, rho=rho, w_dc=w_dc, calibrator=cal)  # no odds book


def backfill(fx: pd.DataFrame, path, trainer=_default_trainer) -> int:
    """Add retro predictions for played fixtures missing from the ledger."""
    book = ledger.load(path)
    missing = fx[(fx["status"] == "played") & (~fx["match_number"].isin(book))]
    if missing.empty:
        return 0
    new = []
    for cutoff, sub in missing.groupby(missing["date"].dt.normalize(), sort=True):
        pred = trainer(cutoff)
        for r in sub.itertuples(index=False):
            m = pred.matrix_for(r.home, r.away, r.date, "FIFA World Cup",
                                r.neutral, country=r.host)
            p = mx.wdl(m)
            top5 = mx.top_scorelines(m, 5)
            new.append({
                "match_number": int(r.match_number), "home": r.home, "away": r.away,
                "kickoff": str(r.date), "predicted": list(top5[0][0]),
                "p": [round(x, 4) for x in p], "tier": mx.tier(max(p)),
                "retro": True,
            })
    return ledger.record(new, path)
