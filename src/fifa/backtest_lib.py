"""Walk-forward backtest pipeline shared by backtest.py and update.py."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import data, elo, evaluate, features
from . import matrix as mx
from .calibrate import WDLCalibrator
from .dixon_coles import DixonColes
from .ensemble import Predictor
from .gbm import GoalModel

TRAIN_END = pd.Timestamp("2021-12-31")
VAL_END = pd.Timestamp("2023-12-31")
RHOS = [round(r, 3) for r in np.arange(-0.20, 0.001, 0.025)]
WS = [round(w, 2) for w in np.arange(0.0, 1.001, 0.1)]


def grid_pick(eval_fn, rhos=RHOS, ws=WS) -> tuple[float, float]:
    best, best_loss = None, float("inf")
    for rho in rhos:
        for w in ws:
            loss = eval_fn(rho, w)
            if loss < best_loss:
                best, best_loss = (rho, w), loss
    return best


def _lambda_pairs(eval_df, dc, gbm, X_eval):
    """Precompute (dc_lambdas, gbm_lambdas) for every eval row — rho/w independent."""
    lh_g, la_g = gbm.predict_lambdas(X_eval)
    dc_ls = [
        dc.predict_lambdas(r.home_team, r.away_team, r.neutral)
        for r in eval_df.itertuples(index=False)
    ]
    gbm_ls = list(zip(lh_g.tolist(), la_g.tolist()))
    return dc_ls, gbm_ls


def _eval_rows(eval_df, dc_ls, gbm_ls, rho, w_dc, calibrator=None):
    """Score every row with a given (rho, w). Returns list of row dicts."""
    pred = Predictor(None, None, None, rho=rho, w_dc=w_dc)
    rows = []
    for i, r in enumerate(eval_df.itertuples(index=False)):
        m = pred.matrix_from_lambdas(dc_ls[i], gbm_ls[i])
        if calibrator is not None:
            m = mx.rescale_wdl(m, tuple(calibrator.transform(mx.wdl(m))[0]))
        rows.append(evaluate.score_prediction(m, r.home_score, r.away_score))
    return rows


def run(report_path=None) -> dict:
    played, _ = data.load_results()
    elo_df, _ = elo.compute_elo(played)
    fb = features.FeatureBuilder()
    X, y_home, y_away = fb.fit_transform(elo_df)
    dates = elo_df["date"]

    is_train = dates <= TRAIN_END
    is_val = (dates > TRAIN_END) & (dates <= VAL_END)
    is_test = dates > VAL_END

    # Stage 1: fit on train, tune (rho, w) on val
    print(f"fitting on train (n={int(is_train.sum())}) …")
    dc = DixonColes().fit(played[is_train], ref_date=TRAIN_END)
    gbm = GoalModel().fit(X[is_train], y_home[is_train.to_numpy()],
                          y_away[is_train.to_numpy()], dates[is_train], ref_date=TRAIN_END)
    val_df = played[is_val]
    print(f"tuning rho/w on validation (n={len(val_df)}) …")
    dc_ls, gbm_ls = _lambda_pairs(val_df, dc, gbm, X[is_val])

    def val_loss(rho, w):
        rows = _eval_rows(val_df, dc_ls, gbm_ls, rho, w)
        return sum(r["rps"] for r in rows) / len(rows)

    rho, w_dc = grid_pick(val_loss)
    print(f"tuned on validation: rho={rho}, w_dc={w_dc}")

    # Fit the isotonic calibrator on validation forecasts (out-of-sample for train fit)
    val_rows = _eval_rows(val_df, dc_ls, gbm_ls, rho, w_dc)
    cal = WDLCalibrator().fit([r["p"] for r in val_rows], [r["outcome"] for r in val_rows])

    # Stage 2: refit on train+val, report on test
    fit_mask = is_train | is_val
    print(f"refitting on train+val (n={int(fit_mask.sum())}) …")
    dc2 = DixonColes().fit(played[fit_mask], ref_date=VAL_END)
    gbm2 = GoalModel().fit(X[fit_mask], y_home[fit_mask.to_numpy()],
                           y_away[fit_mask.to_numpy()], dates[fit_mask], ref_date=VAL_END)
    test_df = played[is_test]
    print(f"evaluating on test (n={len(test_df)}) …")
    dc_ls2, gbm_ls2 = _lambda_pairs(test_df, dc2, gbm2, X[is_test])
    raw_card = evaluate.report_card(_eval_rows(test_df, dc_ls2, gbm_ls2, rho, w_dc))
    test_rows = _eval_rows(test_df, dc_ls2, gbm_ls2, rho, w_dc, calibrator=cal)
    card = evaluate.report_card(test_rows)
    result = {
        "rho": rho,
        "w_dc": w_dc,
        "test_card": card,          # calibrated — the official card
        "raw_card": raw_card,       # uncalibrated, for comparison
        "calibrator": cal.to_dict(),
        "test_span": [str(test_df["date"].min().date()), str(test_df["date"].max().date())],
    }
    if report_path:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(result, indent=2, default=float))
    return result
