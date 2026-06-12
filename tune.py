"""Hyperparameter sweep on validation RPS (train ≤2021, validate 2022-23).

Honest protocol: one-factor-at-a-time around current defaults; any winner is only
adopted if it ALSO improves the full backtest and passes every honesty gate there.
Results → data/tuning_results.json.
"""
import json
import time

import numpy as np

import fifa.gbm as gbm_mod
from fifa import backtest_lib as bl
from fifa import data, elo, features
from fifa.dixon_coles import DixonColes
from fifa.gbm import GoalModel

played, _ = data.load_results()
elo_df, _ = elo.compute_elo(played)
fb = features.FeatureBuilder()
X, yh, ya = fb.fit_transform(elo_df)
dates = elo_df["date"]
is_train = (dates <= bl.TRAIN_END).to_numpy()
is_val = ((dates > bl.TRAIN_END) & (dates <= bl.VAL_END)).to_numpy()
val_df = played[is_val]
print(f"train n={is_train.sum()}  val n={is_val.sum()}")

BASE = {"params": {}, "time_decay": 0.0005, "friendly_w": 0.5, "xi": 0.001}
VARIANTS = [
    ("baseline", {}),
    ("leaves_31", {"params": {"num_leaves": 31}}),
    ("leaves_127", {"params": {"num_leaves": 127}}),
    ("lr03_800", {"params": {"learning_rate": 0.03, "n_estimators": 800}}),
    ("lr06_400", {"params": {"learning_rate": 0.06, "n_estimators": 400}}),
    ("mcs_20", {"params": {"min_child_samples": 20}}),
    ("mcs_80", {"params": {"min_child_samples": 80}}),
    ("decay_3e-4", {"time_decay": 0.0003}),
    ("decay_1e-3", {"time_decay": 0.001}),
    ("friendly_03", {"friendly_w": 0.3}),
    ("friendly_07", {"friendly_w": 0.7}),
    ("xi_5e-4", {"xi": 0.0005}),
    ("xi_2e-3", {"xi": 0.002}),
]

results = []
for name, overrides in VARIANTS:
    cfg = {**BASE, **{k: v for k, v in overrides.items() if k != "params"}}
    params = {**BASE["params"], **overrides.get("params", {})}
    t0 = time.time()
    gbm_mod.TIME_DECAY = cfg["time_decay"]
    gbm_mod.FRIENDLY_WEIGHT = cfg["friendly_w"]
    dc = DixonColes(xi=cfg["xi"]).fit(played[is_train], ref_date=bl.TRAIN_END)
    gm = GoalModel(**params).fit(X[is_train], yh[is_train], ya[is_train],
                                 dates[is_train], ref_date=bl.TRAIN_END)
    dc_ls, gbm_ls = bl._lambda_pairs(val_df, dc, gm, X[is_val])
    best_rps, best_w = min(
        ((sum(r["rps"] for r in bl._eval_rows(val_df, dc_ls, gbm_ls, 0.0, w)) / len(val_df), w)
         for w in np.arange(0.0, 1.001, 0.1).round(2))
    )
    results.append({"name": name, "rps": round(best_rps, 5), "w_dc": best_w,
                    "params": params, **{k: cfg[k] for k in ("time_decay", "friendly_w", "xi")},
                    "secs": round(time.time() - t0, 1)})
    print(f"{name:14s} val RPS {best_rps:.5f}  (w_dc={best_w})  [{results[-1]['secs']}s]")

results.sort(key=lambda r: r["rps"])
(data.DATA_DIR / "tuning_results.json").write_text(json.dumps(results, indent=2))
print("\n=== sorted (lower RPS = better) ===")
for r in results:
    print(f"{r['name']:14s} {r['rps']:.5f}")
print("\nNOTE: adopt a winner only if the full backtest gates still PASS with it.")
