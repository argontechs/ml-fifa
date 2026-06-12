"""Daily refresh: re-download data, retrain, predict, simulate, regenerate dashboard."""
import datetime
import json
from pathlib import Path

import pandas as pd

from fifa import dashboard, data, elo, fixtures, matrix, runtime, tournament
from fifa.runtime import MemoPredictor

print("1/5 refreshing data + retraining…")
pred = MemoPredictor(runtime.build_predictor(force=True))
fx = fixtures.load_fixtures(force=True)
played, _ = data.load_results()
_, ratings = elo.compute_elo(played)

print("2/5 predicting upcoming fixtures…")
up = fx[fx["status"] == "upcoming"].sort_values("date")
horizon = pd.Timestamp.now(tz="UTC").tz_localize(None) + pd.Timedelta(days=7)
preds = []
for r in up.itertuples(index=False):
    if r.date > horizon:
        continue
    m = runtime.predict_fixture(pred, r.home, r.away, r.date, r.neutral, country=r.host)
    p = matrix.wdl(m)
    preds.append({
        "home": r.home, "away": r.away, "when": str(r.date.date()),
        "comp": f"Group {r.group}" if r.group else f"Round {r.round}",
        "p": p, "tier": matrix.tier(max(p)),
        "top5": matrix.top_scorelines(m, 5),
    })

print("3/5 simulating tournament (10,000 runs)…")
sim = tournament.simulate_tournament(fx, pred, ratings, n_runs=10000, seed=42)
(data.DATA_DIR / "sim_results.json").write_text(sim.to_json(orient="index"))

print("4/5 loading backtest card…")
report = json.loads((data.DATA_DIR / "backtest_report.json").read_text())

print("5/5 rendering dashboard…")
html = dashboard.render(
    generated_at=datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d %H:%M UTC"),
    predictions=preds,
    advance=sim["r32"].to_dict(),
    champions=sim["champion"].to_dict(),
    card=report["test_card"],
)
out = Path(__file__).resolve().parent / "dashboard" / "index.html"
out.parent.mkdir(exist_ok=True)
out.write_text(html)
print(f"done → open {out}")
