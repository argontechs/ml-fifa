"""Daily refresh: re-download data, retrain, freeze predictions, simulate, regenerate dashboard."""
import datetime
import json
from pathlib import Path

import pandas as pd

from fifa import dashboard, data, elo, fixtures, ledger, matrix, retro, runtime, tournament
from fifa.runtime import MemoPredictor

LEDGER_PATH = data.DATA_DIR / "predictions_ledger.jsonl"
MYT_OFFSET = pd.Timedelta(hours=8)  # display timezone: Malaysia (UTC+8), no DST


def myt(ts) -> str:
    """Format a naive-UTC timestamp for display in Malaysia time."""
    return (pd.Timestamp(ts) + MYT_OFFSET).strftime("%b %d · %H:%M MYT")

print("1/6 refreshing data + retraining…")
base = runtime.build_predictor(force=True)
pred = MemoPredictor(base)
fx = fixtures.load_fixtures(force=True)
played, _ = data.load_results()
_, ratings = elo.compute_elo(played)

print("2/6 predicting + freezing all confirmed upcoming fixtures…")
up = fx[fx["status"] == "upcoming"].sort_values("date")
all_preds, display = [], []
horizon = pd.Timestamp.now(tz="UTC").tz_localize(None) + pd.Timedelta(days=7)
for r in up.itertuples(index=False):
    m = runtime.predict_fixture(pred, r.home, r.away, r.date, r.neutral, country=r.host)
    p = matrix.wdl(m)
    top5 = matrix.top_scorelines(m, 5)
    rec = {
        "match_number": int(r.match_number), "home": r.home, "away": r.away,
        "kickoff": str(r.date), "predicted": list(top5[0][0]),
        "p": [round(x, 4) for x in p], "tier": matrix.tier(max(p)),
        "frozen_at": str(pd.Timestamp.now(tz="UTC")),
    }
    all_preds.append(rec)
    if r.date <= horizon:
        display.append({
            "home": r.home, "away": r.away,
            "when": myt(r.date),
            "comp": f"Group {r.group}" if r.group else f"Round {r.round}",
            "p": p, "tier": rec["tier"], "score": top5[0][0], "top5": top5,
            "ctx_home": base.fb.team_context(r.home, r.date),
            "ctx_away": base.fb.team_context(r.away, r.date),
            "market": (r.home, r.away) in base.book,
        })
n_frozen = ledger.record(all_preds, LEDGER_PATH)
print(f"   {n_frozen} new predictions frozen into the ledger")
n_retro = retro.backfill(fx, LEDGER_PATH)
if n_retro:
    print(f"   {n_retro} RETRO predictions backfilled (walk-forward, pre-matchday models)")

print("3/6 simulating tournament (10,000 runs)…")
pens = tournament.shootout_table(data.load_shootouts())
sim = tournament.simulate_tournament(fx, pred, ratings, n_runs=10000, seed=42,
                                     pens_tbl=pens)
(data.DATA_DIR / "sim_results.json").write_text(sim.to_json(orient="index"))

print("4/6 scoring the track record…")
tracker_rows, tally = ledger.tracker(ledger.load(LEDGER_PATH), fx)
for row in tracker_rows:
    if row.get("kickoff"):
        row["when"] = myt(row["kickoff"])

print("5/6 loading backtest card…")
report = json.loads((data.DATA_DIR / "backtest_report.json").read_text())

print("6/6 rendering dashboard…")
view = {
    "generated_at": (datetime.datetime.now(datetime.UTC)
                     + datetime.timedelta(hours=8)).strftime("%Y-%m-%d %H:%M MYT"),
    "card": report["test_card"],
    "matches": display,
    "tracker_rows": tracker_rows,
    "tracker_tally": tally,
    "standings": tournament.current_tables(fx),
    "advance": sim["r32"].to_dict(),
    "champions": sim["champion"].to_dict(),
}
out_dir = Path(__file__).resolve().parent / "dashboard"
out_dir.mkdir(exist_ok=True)
(out_dir / "index.html").write_text(dashboard.render(view))
(out_dir / "past.html").write_text(dashboard.render_past(view))
try:  # sentiment snapshot page — best-effort, never blocks the predictor refresh
    from sentiment import db as sdb
    from sentiment import publish

    sconn = sdb.connect(data.DATA_DIR / "sentiment.db")
    (out_dir / "sentiment.html").write_text(publish.render(sconn, view["generated_at"]))
    sconn.close()
except Exception as exc:  # noqa: BLE001
    print(f"sentiment page skipped ({exc})")
print(f"rendered → {out_dir}/index.html + past.html + sentiment.html")

import os
import subprocess

if os.environ.get("DASH_DEPLOY", "1") != "0":
    print("7/7 deploying to Cloudflare Pages…")
    try:
        env = dict(os.environ)
        token_file = data.DATA_DIR / "cf_token.txt"
        if token_file.exists():
            env["CLOUDFLARE_API_TOKEN"] = token_file.read_text().strip()
            env["CLOUDFLARE_ACCOUNT_ID"] = "6c51b7a0e7b4980a0d5897c365ddc36e"
        subprocess.run(
            ["npx", "--yes", "wrangler", "pages", "deploy", str(out_dir),
             "--project-name", "wc2026", "--branch", "main", "--commit-dirty=true"],
            check=True, timeout=300, cwd=Path(__file__).resolve().parent, env=env,
        )
    except Exception as exc:  # noqa: BLE001 — deploys must never block the local refresh
        print(f"WARNING: deploy skipped ({exc})")
