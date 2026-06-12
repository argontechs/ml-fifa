"""Monte Carlo the remaining tournament.

usage: .venv/bin/python simulate.py [--runs 10000] [--seed 42]
"""
import argparse

from fifa import data, elo, fixtures, runtime, tournament


class MemoPredictor:
    """Same-pairing matrices are deterministic — cache across simulation runs."""

    def __init__(self, pred):
        self.pred, self.cache = pred, {}

    def matrix_for(self, home, away, date, tournament_name, neutral):
        key = (home, away, neutral, str(getattr(date, "date", lambda: date)()))
        if key not in self.cache:
            self.cache[key] = self.pred.matrix_for(home, away, date, tournament_name, neutral)
        return self.cache[key]


ap = argparse.ArgumentParser()
ap.add_argument("--runs", type=int, default=10000)
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()

pred = MemoPredictor(runtime.build_predictor())
played, _ = data.load_results()
_, ratings = elo.compute_elo(played)
fx = fixtures.load_fixtures()

result = tournament.simulate_tournament(fx, pred, ratings, n_runs=args.runs, seed=args.seed)
print(f"\n=== {args.runs:,} tournament simulations ===")
print((result.head(15) * 100).round(1).to_string())
(data.DATA_DIR / "sim_results.json").write_text(result.to_json(orient="index"))
print(f"\n({len(pred.cache)} distinct matchups computed)  → data/sim_results.json")
