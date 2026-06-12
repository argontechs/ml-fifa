"""Predict WC2026 fixtures.

usage:
  .venv/bin/python predict.py "France vs Senegal"   # one fixture (must be in the schedule)
  .venv/bin/python predict.py --days 7              # everything in the next N days
  .venv/bin/python predict.py --all                 # all remaining confirmed fixtures
"""
import argparse

import pandas as pd

from fifa import fixtures, matrix, runtime

ap = argparse.ArgumentParser()
ap.add_argument("fixture", nargs="?", help='e.g. "France vs Senegal"')
ap.add_argument("--days", type=int, default=None)
ap.add_argument("--all", action="store_true")
args = ap.parse_args()

fx = fixtures.load_fixtures()
up = fx[fx["status"] == "upcoming"].copy()
if args.fixture:
    h, a = [s.strip() for s in args.fixture.split(" vs ")]
    up = up[((up["home"] == h) & (up["away"] == a)) | ((up["home"] == a) & (up["away"] == h))]
    if up.empty:
        raise SystemExit(
            f"No upcoming confirmed fixture {h} vs {a}. "
            "(Knockout pairings appear once FIFA confirms them.)"
        )
elif args.days:
    horizon = up["date"].min() + pd.Timedelta(days=args.days)
    up = up[up["date"] <= horizon]
elif not args.all:
    ap.error("give a fixture, --days N, or --all")

up = up.sort_values("date")
pred = runtime.build_predictor()
for r in up.itertuples(index=False):
    m = runtime.predict_fixture(pred, r.home, r.away, r.date, r.neutral, country=r.host)
    comp = f"Group {r.group}" if r.group else f"Round {r.round}"
    print(runtime.format_prediction(r.home, r.away, str(r.date.date()), comp,
                                    matrix.wdl(m), matrix.top_scorelines(m, 5)))
    print()
