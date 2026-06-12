"""Scorer process: loads the multilingual model once, scores posts forever.

usage: .venv/bin/python sentiment_score.py [--once]
"""
import argparse

from fifa import data
from sentiment import db, scorer

ap = argparse.ArgumentParser()
ap.add_argument("--once", action="store_true", help="score current backlog and exit")
args = ap.parse_args()

conn = db.connect(data.DATA_DIR / "sentiment.db")
model = scorer.load_model()
if args.once:
    total = 0
    while True:
        n = scorer.run_once(conn, model)
        if not n:
            break
        total += n
    print(f"scored {total} posts")
else:
    scorer.run_loop(conn, model)
