"""Freeze a pre-match crowd-mood snapshot at kickoff, for post-tournament evaluation.

One JSONL row per match, written the first time a kickoff is observed to have passed
(first-write-wins — snapshots cannot be revised after the match starts). After the group
stage these rows get joined against the prediction ledger and results to test whether
pre-match sentiment carries signal beyond the bookmaker blend. Until that test passes,
sentiment is NEVER a model feature.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from . import db


def load(path: Path) -> dict[int, dict]:
    book: dict[int, dict] = {}
    if Path(path).exists():
        for line in Path(path).read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                book.setdefault(rec["match_number"], rec)
    return book


def _mood(posts: pd.DataFrame, side: str) -> float | None:
    s = posts.loc[posts["side"] == side, "score"].dropna()
    return round(float(s.mean()), 4) if len(s) else None


def record_kickoffs(conn, fx: pd.DataFrame, path: Path, now: pd.Timestamp,
                    lookback_min: int = 30) -> int:
    """Snapshot matches whose kickoff passed within the last `lookback_min` minutes."""
    book = load(path)
    new = 0
    for r in fx.itertuples(index=False):
        if r.status == "tbd" or int(r.match_number) in book:
            continue
        if not (now - pd.Timedelta(minutes=lookback_min) <= r.date <= now):
            continue
        kickoff_epoch = pd.Timestamp(r.date, tz="UTC").timestamp()
        posts = db.posts_frame(conn, int(r.match_number))
        pre = posts[posts["ts"] <= kickoff_epoch]
        rec = {
            "match_number": int(r.match_number), "home": r.home, "away": r.away,
            "kickoff": str(r.date), "n_posts": int(len(pre)),
            "mood_home": _mood(pre, "home"), "mood_away": _mood(pre, "away"),
            "mood_both": _mood(pre, "both"),
            "frozen_at": str(now),
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with Path(path).open("a") as f:
            f.write(json.dumps(rec) + "\n")
        book[rec["match_number"]] = rec
        new += 1
    return new
