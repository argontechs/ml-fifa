"""Prediction ledger: freeze pre-kickoff predictions, score them once matches finish.

First write per match_number wins — predictions cannot be quietly revised after the
fact. This is the data behind the dashboard's honesty tracker.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import evaluate


def load(path: Path) -> dict[int, dict]:
    book: dict[int, dict] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                book.setdefault(rec["match_number"], rec)  # first write wins
    return book


def record(preds: list[dict], path: Path) -> int:
    """Append predictions not already frozen. Returns number of new records."""
    book = load(path)
    new = [p for p in preds if p["match_number"] not in book]
    if new:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            for p in new:
                f.write(json.dumps(p) + "\n")
    return len(new)


def tracker(book: dict[int, dict], fixtures) -> tuple[list[dict], dict]:
    """Join frozen predictions against played fixtures. Returns (rows, tally)."""
    rows = []
    for r in fixtures.itertuples(index=False):
        if r.status != "played" or r.match_number not in book:
            continue
        rec = book[r.match_number]
        hs, as_ = int(r.home_score), int(r.away_score)
        ph, pa = rec["predicted"]
        outcome_hit = evaluate.outcome_of(hs, as_) == evaluate.outcome_of(ph, pa)
        exact_hit = (hs, as_) == (ph, pa)
        rows.append({
            "match_number": r.match_number,
            "home": rec["home"], "away": rec["away"],
            "predicted": (ph, pa), "actual": (hs, as_),
            "tier": rec.get("tier", ""),
            "outcome": outcome_hit, "exact": exact_hit,
        })
    tally = {
        "n": len(rows),
        "outcome_hits": sum(r["outcome"] for r in rows),
        "exact_hits": sum(r["exact"] for r in rows),
        "lock_n": sum(r["tier"] == "LOCK" for r in rows),
        "lock_hits": sum(r["outcome"] for r in rows if r["tier"] == "LOCK"),
    }
    return rows, tally
