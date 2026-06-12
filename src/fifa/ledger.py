"""Prediction ledger: freeze pre-kickoff predictions, score them once matches finish.

First write per match_number wins — predictions cannot be quietly revised after the
fact. This is the data behind the dashboard's honesty tracker.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import evaluate


def _key(rec_or_row) -> tuple:
    if isinstance(rec_or_row, dict):
        return (int(rec_or_row["match_number"]), rec_or_row["home"], rec_or_row["away"])
    return (int(rec_or_row.match_number), rec_or_row.home, rec_or_row.away)


def load(path: Path) -> dict[tuple, dict]:
    """Keyed by (match_number, home, away): a feed reshuffle of a pairing under the
    same match number gets a fresh frozen pick instead of mis-scoring the old one."""
    book: dict[tuple, dict] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                book.setdefault(_key(rec), rec)  # first write wins per pairing
    return book


def record(preds: list[dict], path: Path) -> int:
    """Append predictions not already frozen. Returns number of new records."""
    book = load(path)
    new = [p for p in preds if _key(p) not in book]
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
        if r.status != "played" or _key(r) not in book:
            continue
        rec = book[_key(r)]
        hs, as_ = int(r.home_score), int(r.away_score)
        ph, pa = rec["predicted"]
        # Outcome is judged on the W/D/L probabilities, not the modal scoreline —
        # the most likely single score can be 1-1 while a home win is the most
        # likely outcome overall.
        if rec.get("p"):
            picked = max(range(3), key=lambda k: rec["p"][k])
        else:
            picked = evaluate.outcome_of(ph, pa)
        outcome_hit = evaluate.outcome_of(hs, as_) == picked
        exact_hit = (hs, as_) == (ph, pa)
        rows.append({
            "match_number": r.match_number,
            "home": rec["home"], "away": rec["away"],
            "predicted": (ph, pa), "actual": (hs, as_),
            "tier": rec.get("tier", ""),
            "p": tuple(rec.get("p", ())),
            "kickoff": rec.get("kickoff", ""),
            "retro": rec.get("retro", False),
            "outcome": outcome_hit, "exact": exact_hit,
        })
    # cohorts are tallied separately: headline numbers must come from genuinely
    # frozen pre-kickoff picks; RETRO (walk-forward) rows are reported on their own
    frozen = [r for r in rows if not r.get("retro")]
    retro_rows = [r for r in rows if r.get("retro")]
    tally = {
        "n": len(frozen),
        "outcome_hits": sum(r["outcome"] for r in frozen),
        "exact_hits": sum(r["exact"] for r in frozen),
        "lock_n": sum(r["tier"] == "LOCK" for r in frozen),
        "lock_hits": sum(r["outcome"] for r in frozen if r["tier"] == "LOCK"),
        "retro_n": len(retro_rows),
        "retro_outcome_hits": sum(r["outcome"] for r in retro_rows),
        "retro_exact_hits": sum(r["exact"] for r in retro_rows),
    }
    return rows, tally
