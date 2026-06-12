"""Goal-event detection: diff live fixture scores against the last seen state."""
from __future__ import annotations

import pandas as pd

from . import db


def poll_once(conn, fx: pd.DataFrame, active_keys: set[int]) -> int:
    """Compare current scores for active matches; record goal events on deltas.
    Returns number of new events."""
    new = 0
    for r in fx.itertuples(index=False):
        key = int(r.match_number)
        if key not in active_keys or r.home_score is None:
            continue
        status = f"{int(r.home_score)}-{int(r.away_score)}"
        row = conn.execute(
            "SELECT status FROM matches WHERE match_key=?", (key,)
        ).fetchone()
        if row is None:
            # first sighting: record baseline silently (no phantom goal event)
            db.upsert_match(conn, key, r.home, r.away, str(r.date), status)
            continue
        if row[0] != status:
            old_h, old_a = (int(x) for x in row[0].split("-"))
            team = r.home if int(r.home_score) > old_h else r.away
            db.record_score_change(conn, key, team, status)
            new += 1
    return new
