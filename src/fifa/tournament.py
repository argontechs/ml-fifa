"""WC2026 tournament mechanics: group ranking, best thirds, Monte Carlo simulation."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import matrix as mx


def _table(teams, results):
    """results: list of (home, away, hs, as). Returns {team: {pts, gd, gf}}."""
    t = {x: {"pts": 0, "gd": 0, "gf": 0} for x in teams}
    for h, a, hs, as_ in results:
        t[h]["gf"] += hs
        t[h]["gd"] += hs - as_
        t[a]["gf"] += as_
        t[a]["gd"] += as_ - hs
        if hs > as_:
            t[h]["pts"] += 3
        elif hs < as_:
            t[a]["pts"] += 3
        else:
            t[h]["pts"] += 1
            t[a]["pts"] += 1
    return t


def rank_group(teams, results, rng) -> list[str]:
    """FIFA group ranking: pts, gd, gf, then head-to-head mini-table, then lot (rng)."""
    t = _table(teams, results)
    block = sorted(teams, key=lambda x: (t[x]["pts"], t[x]["gd"], t[x]["gf"]), reverse=True)
    out, i = [], 0
    while i < len(block):
        key_i = (t[block[i]]["pts"], t[block[i]]["gd"], t[block[i]]["gf"])
        tied = [x for x in block if (t[x]["pts"], t[x]["gd"], t[x]["gf"]) == key_i]
        if len(tied) > 1:
            sub = [r for r in results if r[0] in tied and r[1] in tied]
            st = _table(tied, sub)
            tied = sorted(
                tied,
                key=lambda x: (st[x]["pts"], st[x]["gd"], st[x]["gf"], rng.random()),
                reverse=True,
            )
        out.extend(tied)
        i += len(tied)
    return out


def best_thirds(thirds, rng) -> list[str]:
    """thirds: list of (team, {pts, gd, gf}). Top 8 by pts, gd, gf, lot."""
    ranked = sorted(
        thirds,
        key=lambda kv: (kv[1]["pts"], kv[1]["gd"], kv[1]["gf"], rng.random()),
        reverse=True,
    )
    return [team for team, _ in ranked[:8]]
