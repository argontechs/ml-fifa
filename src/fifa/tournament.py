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


# ---------------------------------------------------------------------------
# Official knockout structure.
# R32 seeds verbatim from the fixturedownload feed (matches 73-88):
# "1X"/"2X" = winner/runner-up of group X; "3ABCDF" = a qualified third-placed
# team drawn from those candidate groups. Later rounds chain by match number
# per the published bracket (Wikipedia: 2026 FIFA World Cup knockout stage).
# ---------------------------------------------------------------------------
R32_SEEDS = {
    73: ("2A", "2B"), 74: ("1E", "3ABCDF"), 75: ("1F", "2C"), 76: ("1C", "2F"),
    77: ("1I", "3CDFGH"), 78: ("2E", "2I"), 79: ("1A", "3CEFHI"), 80: ("1L", "3EHIJK"),
    81: ("1D", "3BEFIJ"), 82: ("1G", "3AEHIJ"), 83: ("2K", "2L"), 84: ("1H", "2J"),
    85: ("1B", "3EFGIJ"), 86: ("1J", "2H"), 87: ("1K", "3DEIJL"), 88: ("2D", "2G"),
}
CHAIN = {
    89: (73, 75), 90: (74, 77), 91: (76, 78), 92: (79, 80),
    93: (83, 84), 94: (81, 82), 95: (86, 88), 96: (85, 87),
    97: (89, 90), 98: (93, 94), 99: (91, 92), 100: (95, 96),
    101: (97, 98), 102: (99, 100), 104: (101, 102),
}
STAGE_OF = {**{m: "r16" for m in range(89, 97)}, **{m: "qf" for m in range(97, 101)},
            101: "sf", 102: "sf", 104: "final"}


def _sample_score(m, rng) -> tuple[int, int]:
    flat = m.ravel()
    k = rng.choice(len(flat), p=flat / flat.sum())
    n = m.shape[0]
    return int(k // n), int(k % n)


def _lambdas_of(m):
    g = np.arange(m.shape[0])
    return float((m.sum(axis=1) * g).sum()), float((m.sum(axis=0) * g).sum())


def _play_group_stage(fixtures, predictor, rng):
    """Returns ({group: ordered teams}, [(third_team, stats)]), sampling unplayed matches."""
    group_rows = fixtures[fixtures["group"].notna()]
    results_by_group: dict[str, list] = {}
    teams_by_group: dict[str, list] = {}
    for r in group_rows.itertuples(index=False):
        g = r.group
        teams_by_group.setdefault(g, [])
        for t in (r.home, r.away):
            if t not in teams_by_group[g]:
                teams_by_group[g].append(t)
        if r.status == "played":
            hs, as_ = int(r.home_score), int(r.away_score)
        else:
            m = predictor.matrix_for(r.home, r.away, r.date, "FIFA World Cup", r.neutral,
                                     country=getattr(r, "host", None))
            hs, as_ = _sample_score(m, rng)
        results_by_group.setdefault(g, []).append((r.home, r.away, hs, as_))
    standings, thirds = {}, []
    for g, teams in teams_by_group.items():
        order = rank_group(teams, results_by_group[g], rng)
        standings[g] = order
        t = _table(teams, results_by_group[g])
        thirds.append((order[2], t[order[2]]))
    return standings, thirds


def simulate_groups(fixtures, predictor, n_runs=10000, seed=0) -> dict:
    """Group-stage Monte Carlo: P(win group), P(top-2), P(advance incl. best-third)."""
    rng = np.random.default_rng(seed)
    teams = sorted(
        set(fixtures.loc[fixtures["group"].notna(), "home"])
        | set(fixtures.loc[fixtures["group"].notna(), "away"])
    )
    win = {t: 0 for t in teams}
    top2 = {t: 0 for t in teams}
    adv = {t: 0 for t in teams}
    for _ in range(n_runs):
        standings, thirds = _play_group_stage(fixtures, predictor, rng)
        qualified = set(best_thirds(thirds, rng))
        for order in standings.values():
            win[order[0]] += 1
            top2[order[0]] += 1
            top2[order[1]] += 1
            for pos, t in enumerate(order):
                if pos < 2 or t in qualified:
                    adv[t] += 1
    return {
        "win_group": {t: c / n_runs for t, c in win.items()},
        "top2": {t: c / n_runs for t, c in top2.items()},
        "advance": {t: c / n_runs for t, c in adv.items()},
    }


def _assign_thirds(qualified, rng, max_tries=200):
    """Assign the 8 qualified thirds (by group letter) to the 8 third-slots so that
    each slot's team comes from its candidate set. Random valid assignment."""
    slots = [(mn, seed) for mn, pair in R32_SEEDS.items() for seed in pair
             if seed.startswith("3")]
    by_group = dict(qualified)  # group letter → team
    letters = list(by_group)
    for _ in range(max_tries):
        rng.shuffle(letters)
        assignment, used = {}, set()
        ok = True
        for mn, seed in slots:
            cands = [g for g in letters if g in seed[1:] and g not in used]
            if not cands:
                ok = False
                break
            pick = cands[0]
            assignment[(mn, seed)] = by_group[pick]
            used.add(pick)
        if ok:
            return assignment
    # Extremely unlikely fallback: ignore candidate constraints
    rng.shuffle(letters)
    return {slot: by_group[g] for slot, g in zip(slots, letters)}


def _knockout_match(t1, t2, predictor, rng, elo_ratings):
    # 13 of 16 R32 venues and all matches from the QF onward are in the US —
    # documented approximation for the displacement feature in knockouts.
    m = predictor.matrix_for(t1, t2, pd.Timestamp("2026-07-01"), "FIFA World Cup", True,
                             country="United States")
    hs, as_ = _sample_score(m, rng)
    if hs != as_:
        return t1 if hs > as_ else t2
    lh, la = _lambdas_of(m)
    met = mx.score_matrix(lh / 3, la / 3, rho=0.0, max_goals=5)
    ehs, eas = _sample_score(met, rng)
    if ehs != eas:
        return t1 if ehs > eas else t2
    from . import elo as elo_mod
    p1 = elo_mod.expected(elo_ratings.get(t1, 1500), elo_ratings.get(t2, 1500), neutral=True)
    return t1 if rng.random() < p1 else t2


def simulate_tournament(fixtures, predictor, elo_ratings, n_runs=10000, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    counters: dict[str, dict[str, int]] = {}

    def bump(team, stage):
        counters.setdefault(team, {}).setdefault(stage, 0)
        counters[team][stage] += 1

    for _ in range(n_runs):
        standings, thirds = _play_group_stage(fixtures, predictor, rng)
        qualified_set = set(best_thirds(thirds, rng))
        qualified = [(g, standings[g][2]) for g in standings if standings[g][2] in qualified_set]
        third_assignment = _assign_thirds(qualified, rng)

        def resolve(seed_token, match_no):
            kind, rest = seed_token[0], seed_token[1:]
            if kind == "1":
                return standings[rest][0]
            if kind == "2":
                return standings[rest][1]
            return third_assignment[(match_no, seed_token)]

        winners: dict[int, str] = {}
        for mn, (s1, s2) in R32_SEEDS.items():
            t1, t2 = resolve(s1, mn), resolve(s2, mn)
            bump(t1, "r32")
            bump(t2, "r32")
            winners[mn] = _knockout_match(t1, t2, predictor, rng, elo_ratings)
        for mn in sorted(CHAIN):
            a, b = CHAIN[mn]
            t1, t2 = winners[a], winners[b]
            stage = STAGE_OF[mn]
            bump(t1, stage)
            bump(t2, stage)
            winners[mn] = _knockout_match(t1, t2, predictor, rng, elo_ratings)
        bump(winners[104], "champion")

    teams = sorted(counters)
    stages = ["r32", "r16", "qf", "sf", "final", "champion"]
    out = pd.DataFrame(
        {s: [counters[t].get(s, 0) / n_runs for t in teams] for s in stages}, index=teams
    )
    return out.sort_values("champion", ascending=False)
