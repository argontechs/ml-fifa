"""Scoreline probability matrix: independent Poisson + Dixon-Coles tau correction."""
from __future__ import annotations

import numpy as np
from scipy.stats import poisson

MAX_GOALS = 10
LOCK, STRONG, LEAN = 0.70, 0.55, 0.45


def score_matrix(lam_home: float, lam_away: float, rho: float = 0.0,
                 max_goals: int = MAX_GOALS) -> np.ndarray:
    """m[i, j] = P(home scores i, away scores j)."""
    g = np.arange(max_goals + 1)
    m = np.outer(poisson.pmf(g, lam_home), poisson.pmf(g, lam_away))
    # Dixon-Coles tau on the four low-score cells
    m[0, 0] *= 1 - lam_home * lam_away * rho
    m[0, 1] *= 1 + lam_home * rho
    m[1, 0] *= 1 + lam_away * rho
    m[1, 1] *= 1 - rho
    np.clip(m, 0.0, None, out=m)
    return m / m.sum()


def wdl(m: np.ndarray) -> tuple[float, float, float]:
    return (
        float(np.tril(m, -1).sum()),   # home win: i > j
        float(np.trace(m)),
        float(np.triu(m, 1).sum()),    # away win: j > i
    )


def top_scorelines(m: np.ndarray, k: int = 5) -> list[tuple[tuple[int, int], float]]:
    flat = [((i, j), float(m[i, j])) for i in range(m.shape[0]) for j in range(m.shape[1])]
    return sorted(flat, key=lambda t: -t[1])[:k]


def rescale_wdl(m: np.ndarray, target: tuple[float, float, float]) -> np.ndarray:
    """Rescale the win/draw/loss regions of a score matrix to hit target W/D/L probs,
    preserving the relative scoreline shape inside each region."""
    ph, pd_, pa = wdl(m)
    out = m.copy()
    n = m.shape[0]
    il, iu, di = np.tril_indices(n, -1), np.triu_indices(n, 1), np.diag_indices(n)
    out[il] *= target[0] / max(ph, 1e-12)
    out[di] *= target[1] / max(pd_, 1e-12)
    out[iu] *= target[2] / max(pa, 1e-12)
    return out / out.sum()


def p_over(m: np.ndarray, line: float = 2.5) -> float:
    """P(total goals > line) — a readout of the same matrix, e.g. over 2.5 = i+j >= 3."""
    n = m.shape[0]
    i, j = np.indices((n, n))
    return float(m[(i + j) > line].sum())


def double_chance(p: tuple[float, float, float]) -> tuple[str, float]:
    """The 'safer' two-way cover: drop the least-likely single outcome, keep the rest.
    p = (home, draw, away) → ('1X' | '12' | 'X2', combined prob).
    drop away → 1X (home or draw) · drop draw → 12 (home or away) · drop home → X2 (draw or away)."""
    drop = min(range(3), key=lambda k: p[k])
    label = {0: "X2", 1: "12", 2: "1X"}[drop]
    return label, float(sum(p) - p[drop])


def tier(p_max: float) -> str:
    if p_max >= LOCK:
        return "LOCK"
    if p_max >= STRONG:
        return "STRONG"
    if p_max >= LEAN:
        return "LEAN"
    return "TOSS-UP"
