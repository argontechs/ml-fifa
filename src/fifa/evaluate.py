"""Honest scoring: RPS, log loss, Brier, exact-score vs 1-1 baseline, tier accuracy."""
from __future__ import annotations

import numpy as np

from . import matrix as mx


def outcome_of(home_score: int, away_score: int) -> int:
    """0 = home win, 1 = draw, 2 = away win."""
    if home_score > away_score:
        return 0
    if home_score == away_score:
        return 1
    return 2


def rps(p: tuple[float, float, float], outcome: int) -> float:
    cum_p = np.cumsum(p)[:2]
    cum_o = np.cumsum(np.eye(3)[outcome])[:2]
    return float(np.sum((cum_p - cum_o) ** 2) / 2)


def score_prediction(m: np.ndarray, home_score: int, away_score: int) -> dict:
    """Everything we need to grade one match prediction."""
    p = mx.wdl(m)
    out = outcome_of(home_score, away_score)
    top5 = mx.top_scorelines(m, 5)
    pred_score = top5[0][0]
    p_max = max(p)
    return {
        "p": p,
        "outcome": out,
        "picked": int(np.argmax(p)),
        "tier": mx.tier(p_max),
        "rps": rps(p, out),
        "logloss": -float(np.log(max(p[out], 1e-12))),
        "brier": float(np.sum((np.array(p) - np.eye(3)[out]) ** 2)),
        "exact": pred_score == (home_score, away_score),
        "top5": (home_score, away_score) in [s for s, _ in top5],
        "is_11": (home_score, away_score) == (1, 1),
    }


def report_card(rows: list[dict]) -> dict:
    n = len(rows)
    locks = [r for r in rows if r["tier"] == "LOCK"]
    draws = [r for r in rows if r["outcome"] == 1]
    return {
        "n": n,
        "wdl_acc": sum(r["picked"] == r["outcome"] for r in rows) / n,
        "draw_recall": (sum(r["picked"] == 1 for r in draws) / len(draws)) if draws else None,
        "rps": sum(r["rps"] for r in rows) / n,
        "logloss": sum(r["logloss"] for r in rows) / n,
        "brier": sum(r["brier"] for r in rows) / n,
        "exact_rate": sum(r["exact"] for r in rows) / n,
        "top5_rate": sum(r["top5"] for r in rows) / n,
        "baseline11_rate": sum(r["is_11"] for r in rows) / n,
        "lock_n": len(locks),
        "lock_acc": (sum(r["picked"] == r["outcome"] for r in locks) / len(locks)) if locks else None,
    }


def format_card(card: dict, gates: bool = True) -> str:
    draw_recall = (
        f"draw recall            {card['draw_recall']:.1%}"
        if card["draw_recall"] is not None
        else "draw recall            n/a"
    )
    lock = f"{card['lock_acc']:.1%}" if card["lock_acc"] is not None else "n/a"
    lines = [
        f"matches evaluated      {card['n']}",
        f"W/D/L accuracy         {card['wdl_acc']:.1%}",
        draw_recall,
        f"mean RPS               {card['rps']:.4f}",
        f"mean log loss          {card['logloss']:.4f}",
        f"exact-score rate       {card['exact_rate']:.1%}  (always-1-1 baseline: {card['baseline11_rate']:.1%})",
        f"top-5 scoreline rate   {card['top5_rate']:.1%}",
        f"LOCK picks             {card['lock_n']}  acc {lock}",
    ]
    if gates:
        checks = [
            ("W/D/L acc in honest band (>=0.50)", card["wdl_acc"] >= 0.50),
            ("exact-score beats 1-1 baseline", card["exact_rate"] >= card["baseline11_rate"]),
            ("exact-score below leakage alarm (<0.15)", card["exact_rate"] < 0.15),
            ("RPS <= 0.215", card["rps"] <= 0.215),
            ("draws actually predicted (recall > 0)", (card["draw_recall"] or 0) > 0),
        ]
        lines.append("gates:")
        lines += [f"  [{'PASS' if ok else 'FAIL'}] {name}" for name, ok in checks]
    return "\n".join(lines)
