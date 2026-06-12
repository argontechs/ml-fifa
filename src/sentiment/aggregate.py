"""Read-side aggregation for the dashboard: per-minute sentiment buckets, tallies."""
from __future__ import annotations

import pandas as pd

from . import db


def timeline(conn, match_key: int, bucket_s: int = 60,
             halflife_buckets: float = 3.0) -> pd.DataFrame:
    """Per-bucket mean sentiment and volume per side, with EWMA-smoothed columns."""
    posts = db.posts_frame(conn, match_key)
    scored = posts[posts["score"].notna()].copy()
    if scored.empty:
        return pd.DataFrame()
    scored["bucket"] = (scored["ts"] // bucket_s).astype(int)
    out = None
    for side in ("home", "away", "both"):
        sub = scored[scored["side"] == side]
        agg = sub.groupby("bucket")["score"].agg(["mean", "count"])
        agg.columns = [f"{side}_mean", f"{side}_n"]
        out = agg if out is None else out.join(agg, how="outer")
    out = out.sort_index()
    for side in ("home", "away", "both"):
        out[f"{side}_n"] = out[f"{side}_n"].fillna(0).astype(int)
        out[f"{side}_ewma"] = (
            out[f"{side}_mean"].ewm(halflife=halflife_buckets, ignore_na=True).mean()
        )
    return out.reset_index()


def tallies(conn, match_key: int, window_min: int = 15) -> dict:
    posts = db.posts_frame(conn, match_key)
    scored = posts[posts["score"].notna()]
    cutoff = posts["ts"].max() - window_min * 60 if len(posts) else 0
    recent = scored[scored["ts"] >= cutoff]
    def mood(side):
        s = recent.loc[recent["side"] == side, "score"]
        return float(s.mean()) if len(s) else None
    return {
        "total_posts": int(len(posts)),
        "scored": int(scored.shape[0]),
        "mood_home": mood("home"),
        "mood_away": mood("away"),
        "rate_per_min": round(len(recent) / max(window_min, 1), 1),
    }


def recent_posts(conn, match_key: int, n: int = 12) -> pd.DataFrame:
    posts = db.posts_frame(conn, match_key)
    return posts.sort_values("ts", ascending=False).head(n)
