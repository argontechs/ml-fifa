"""ESPN scoreboard as a redundant live-score source.

fixturedownload.com lagged Canada-Bosnia's final score by 75+ minutes (2026-06-12),
leaving the leaderboard, goal events and the tracker blind. ESPN's public scoreboard
is real-time and keyless — we merge it over the feed: FINAL scores flip a fixture to
played; in-progress scores fill the score columns (for goal events) without ever
counting as played. fixturedownload remains the schedule/bracket source of truth.
"""
from __future__ import annotations

import pandas as pd
import requests

from . import data

URL = "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard"


def parse_scoreboard(payload: dict) -> dict[frozenset, dict]:
    """{frozenset({home, away}): {scores: {team: goals}, final: bool, live: bool}}"""
    book: dict[frozenset, dict] = {}
    for ev in payload.get("events", []):
        try:
            comp = ev["competitions"][0]
            sides = {c["homeAway"]: c for c in comp["competitors"]}
            names = {ha: data.normalize_team(c["team"]["displayName"])
                     for ha, c in sides.items()}
            raw = {ha: sides[ha].get("score") for ha in sides}
            if raw.get("home") is None or raw.get("away") is None:
                continue
            state = comp["status"]["type"]
            book[frozenset(names.values())] = {
                "scores": {names[ha]: int(raw[ha]) for ha in names},
                "final": bool(state.get("completed")),
                "live": state.get("state") == "in",
            }
        except Exception:  # noqa: BLE001 — one malformed event must not kill the merge
            continue
    return book


def fetch_scoreboard(dates: list[str]) -> dict[frozenset, dict]:
    book: dict[frozenset, dict] = {}
    for ds in dates:
        try:
            resp = requests.get(URL, params={"dates": ds}, headers=data.UA, timeout=20)
            resp.raise_for_status()
            book.update(parse_scoreboard(resp.json()))
        except Exception as exc:  # noqa: BLE001 — fallback source must never break loads
            print(f"espn scoreboard skipped ({ds}: {exc})")
    return book


def merge_scores(fx: pd.DataFrame, book: dict[frozenset, dict],
                 now: pd.Timestamp) -> pd.DataFrame:
    """Fill missing scores for fixtures whose kickoff has passed."""
    if not book:
        return fx
    fx = fx.copy()
    for i, r in fx.iterrows():
        if r["status"] != "upcoming" or r["date"] > now:
            continue
        rec = book.get(frozenset({r["home"], r["away"]}))
        if rec is None or r["home"] not in rec["scores"]:
            continue
        fx.at[i, "home_score"] = rec["scores"][r["home"]]
        fx.at[i, "away_score"] = rec["scores"][r["away"]]
        if rec["final"]:
            fx.at[i, "status"] = "played"
    return fx


def needed_dates(fx: pd.DataFrame, now: pd.Timestamp, lookback_h: int = 36) -> list[str]:
    """Datestrings (YYYYMMDD) of past-kickoff fixtures the feed hasn't scored yet."""
    mask = ((fx["status"] == "upcoming") & (fx["date"] <= now)
            & (fx["date"] >= now - pd.Timedelta(hours=lookback_h)))
    return sorted({d.strftime("%Y%m%d") for d in fx.loc[mask, "date"]})
