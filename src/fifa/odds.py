"""Bookmaker consensus odds via The Odds API. Optional: absent key → empty book."""
from __future__ import annotations

import json
import os
import time

import numpy as np

from . import data

SPORT = "soccer_fifa_world_cup"
URL = f"https://api.the-odds-api.com/v4/sports/{SPORT}/odds"
# the blend weight lives in ensemble.market_weight() (depth-aware); odds only supplies
# the consensus probabilities + how many books backed them


def _api_key() -> str | None:
    key = os.environ.get("ODDS_API_KEY")
    if key:
        return key
    f = data.DATA_DIR / "odds_api_key.txt"
    return f.read_text().strip() if f.exists() else None


def parse_feed(feed) -> dict[tuple[str, str], tuple[float, float, float, int]]:
    """(home, away) → consensus devigged (p_home, p_draw, p_away, n_books)."""
    book = {}
    for match in feed:
        home = data.normalize_team(match["home_team"])
        away = data.normalize_team(match["away_team"])
        probs = []
        for bm in match.get("bookmakers", []):
            for mkt in bm.get("markets", []):
                if mkt["key"] != "h2h":
                    continue
                prices = {o["name"]: o["price"] for o in mkt["outcomes"]}
                if not {match["home_team"], match["away_team"], "Draw"} <= prices.keys():
                    continue
                raw = np.array([
                    1 / prices[match["home_team"]],
                    1 / prices["Draw"],
                    1 / prices[match["away_team"]],
                ])
                probs.append(raw / raw.sum())  # devig: normalize the overround away
        if probs:
            p = np.mean(probs, axis=0)
            book[(home, away)] = (float(p[0]), float(p[1]), float(p[2]), len(probs))
    return book


def fetch_book(force: bool = False) -> dict:
    key = _api_key()
    if not key:
        print("NOTE: no ODDS_API_KEY — predictions are model-only")
        return {}
    cache = data.DATA_DIR / "odds_cache.json"
    # 6h cache: repeated runs spend ZERO credits (each live fetch costs 2 of 500/mo)
    if not force and cache.exists() and time.time() - cache.stat().st_mtime < 6 * 3600:
        return parse_feed(json.loads(cache.read_text()))
    try:
        import requests

        resp = requests.get(
            URL,
            params={"apiKey": key, "regions": "eu,uk", "markets": "h2h"},
            timeout=30,
        )
        resp.raise_for_status()
        cache.write_text(resp.text)
        return parse_feed(resp.json())
    except Exception as exc:  # noqa: BLE001 — odds must never break predictions
        if cache.exists():
            print(f"WARNING: odds fetch failed ({exc}); using cached odds")
            return parse_feed(json.loads(cache.read_text()))
        print(f"WARNING: odds unavailable ({exc}); model-only")
        return {}
