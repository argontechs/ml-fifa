"""Bluesky Jetstream consumer: keyword-route posts to active matches, store raw."""
from __future__ import annotations

import asyncio
import json
import re
import time

import pandas as pd

from . import db, match_window

JETSTREAM_URL = (
    "wss://jetstream2.us-east.bsky.network/subscribe"
    "?wantedCollections=app.bsky.feed.post"
)


# words that, immediately preceding a team term, mean it is NOT the team
# ("new mexico", "new england", "air jordan", "michael jordan")
_COMPOUND_BLOCKERS = {"mexico": {"new"}, "england": {"new"}, "jordan": {"air", "michael"}}


def _term_hit(low: str, term: str) -> bool:
    """Word-boundary match (audit: bare substring matched 'usa' in 'thousands')."""
    for m in re.finditer(r"(?<![a-z0-9])" + re.escape(term) + r"(?![a-z0-9])", low):
        before = low[: m.start()].rstrip()
        prev = before.split()[-1] if before else ""
        if prev in _COMPOUND_BLOCKERS.get(term, ()):
            continue
        return True
    return False


def route(text: str, kwsets: dict[str, set[str]]) -> str | None:
    """home | away | both | None for one match's keyword sets."""
    low = text.lower()
    if any(_term_hit(low, tag) for tag in kwsets["both"]):
        return "both"
    home_hit = any(_term_hit(low, k) for k in kwsets["home"])
    away_hit = any(_term_hit(low, k) for k in kwsets["away"])
    if home_hit and away_hit:
        return "both"
    if home_hit:
        return "home"
    if away_hit:
        return "away"
    return None


def _cursor_url(url: str, cursor: dict) -> str:
    """Jetstream resumes from time_us — reconnects without it silently drop posts."""
    t = cursor.get("time_us")
    if not t:
        return url
    sep = "&" if "?" in url else "?"
    return f"{url}{sep}cursor={t}"


def _extract_text(raw: str, cursor: dict | None = None) -> str | None:
    try:
        msg = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    if cursor is not None and isinstance(msg.get("time_us"), int):
        cursor["time_us"] = msg["time_us"]
    if msg.get("kind") != "commit":
        return None
    commit = msg.get("commit") or {}
    if commit.get("operation") != "create":
        return None
    if commit.get("collection") != "app.bsky.feed.post":
        return None
    text = (commit.get("record") or {}).get("text")
    return text if isinstance(text, str) and text.strip() else None


async def consume(source, conn, windows_fn, clock=time.time, source_tag="bsky",
                  cursor: dict | None = None) -> int:
    """Drain `source` (async iterator of raw Jetstream JSON). Returns posts stored."""
    stored = 0
    async for raw in source:
        try:  # one bad message/db hiccup must never kill collection (audit)
            text = _extract_text(raw, cursor)
            if text is None:
                continue
            for match, kwsets in windows_fn():
                side = route(text, kwsets)
                if side is not None:
                    db.insert_post(conn, ts=clock(), source=source_tag,
                                   match_key=match["key"], side=side, text=text)
                    stored += 1
        except Exception as exc:  # noqa: BLE001
            print(f"collector: message skipped ({exc})")
    return stored


async def jetstream_source(url: str = JETSTREAM_URL, cursor: dict | None = None):
    """Yield raw messages forever; reconnect with exponential backoff on drop,
    resuming from the last seen time_us so outages don't lose posts."""
    import websockets

    backoff = 1.0
    cursor = cursor if cursor is not None else {}
    while True:
        try:
            async with websockets.connect(_cursor_url(url, cursor),
                                          ping_interval=30) as ws:
                backoff = 1.0
                async for raw in ws:
                    yield raw
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 — reconnect on anything else
            print(f"jetstream reconnect in {backoff:.0f}s ({exc})")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60.0)


def windows_from_fixtures(ttl: float = 60.0):
    """Returns windows_fn: cached [(match, kwsets)] for currently active matches."""
    from fifa import fixtures

    cache: dict = {"at": 0.0, "value": []}

    def windows_fn():
        now = time.time()
        if now - cache["at"] > ttl:
            fx = fixtures.load_fixtures()
            active = match_window.active_matches(
                fx, pd.Timestamp.now(tz="UTC").tz_localize(None))
            cache["value"] = [(m, match_window.keywords_for(m)) for m in active]
            cache["at"] = now
            names = [f"{m['home']} v {m['away']}" for m, _ in cache["value"]]
            print(f"active matches: {names or 'none'}")
        return cache["value"]

    return windows_fn
