"""Collector process: Bluesky Jetstream (or a replay file) → data/sentiment.db.

usage:
  .venv/bin/python sentiment_collect.py                 # live firehose + goal polling
  .venv/bin/python sentiment_collect.py --replay tests/replay_sample.jsonl --speed 50
"""
import argparse
import asyncio

import pandas as pd

from fifa import data, fixtures
from sentiment import collector, db, events, match_window

DB_PATH = data.DATA_DIR / "sentiment.db"


async def replay_source(path: str, speed: float):
    delay = 2.0 / max(speed, 0.01)  # posts arrive ~2s apart at speed 1
    with open(path) as f:
        for line in f:
            yield line.rstrip("\n")
            await asyncio.sleep(delay)


async def goal_poller(conn, interval: float = 60.0):
    while True:
        try:
            fx = fixtures.load_fixtures()
            now = pd.Timestamp.now(tz="UTC").tz_localize(None)
            active = {m["key"] for m in match_window.active_matches(fx, now)}
            n = events.poll_once(conn, fx, active)
            if n:
                print(f"⚽ {n} goal event(s) recorded")
        except Exception as exc:  # noqa: BLE001 — polling must never kill collection
            print(f"goal poll skipped ({exc})")
        await asyncio.sleep(interval)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay")
    ap.add_argument("--speed", type=float, default=50.0)
    args = ap.parse_args()

    conn = db.connect(DB_PATH)
    if args.replay:
        # replay assumes the sample's Mexico-South Africa match regardless of clock
        match = {"key": 1, "home": "Mexico", "away": "South Africa"}
        windows = [(match, match_window.keywords_for(match))]
        db.upsert_match(conn, 1, "Mexico", "South Africa", "replay", "0-0")
        stored = await collector.consume(replay_source(args.replay, args.speed), conn,
                                         windows_fn=lambda: windows)
        print(f"replay done — {stored} posts stored in {DB_PATH}")
        return
    print("collector running — Ctrl-C to stop")
    await asyncio.gather(
        collector.consume(collector.jetstream_source(), conn,
                          windows_fn=collector.windows_from_fixtures()),
        goal_poller(conn),
    )


if __name__ == "__main__":
    asyncio.run(main())
