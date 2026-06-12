"""SQLite storage shared by collector, scorer and the Dash app. WAL, one conn per process."""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pandas as pd

_SCHEMA = """
CREATE TABLE IF NOT EXISTS posts(
    id INTEGER PRIMARY KEY,
    ts REAL NOT NULL,
    source TEXT NOT NULL,
    match_key INTEGER NOT NULL,
    side TEXT NOT NULL,
    text TEXT NOT NULL,
    score REAL
);
CREATE INDEX IF NOT EXISTS idx_posts_match_ts ON posts(match_key, ts);
CREATE INDEX IF NOT EXISTS idx_posts_unscored ON posts(id) WHERE score IS NULL;
CREATE TABLE IF NOT EXISTS matches(
    match_key INTEGER PRIMARY KEY,
    home TEXT, away TEXT, kickoff TEXT, status TEXT
);
CREATE TABLE IF NOT EXISTS events(
    id INTEGER PRIMARY KEY,
    match_key INTEGER NOT NULL,
    ts REAL NOT NULL,
    team TEXT,
    kind TEXT NOT NULL,
    detail TEXT
);
"""


def connect(path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path, timeout=5.0)
    conn.execute("PRAGMA journal_mode=wal")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.executescript(_SCHEMA)
    conn.commit()
    return conn


def insert_post(conn, ts: float, source: str, match_key: int, side: str, text: str) -> None:
    conn.execute(
        "INSERT INTO posts(ts, source, match_key, side, text) VALUES(?,?,?,?,?)",
        (ts, source, match_key, side, text),
    )
    conn.commit()


def unscored_batch(conn, limit: int = 128) -> list[tuple[int, str]]:
    return conn.execute(
        "SELECT id, text FROM posts WHERE score IS NULL ORDER BY id LIMIT ?", (limit,)
    ).fetchall()


def set_scores(conn, ids: list[int], scores: list[float]) -> None:
    conn.executemany("UPDATE posts SET score=? WHERE id=?", list(zip(scores, ids)))
    conn.commit()


def upsert_match(conn, key: int, home: str, away: str, kickoff: str, status: str) -> None:
    conn.execute(
        "INSERT INTO matches(match_key, home, away, kickoff, status) VALUES(?,?,?,?,?) "
        "ON CONFLICT(match_key) DO UPDATE SET status=excluded.status",
        (key, home, away, kickoff, status),
    )
    conn.commit()


def record_score_change(conn, key: int, team: str | None, new_status: str) -> bool:
    """Update matches.status; insert a goal event if it changed. Returns True on delta."""
    row = conn.execute("SELECT status FROM matches WHERE match_key=?", (key,)).fetchone()
    if row is not None and row[0] == new_status:
        return False
    conn.execute("UPDATE matches SET status=? WHERE match_key=?", (new_status, key))
    conn.execute(
        "INSERT INTO events(match_key, ts, team, kind, detail) VALUES(?,?,?,?,?)",
        (key, time.time(), team, "goal", new_status),
    )
    conn.commit()
    return True


def posts_frame(conn, match_key: int) -> pd.DataFrame:
    return pd.read_sql_query(
        "SELECT ts, side, text, score FROM posts WHERE match_key=? ORDER BY ts",
        conn, params=(match_key,),
    )


def events_frame(conn, match_key: int) -> pd.DataFrame:
    return pd.read_sql_query(
        "SELECT ts, team, kind, detail FROM events WHERE match_key=? ORDER BY ts",
        conn, params=(match_key,),
    )
