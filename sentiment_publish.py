"""Publish sentiment.html to the live site — once, or as a near-live loop during matches.

usage:
  .venv/bin/python sentiment_publish.py                 # render + deploy once
  .venv/bin/python sentiment_publish.py --loop 180      # every 3 min (match nights)
"""
import argparse
import datetime
import time
from pathlib import Path

from fifa import data
from fifa.deploy import deploy_dashboard
from sentiment import db, publish


def render_once() -> Path:
    conn = db.connect(data.DATA_DIR / "sentiment.db")
    try:
        gen = (datetime.datetime.now(datetime.UTC)
               + datetime.timedelta(hours=8)).strftime("%Y-%m-%d %H:%M MYT")
        out = Path(__file__).resolve().parent / "dashboard" / "sentiment.html"
        out.write_text(publish.render(conn, gen))
        return out
    finally:
        conn.close()


def fingerprint() -> tuple:
    """Change detector: new posts, newly-scored posts, OR new goal events."""
    conn = db.connect(data.DATA_DIR / "sentiment.db")
    try:
        row = conn.execute(
            "SELECT (SELECT COALESCE(MAX(id),0) FROM posts), "
            "(SELECT COUNT(*) FROM posts WHERE score IS NOT NULL), "
            "(SELECT COALESCE(MAX(id),0) FROM events)"
        ).fetchone()
        return tuple(row)
    finally:
        conn.close()


ap = argparse.ArgumentParser()
ap.add_argument("--loop", type=int, default=0, metavar="SECONDS")
ap.add_argument("--no-deploy", action="store_true")
args = ap.parse_args()

last_seen = None
while True:
    newest = fingerprint()
    if newest != last_seen:  # only render+deploy when there's actually new data
        out = render_once()
        ok = (not args.no_deploy) and deploy_dashboard()
        print(f"{datetime.datetime.now():%H:%M:%S} rendered {out.name}"
              + (" + deployed" if ok else ""), flush=True)
        last_seen = newest
    if not args.loop:
        break
    time.sleep(args.loop)
