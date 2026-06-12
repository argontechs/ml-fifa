# WC2026 Live Sentiment Tracker — Design Spec

**Date:** 2026-06-12 · **Status:** approved scope (user greenlit 2026-06-12), spec self-reviewed
**Decision context:** user proposed Tweepy + HF transformers + Plotly Dash. X/Twitter API rejected
(free tier is write-only; meaningful streaming starts at $5,000/mo). Approved replacement keeps the
transformer + Dash core and swaps the data source for free real-time firehoses.

## 1. Goal

During live World Cup 2026 matches, continuously collect public posts about the match, score
their sentiment with a multilingual transformer, and visualize crowd mood shifting in real time —
with goal events marked on the timeline so opinion swings line up with what happened on the pitch.

### Success criteria

| Check | Target |
|---|---|
| Collection during a live match | > 0 match-tagged posts/min sustained, no crash over 3h |
| Scoring lag | a post is sentiment-scored < 10 s after collection |
| Dashboard freshness | charts update every ~3 s without manual reload |
| Goal markers | appear within ~2 min of a real goal (fixtures-feed latency) |
| Restart safety | any of the three processes can die and be restarted without data loss |
| Offline developability | full pipeline runs against a recorded replay file — no live match needed |

### Non-goals

- X/Twitter (paywalled), historic backfill, precise fan-allegiance classification.
- Publishing the live dashboard to argontechs.dev (v2 option: periodic static snapshot pushed to
  Pages next to the predictor; the live Dash app is localhost-only in v1).
- Reusing the predictor's model — this subsystem only *reads* `fixtures.py` and the alias map.

## 2. Data sources (all free)

| Source | Transport | Auth | Notes |
|---|---|---|---|
| **Bluesky Jetstream** (primary) | `wss://jetstream2.us-east.bsky.network/subscribe?wantedCollections=app.bsky.feed.post` | none | Full-network post firehose as JSON; we keyword-filter client-side. Reconnect with cursor on drop. |
| **Reddit match threads** (optional) | PRAW streaming comments from r/soccer match threads | free script app — **requires user-created credentials** in `data/reddit_creds.json` | Enabled automatically when the creds file exists; absent → Bluesky-only, no errors. |
| Goal events | existing `fifa.fixtures.load_fixtures()` (live scores, 60 s poll) | none | Score change on an active match = goal marker. |

## 3. Architecture — three restart-safe processes around one SQLite file

```
collector.py ──┐                         ┌── sentiment_app.py (Dash, :8050)
 (Jetstream ws │   data/sentiment.db     │     reads aggregates every 3s
  + Reddit opt)├──► WAL mode, 3 tables ◄─┤
scorer.py ─────┘   posts / matches /     │
 (XLM-RoBERTa,      events               │
  batch loop)                            │
```

- **`src/sentiment/match_window.py`** — decides which matches are "active" (kickoff −30 min →
  +150 min or FT+30) from the fixtures feed, and builds each match's keyword set: both team
  names + dataset aliases + FIFA trigraph hashtag (`#MEXRSA` style) from a 48-team trigraph map.
- **`src/sentiment/collector.py`** — async websocket consumer; lowercase keyword match against
  active-match keyword sets; INSERT raw posts `(ts, source, match_key, text)`. The websocket
  client is injected (an async iterator), so tests and the replay harness feed it recorded JSON.
- **`src/sentiment/scorer.py`** — polling loop: SELECT unscored batch (≤128), run
  `cardiffnlp/twitter-xlm-roberta-base-sentiment` (multilingual — WC crowds post in es/fr/ar/ja),
  score = P(pos) − P(neg) ∈ [−1, 1], UPDATE rows. Device: Apple MPS, CPU fallback. The model
  handle is injected for tests (stub returns fixed scores).
- **`src/sentiment/aggregate.py`** — pure SQL/pandas: per-minute buckets of mean sentiment and
  volume per match side (home-mention / away-mention / both), EWMA smoothing, latest sample posts.
- **`src/sentiment/events.py`** — polls fixtures feed; score deltas on active matches → `events`
  rows (minute, team, new score).
- **`sentiment_app.py`** — Dash: match selector dropdown (active first), sentiment timeline
  (two team lines + volume bars + goal annotations), live tally cards, recent-posts ticker.
  Dark floodlit-pitch styling consistent with the predictor site. `dcc.Interval` 3 s.
- **`sentiment_replay.py`** — replays a recorded JSONL capture through collector+scorer at ×N
  speed for development and demos.

### Per-team attribution (honesty note)

A post mentioning only one team's keywords is attributed to that team's bucket; posts mentioning
both go to the `both` bucket (shown as the neutral "match mood" line). This measures *crowd mood
around team mentions*, not verified fan allegiance — the dashboard labels it accordingly.

## 4. Storage schema (`data/sentiment.db`, WAL)

```sql
posts(id INTEGER PK, ts REAL, source TEXT, match_key TEXT, side TEXT,  -- home|away|both
      text TEXT, score REAL NULL)                                     -- NULL = unscored
matches(match_key TEXT PK, home TEXT, away TEXT, kickoff TEXT, status TEXT)
events(id INTEGER PK, match_key TEXT, ts REAL, team TEXT, kind TEXT, detail TEXT)
```

`match_key` = fixture `match_number`. Indexes on `posts(match_key, ts)` and `posts(score) WHERE score IS NULL`.

## 5. Dependencies & footprint

New: `torch`, `transformers`, `dash`, `plotly`, `websockets`, `praw` (optional import) — in a
separate `requirements-sentiment.txt` (torch ≈ 2.5 GB; disk has 38 GB free). Model download
≈ 1.1 GB one-time into the HF cache. M4/16 GB runs the model at hundreds of posts/sec batched;
expected live load is < 500 posts/min — two orders of magnitude of headroom.

## 6. Error handling

- Websocket drop → exponential backoff reconnect (1→60 s), cursor resume; logged, never fatal.
- Model load failure → scorer exits with a clear message; collector keeps collecting (scores
  backfill when scorer returns — restart safety).
- No active match → collector idles politely (no subscriptions), dashboard shows next kickoff.
- SQLite contention → WAL mode + busy_timeout 5000 ms; each process owns its own connection.

## 7. Testing strategy

All offline: collector fed synthetic async iterators (keyword routing, side attribution,
reconnect logic); scorer with stubbed model (batching, NULL→scored transitions); aggregate with
hand-built rows (bucket math, EWMA); events with fake fixture frames (goal delta detection);
trigraph map completeness against the 48 teams. One `@pytest.mark.live` smoke (real model on 8
sentences; skipped in default runs). End-to-end via the replay harness on a committed 200-post
fixture capture.

## 8. Self-review notes (completed)

- **Ambiguity fixed:** "active match" window pinned to kickoff−30 → kickoff+150 or FT+30.
- **Scope check:** three small processes + one Dash app — single plan, ~2–3 days as estimated.
- **Consistency:** alias/trigraph maps reuse `fifa.data` naming; no circular imports
  (sentiment package imports fifa, never the reverse).
- **Placeholder scan:** trigraph map values must be FIFA's actual codes (RSA not SAF, etc.) —
  plan includes a verification step against a cited list rather than guessing.
- **Risk:** Bluesky volume for non-English fanbases may skew low; Reddit option mitigates;
  the volume chart makes thin data visible rather than hiding it.
