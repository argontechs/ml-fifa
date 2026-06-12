# WC2026 Live Sentiment Tracker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans. TDD per step, commit per task.

**Goal:** Three restart-safe processes (collector → scorer → Dash app) around one SQLite file: Bluesky Jetstream posts keyword-routed per active WC match, scored by multilingual XLM-RoBERTa sentiment, visualized live with goal markers.

**Architecture:** per spec `docs/superpowers/specs/2026-06-12-sentiment-tracker-design.md`. New package `src/sentiment/` (imports `fifa.*`, never the reverse). All I/O injected for offline tests; replay harness substitutes the live websocket.

**Tech stack:** torch (MPS), transformers (`cardiffnlp/twitter-xlm-roberta-base-sentiment`), dash+plotly, websockets, sqlite3 stdlib. Separate `requirements-sentiment.txt`.

---

### Task S1: Dependencies + package skeleton
- [ ] Write `requirements-sentiment.txt`: `torch`, `transformers`, `dash`, `plotly`, `websockets`
- [ ] `.venv/bin/pip install -r requirements-sentiment.txt` (background — torch is ~2.5 GB)
- [ ] Create `src/sentiment/__init__.py`; commit `chore: sentiment tracker skeleton`

### Task S2: Trigraphs + match windows (`src/sentiment/match_window.py`)
- [ ] Failing tests `tests/test_match_window.py`: trigraph completeness for all 48 teams (against `fifa.dashboard.FLAGS` keys); `active_matches` includes a fixture 10 min before kickoff and 100 min after, excludes one 4 h away and all `tbd`; `keywords_for` yields lowercase team names, aliases (from `fifa.data.FIFA_ALIASES` reverse map), and both hashtag orderings (`#mexrsa`, `#rsamex`)
- [ ] **Verify trigraphs from a cited source, not memory:** `curl -s "https://en.wikipedia.org/wiki/List_of_FIFA_country_codes"` and grep the non-obvious ones (SUI, NED, GER, CRO, KSA, CPV, CUW, COD, RSA, ALG, IRN, URU, PAR, HAI, SCO/ENG). Fix any mismatch before proceeding.
- [ ] Implement: `TRIGRAPH: dict[str, str]` (48 entries), `active_matches(fx, now) -> list[dict]` (window kickoff−30 min → kickoff+150 min, skip `status == "tbd"`), `keywords_for(match) -> dict[str, set]` with keys `home`/`away`/`both`
- [ ] Tests pass → commit

### Task S3: Storage (`src/sentiment/db.py`)
- [ ] Failing tests: schema creation idempotent; `insert_post` → `unscored_batch` returns it → `set_scores` empties the unscored set; WAL mode actually on (`PRAGMA journal_mode` returns `wal`); `record_score_change` inserts an event only when the score string differs
- [ ] Implement: `connect(path)` (WAL, busy_timeout 5000, `init_schema`), `insert_post(conn, ts, source, match_key, side, text)`, `unscored_batch(conn, limit=128) -> list[(id, text)]`, `set_scores(conn, ids, scores)`, `upsert_match(conn, key, home, away, kickoff, status)`, `record_score_change(conn, key, team, detail)`, read helpers `posts_frame(conn, key)` / `events_frame(conn, key)` returning DataFrames
- [ ] Tests pass → commit

### Task S4: Collector (`src/sentiment/collector.py`)
- [ ] Failing tests: feed a fake async source of Jetstream-shaped JSON (3 posts: one home-keyword, one away-keyword, one both/hashtag; plus one non-post commit and one malformed line) → db contains 3 rows with correct `side`; a post matching two *different* matches lands once per match; malformed input never raises
- [ ] Implement:
  - `route(text, kwsets) -> str | None` — lowercase containment; `both` if home+away or a both-hashtag hits
  - `async consume(source, conn, windows_fn, clock=time.time)` — parse `kind=="commit" / operation=="create" / collection=="app.bsky.feed.post"`, route against `windows_fn()` (list of (match, kwsets)), insert
  - `async jetstream_source(url=JETSTREAM_URL)` — websockets client, exponential backoff 1→60 s, yields raw strings
  - `windows_from_fixtures(ttl=60)` — cached active-match keyword sets, refreshed via `fifa.fixtures.load_fixtures()`
- [ ] Tests pass → commit

### Task S5: Scorer (`src/sentiment/scorer.py`)
- [ ] Failing tests: with a stub model (callable returning fixed label-prob lists) `score_texts` maps (pos .8/neu .1/neg .1)→0.7; `run_once` scores exactly the unscored batch and writes back; empty table → returns 0
- [ ] Implement: `load_model()` (transformers pipeline, `device="mps"` if `torch.backends.mps.is_available()` else CPU, `top_k=None`, truncation max_length=128); `score_texts(model, texts) -> list[float]` (P(positive)−P(negative)); `run_once(conn, model, batch=128) -> int`; `run_loop(conn, model, poll=2.0)`
- [ ] Add `@pytest.mark.live` smoke: real model on 8 multilingual sentences (es/fr/ar/en), asserts sign of obvious cases ("GOOOOAL what a strike!" > 0 > "embarrassing performance, we were robbed")
- [ ] Offline tests pass → commit (live test run once manually when model downloaded)

### Task S6: Aggregation + events (`src/sentiment/aggregate.py`, `src/sentiment/events.py`)
- [ ] Failing tests: hand-built posts across 3 minutes/2 sides → `timeline` returns per-minute buckets with correct means/counts and EWMA columns; `tallies` window math; `events.poll_once` with two fake fixture frames (0-0 → 1-0) records exactly one home-goal event
- [ ] Implement: `timeline(conn, key, bucket_s=60, halflife_buckets=3) -> DataFrame`; `tallies(conn, key, window_min=15) -> dict`; `recent_posts(conn, key, n=12)`; `events.poll_once(conn, fx_frame)` comparing `"{hs}-{as}"` against `matches.status`
- [ ] Tests pass → commit

### Task S7: CLIs + replay harness
- [ ] `sentiment_collect.py` (root): argparse `--replay FILE --speed 50`; asyncio gathers `consume(...)` + a 60 s events/fixtures poll task; replay source reads JSONL with simulated inter-post delay
- [ ] `sentiment_score.py` (root): load model, `run_loop`
- [ ] Generate `tests/replay_sample.jsonl`: 200 synthetic Jetstream messages for a fake active match (mix of home/away/both keywords, multilingual snippets, 5% junk lines) via a committed `tests/make_replay_sample.py`
- [ ] Integration test (offline): replay 200 posts through consume with stub clock + stub scorer → timeline non-empty, sides populated
- [ ] Commit

### Task S8: Dash app (`sentiment_app.py`)
- [ ] Failing tests (pure functions only): `figure_for(timeline_df, events_df)` returns a plotly Figure with 2 sentiment traces + volume bars + one vline per goal; `match_options(fx)` lists active matches first
- [ ] Implement Dash app: dropdown, `dcc.Interval(3000)`, timeline graph (home line volt #b8f53d, away line #f0573f, neutral grey dashed, volume bars subdued, goal vlines + ⚽ annotations), tally cards (mood now / posts collected / scoring lag), recent-posts ticker; floodlit-pitch dark CSS matching the site; honesty footnote ("mention mood ≠ verified fan allegiance")
- [ ] Manual smoke: `python sentiment_collect.py --replay tests/replay_sample.jsonl --speed 100` + stub-scored db + `python sentiment_app.py` → screenshot via Playwright, verify visually
- [ ] Commit

### Task S9: Final verification + docs
- [ ] Full `pytest -q` green (live-marked tests excluded by default)
- [ ] Real-model smoke: run `@pytest.mark.live` scorer test once
- [ ] E2E dress rehearsal: replay → real scorer → Dash, screenshot
- [ ] README section "Sentiment tracker" (3 commands + match-day runbook); vault hub update; commit

## Self-review notes (completed)
- **Spec coverage:** sources S4 (+Reddit deferred: spec marks it optional-on-creds; v1 ships Bluesky-only with the `source` column already in schema — recorded as a conscious cut, not an omission); storage S3; scorer S5; aggregation/events S6; replay S7; Dash S8; success criteria exercised in S8/S9.
- **Type consistency:** `windows_fn() -> list[tuple[match_dict, kwsets_dict]]` used by S4 consume and S7 replay alike; `match_key` is the fixture `match_number` int everywhere; score ∈ [−1,1] floats.
- **No placeholders:** trigraphs get a verification step; replay fixture is generated by committed code.
- **Risk called out:** torch install size and model download are S1/S5 prerequisites — both background-able.
