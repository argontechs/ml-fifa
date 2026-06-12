# WC2026 Player Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans. TDD per task, commit per task.
> Spec: `docs/superpowers/specs/2026-06-12-player-dashboard-design.md`

**Goal:** `players.html` on the live site: per-90 style archetypes (k-means over position-normalized
Big-5 stats), interactive Plotly scatter, WC-nation squad tables. One rebuild command.

**Canonical feature names** (single source of truth, used by features/cluster/page and all tests):
`g90, npxg90, sh90, ast90, xa90, kp90, prgp90, prgc90, drb90, att3rd90, tklint90, blkclr90,
aer90, fld90, dist, passpct` + identity columns `player, nation, pos, club, age, minutes`.

### Task P1: Probe the data sources (decision point — do FIRST, record verdicts here)
- [ ] `.venv/bin/pip install soccerdata`
- [ ] Probe FBref: `soccerdata.FBref(leagues="Big 5 European Leagues Combined", seasons="2025-2026")`,
  `read_player_season_stats(stat_type="standard")` — print shape + column tuples. If blocked/empty →
  fallback Understat (top-5, fewer dims: drop `kp90, blkclr90, aer90, fld90, dist, passpct` from
  canonical set) and record the cut here.
- [ ] Probe Transfermarkt reader: `hasattr(soccerdata, "Transfermarkt")` (or equivalent in installed
  version) → if absent, market value/injuries are CUT (on-page note), per spec.
- [ ] Record verdicts in this file; commit.

### Task P2: `src/players/fetch.py` — readers injected, FBref columns pinned from probe output
- [ ] Failing tests (synthetic frames with REAL FBref column tuples captured in P1):
  `assemble(readers)` returns one row per player with all canonical columns; nation bridge maps
  FBref 3-letter codes via the verified `sentiment.match_window.TRIGRAPH` reverse map
  (e.g. `ESP → Spain`), unknown codes → `None` (kept, just not WC-taggable)
- [ ] Implement `assemble(readers: dict[str, Callable]) -> pd.DataFrame`: join the 6 stat frames on
  player id, flatten MultiIndex per the P1-pinned `COLUMN_MAP`, compute per-90s from totals + 90s,
  derive `wc_nation`; `default_readers()` builds the real soccerdata closures (network only here)
- [ ] Commit

### Task P3: `src/players/features.py`
- [ ] Failing tests: minutes filter (≥900); GKs excluded from the matrix but returned in `listed`;
  scaling is position-grouped (construct two positions with different means → both scale to ~0 mean
  within group); output matrix columns == canonical stat list; NaNs imputed to group median
- [ ] Implement `build_matrix(df) -> (X_scaled: DataFrame, meta: DataFrame)` with
  `StandardScaler` per `pos_group ∈ {DF, MF, FW}` (FBref pos string → first listed group;
  GK rows → meta only)
- [ ] Commit

### Task P4: `src/players/cluster.py`
- [ ] Failing tests: on synthetic blobs (3 known styles) k-means recovers 3 clusters at the
  silhouette-chosen k; `centroid_signature` returns the top-3 distinguishing features; naming:
  a signature dominated by `(g90, npxg90, sh90)` maps to "Penalty-box striker", unknown
  signatures get a generated label like "High prgp90 · drb90 profile" (never blank); seeded determinism
- [ ] Implement `fit(X, k_range=range(6,13), seed=42)` (silhouette pick),
  `centroid_signature(centers, cols)`, `ARCHETYPES` signature table (≥8 entries:
  Penalty-box striker / Complete forward / Wide creator / Advanced playmaker / Progressor /
  Ball-winner / Stopper / Ball-playing defender / Workhorse), `name_clusters(...)`,
  `project2d(X)` (PCA, seeded)
- [ ] Commit

### Task P5: `src/players/page.py` + third nav tab
- [ ] Failing tests: render contains scatter div, archetype legend names, a nation table with a
  player row + flag img, the coverage honesty note; `fifa/dashboard.py` `_nav` now emits three
  tabs (update its tests)
- [ ] Implement `render(meta, labels, xy, archetype_names, generated_at) -> str` —
  Plotly `Scatter` fragment via `fig.to_html(full_html=False, include_plotlyjs="cdn")`,
  hover = player/club/archetype/key stats; nation sections sorted by sim champion odds when
  `data/sim_results.json` exists; reuse `fifa.dashboard.FLAGS`/CSS palette. Add
  `Players` to `_nav` in `fifa/dashboard.py` (href `players.html`)
- [ ] Commit

### Task P6: `players_build.py` + real run + ship
- [ ] CLI: fetch (cached) → features → cluster → render → write `dashboard/players.html` →
  optional `--deploy` (same wrangler call as update.py)
- [ ] Real run; spot-check face validity (Haaland-type vs Rodri-type in different clusters);
  Playwright screenshot; deploy; README section; vault hub update; commit

## Self-review notes (completed)
- **Spec coverage:** data P1/P2, method P3/P4, page P5, repeatability P6; honesty notes land in P5's
  template; coverage target asserted in P6's real-run check.
- **Type consistency:** canonical names declared once at the top; every module/test imports the same
  list from `features.CANONICAL`.
- **Risk routing:** the two externals (FBref, Transfermarkt) both resolve in P1 before any dependent
  code is written — no mid-build surprises.
- **No placeholders:** FBref column tuples get pinned from real probe output, mirroring the
  fixtures-feed and trigraph precedents.
