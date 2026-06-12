# WC2026 Player Performance Dashboard — Design Spec

**Date:** 2026-06-12 · **Status:** approved scope (user), spec self-reviewed
**User ask:** pull player stats, visualize everything, cluster players by playing style,
show injuries/market context. Earlier decision: FBref for per-90 performance (the standard
for style analysis) + Transfermarkt for market value/injuries, via the `soccerdata` package —
not Transfermarkt-only scraping.

## 1. Goal

A **Players** page on the live site (`wc2026.argontechs.dev/players.html`) that answers:
*what kind of player is X, who are the WC2026 nations actually bringing, and who's in form?*
Core feature: **playing-style archetypes** learned by clustering standardized per-90 stat
vectors, visualized as an interactive scatter (every player a dot, colored by archetype,
hover for details), plus per-nation squad tables.

### Success criteria

| Check | Target |
|---|---|
| Coverage | ≥ 1,500 players with ≥ 900 league minutes (2025-26 Big-5 season) clustered |
| WC relevance | players filterable/grouped by WC2026 nation (nationality-based) |
| Archetypes | 8 ± 2 named outfield clusters that pass a face-validity spot check (e.g. Haaland-type ≠ Rodri-type) |
| Page | static HTML, same floodlit-pitch design, embedded interactive Plotly scatter, no server |
| Repeatability | one command (`players_build.py`) refreshes data → clusters → page → deploy |
| Honesty | coverage gaps stated on-page (players outside covered leagues aren't scored, nationality ≠ confirmed squad) |

### Non-goals (v1, recorded cuts)

- **Confirmed 26-man squads** — no reliable machine-readable source; v1 filters by FBref
  nationality + minutes. Labeled on-page.
- **Live in-tournament player stats** — v1 is the club-season profile (2025-26).
- **Injury feeds** — only if `soccerdata`'s Transfermarkt reader works out of the box at
  build time; otherwise cut with an on-page note (probe step in plan, no hard dependency).
- Goalkeepers get listed but not clustered (their stats live in a different space).

## 2. Data

| Need | Source | Notes |
|---|---|---|
| Per-90 performance | FBref "Big 5 European Leagues Combined", season 2025-26, via `soccerdata.FBref.read_player_season_stats(stat_type=…)` for `standard, shooting, passing, defense, possession, misc` | ~6 cached requests; soccerdata rate-limits politely. **Risk:** FBref anti-scraping; plan has a probe step with Understat fallback (fewer stat dims, documented). |
| Market value / injuries | `soccerdata` Transfermarkt reader **if available in installed version** | probe → adopt or cut. |
| WC nations | `fifa.dashboard.FLAGS` keys ∩ FBref nationality codes | needs a country-code → dataset-name bridge (FBref uses 3-letter codes ≈ our verified TRIGRAPH map, reused). |

## 3. Method

1. **Features (outfield):** per-90: goals, npxG, shots, assists, xA, key passes, progressive
   passes, progressive carries, dribbles completed, touches in att-3rd, tackles+interceptions,
   blocks+clearances, aerials won, fouls drawn — plus shooting distance and pass completion %.
   Minutes ≥ 900. Per-90 everything, then `StandardScaler` **within position group**
   (DF/MF/FW from FBref's position column) so "high tackles for a forward" means something.
2. **Clustering:** k-means (k chosen from silhouette over k=6..12, fixed seed) on the scaled
   matrix. Archetype names assigned by inspecting cluster centroids programmatically
   (top-3 distinguishing features) + a hand-written name table mapping centroid signatures to
   labels ("Penalty-box striker", "Progressor", "Ball-winner", "Wide creator", …).
3. **Projection:** PCA(2) for the scatter (deterministic; no UMAP dependency).
4. **Page:** Plotly scatter embedded via `to_html(include_plotlyjs="cdn")` fragment; nation
   tables (player, age, club, 90s, key stats, archetype, value/flag if available); nav gains
   a third tab. Same CSS family as the predictor pages.

## 4. Components

```
src/players/
├── fetch.py      # soccerdata wrappers + caching + the FBref→dataset nation bridge
├── features.py   # per-90 matrix, minutes filter, position-group scaling
├── cluster.py    # k-means + silhouette, centroid signatures, archetype naming
└── page.py       # players.html renderer (Plotly fragment + tables + nav)
players_build.py  # CLI: fetch → features → cluster → render → (optional) deploy
```

All fetchers injected for tests (synthetic frames); only `players_build.py` touches the network.

## 5. Self-review notes (completed)

- **Scope check:** one page + one pipeline = single plan. The riskiest external (FBref
  availability) is front-loaded as Task P1's probe with a named fallback.
- **Ambiguity fixed:** "playing style" = per-90 club-season profile, position-normalized;
  squads = nationality proxy (stated on page).
- **Consistency:** nation naming flows through the existing verified TRIGRAPH/FLAGS maps;
  no new name systems invented.
- **Placeholder scan:** archetype names come from a committed signature→name table applied
  to real centroids — if a centroid matches nothing, it gets a generated descriptive label
  (top features), never a blank.
- **YAGNI:** UMAP, live match stats, squad scraping, transfer history — all cut from v1.
