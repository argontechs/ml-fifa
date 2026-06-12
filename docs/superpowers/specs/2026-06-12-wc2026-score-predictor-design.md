# WC2026 Scoreline Predictor — Design Spec

**Date:** 2026-06-12 · **Status:** awaiting user review
**Scope decision:** per-match scoreline predictor + Monte Carlo tournament simulation
**Interface decision:** CLI scripts + statically generated web dashboard

## 1. Goal

Predict FIFA World Cup 2026 match scorelines and outcome probabilities, trained on the full history of men's international football (~49,400 played matches, 1872 → yesterday), with an Elo rating engine as the backbone feature. Output per fixture: most likely score, top-5 scorelines with probabilities, and Win/Draw/Loss probabilities. Output per tournament: group advancement odds and champion probabilities from Monte Carlo simulation. The tournament is live (June 11 – July 19, 2026), so the system must refresh daily as results land.

### Success criteria (evidence-based, from verified benchmarks)

| Metric | Target | Context |
|---|---|---|
| Exact-score hit rate (test) | ≥ 11% (beat always-1-1 baseline) | Best published models: ~9–12%. **>15% ⇒ suspect leakage, investigate** |
| W/D/L accuracy (test) | 53–59% | Published band for internationals on this dataset |
| Ranked Probability Score | ≤ ~0.205 | 2017 SPC winners: 0.2054; bookmakers ≈ 0.198 (ceiling) |
| Draw recall | Materially > 0 | Reference LightGBM classifier predicted 2 of 1,784 draws — our architecture must not collapse on draws |
| Elo engine correctness | Reproduce published eloratings.net per-match changes | e.g. Argentina −6 in the 2022 WC final |
| LOCK-tier accuracy (picks with ≥70% outcome confidence) | ≥ 70% | User contract (relaxed from 80% on 2026-06-12); calibrated models historically land 75–85% on this subset |

### Non-goals

- Beating bookmaker odds (no published model does consistently).
- Player-level features (injuries, xG, lineups) — no usable historical data for internationals; v2 candidate: squad market values (Transfermarkt).
- Women's football, club football, real-time in-match prediction.

## 2. Data sources (all verified reachable 2026-06-12, no accounts/keys)

| Source | URL | Notes |
|---|---|---|
| Match history | `https://raw.githubusercontent.com/martj42/international_results/master/results.csv` | CC0, updated daily. 49,477 rows = 49,405 played + 72 future WC2026 group fixtures with literal `NA` scores — **filter before casting to int**. Schema: `date, home_team, away_team, home_score, away_score, tournament, city, country, neutral`. Scores include extra time, exclude shootouts. |
| Shootouts | `.../master/shootouts.csv` | 678 rows; winner per shootout. Elo treats shootout matches as draws (W=0.5). |
| Name history | `.../master/former_names.csv` | USSR→Russia, West Germany→Germany already merged in results.csv. **We must add**: Yugoslavia→Serbia, Czechoslovakia→Czech Republic successor chains; German DR terminates. |
| WC2026 fixtures + live scores | `https://fixturedownload.com/feed/json/fifa-world-cup-2026` | All 104 matches, live-updated scores, group/round/venue. **Requires browser User-Agent (403 otherwise).** CSV fallback exists. |
| Bookmaker consensus odds (optional) | `api.the-odds-api.com/v4/sports/soccer_fifa_world_cup/odds` | Free tier (500 credits/mo), key via `ODDS_API_KEY` env or `data/odds_api_key.txt`. Live predictions only (no free historical odds → backtest stays model-pure). Absent key → model-only, never breaks. |

**Team-name alias map (hard requirement):** the fixtures feed uses FIFA names ("Korea Republic", "Czechia", "IR Iran", "Côte d'Ivoire", "Cabo Verde", "Türkiye", "Congo DR"); martj42 uses common names ("South Korea", "Czech Republic", "Iran", "Ivory Coast", "Cape Verde", "Turkey", "DR Congo"). Unmapped names are a **hard error** listing the offenders — never silent fuzzy matching.

## 3. Architecture

Pipeline: **data → Elo engine → feature builder → goal-rate models → score matrix → (a) per-match predictions (b) Monte Carlo sim → CLI + dashboard.**

### 3.1 Model choice (decided)

Goal-rate regression, not outcome classification. Two LightGBM regressors with `objective='poisson'` predict λ_home and λ_away (expected goals). The two Poisson distributions form an 11×11 scoreline probability matrix (0–10 goals each), corrected for low-score/draw dependence with the Dixon-Coles τ adjustment (ρ fit on training data; fallback: FiveThirtyEight-style ~9% diagonal inflation). Everything reads off the matrix: argmax cell = most likely score; lower triangle/diagonal/upper triangle sums = W/D/L probabilities.

Two models, ensembled by validation-weighted probability averaging:
1. **Baseline:** classic time-decayed Dixon-Coles MLE (attack/defense per team + home advantage + ρ). Interpretable, strong; same family as the academic WC2026 forecast.
2. **Main:** twin LightGBM Poisson regressors on engineered features.

Rejected: logistic regression / RF / NN / multiclass scoreline classifiers — outcome classifiers can't emit scorelines and demonstrably collapse on draws; NNs offer no edge on 49k tabular rows (Fischer & Heuer 2024).

Two post-processing layers on the blended matrix (added 2026-06-12, plan Tasks 22–23): (1) **isotonic W/D/L calibration** fitted on validation so stated confidence matches realized frequency — the LOCK contract's guarantee; (2) optional **bookmaker odds blend** (market weight 0.6) on live predictions, applied by rescaling the matrix's win/draw/loss regions to the blended targets. The knockout simulation additionally resolves penalty shootouts with smoothed historical shootout records (Task 21) instead of a pure Elo coin flip.

### 3.2 Elo engine (exact eloratings.net formulas, verified)

- Expectancy: `We = 1 / (1 + 10^(−dr/400))`, `dr = R_home − R_away + (100 if not neutral else 0)`.
- Update: `ΔR = K · G · (W − We)`, zero-sum (home gains what away loses).
- K by tournament tier: World Cup finals 60; continental finals (Euro, Copa América, AFCON, Asian Cup, Gold Cup, Confederations Cup, Finalissima) 50; qualifiers + Nations Leagues 40; minor named cups 30; friendlies 20. Mapping table from martj42 `tournament` strings, with explicit default + log for unmapped tournaments.
- Margin multiplier G: ≤1 goal → 1.0; 2 → 1.5; 3 → 1.75; N≥4 → 1.75 + (N−3)/8.
- W: 1 / 0.5 / 0. Shootout after draw stays W=0.5.
- Init: uniform 1500, full-history burn-in from 1872 (self-correcting; only 4 teams pre-1900). Teams with <30 matches flagged provisional.
- Stored output: **pre-match** Elo for both teams on every row (the post-match rating is the classic leak).
- Validation: reproduce known per-match changes from eloratings.net team TSVs (Scotland −3 in 1872; Argentina −6 in the 2022 final) within ±1 point.

### 3.3 Feature builder (strictly pre-match)

Per match row, computed from prior rows only: Elo_home, Elo_away, Elo_diff (with home-advantage term — "how strong the team is"), time-decayed rolling goals for/against (decay ξ = 0.001/day), win rate over last 10 matches, rest days since last match, **Elo momentum** (rating change over the trailing 1 and 2 years — is the team trending up or down), **World Cup experience** (career WC-finals matches played before kickoff), tournament K-tier, neutral flag, WC2026-host flag, confederation of each side, head-to-head aggregate (decayed), matches played (provisional-team signal). ~24 features total. **Leakage guards are unit-tested:** the builder must produce identical features whether or not the target row's result is present in the input.

### 3.4 Validation protocol

Chronological walk-forward only — never random splits. Train ≤2021 → validate 2022–2023 (includes WC2022 backtest) → test 2024 – 2026-06-10. Metrics: RPS, multiclass log loss, Brier, exact-score hit rate vs always-1-1 baseline, calibration plots. Final production model retrains on all data through yesterday. Bonus backtest: the 2 WC2026 matches already played (Mexico 2-0 South Africa, South Korea 2-1 Czechia).

### 3.5 Tournament simulation

Monte Carlo, default 10,000 runs (`--runs 100000` flag). Each run samples every remaining group match from its score matrix; completed matches use real results from the live feed. Group ranking implements FIFA tiebreakers (points, GD, GF, H2H mini-table, fair-play unavailable → random lot). Round of 32 = 12 group winners + 12 runners-up + 8 best thirds, slotted per the official FIFA bracket mapping from the fixtures feed. Knockouts: sample from matrix; if draw, extra time sampled from a scaled (×⅓) matrix, then penalties as an Elo-weighted coin flip. Outputs: per-team advance/quarterfinal/semifinal/final/champion probabilities.

### 3.6 Components

```
ML-FIFA/
├── requirements.txt / .venv          # pandas numpy scipy scikit-learn lightgbm requests pytest
├── src/fifa/
│   ├── data.py        # download w/ caching, NA-fixture filtering, successor mapping, alias map
│   ├── elo.py         # chronological Elo engine → per-match pre-Elo columns
│   ├── features.py    # pre-match feature matrix
│   ├── dixon_coles.py # baseline model (MLE + time decay + ρ)
│   ├── gbm.py         # twin LightGBM Poisson regressors
│   ├── matrix.py      # λs → DC-corrected score matrix → scorelines/WDL
│   ├── ensemble.py    # validation-weighted blend
│   ├── evaluate.py    # walk-forward harness, RPS/logloss/Brier/calibration
│   ├── fixtures.py    # fixturedownload feed client (browser UA, cache fallback)
│   └── tournament.py  # Monte Carlo sim, tiebreakers, bracket logic
├── cli: predict.py · simulate.py · backtest.py · update.py
├── dashboard/         # update.py regenerates static index.html (no server needed)
│                      # shows: next 7 days of fixtures w/ top-5 scorelines + W/D/L bars,
│                      # current group tables w/ advance odds, champion-odds leaderboard,
│                      # model report card (backtest RPS / accuracy / calibration)
├── tests/             # Elo published-value reproduction, leakage guard, matrix sums to 1, alias completeness (all 48 WC teams map)
├── data/              # downloaded CSVs + feed cache (gitignored)
└── docs/superpowers/{specs,plans}/
```

Each module is import-only (no side effects); CLI scripts compose them. `update.py` = re-download → recompute Elo → retrain (seconds) → regenerate dashboard.

### 3.7 Error handling

- Network failure → use cached copy, print its age prominently; hard fail if no cache.
- Unmapped team name (alias or tournament tier) → hard error with the exact missing names.
- Feed 403 → browser User-Agent header set by default; clear message if it still fails.
- `NA` scores → filtered into a separate "upcoming" frame, never coerced.

### 3.8 Environment risk

Python 3.14.3 (Homebrew) is current; if LightGBM has no cp314 wheel, fall back in order: XGBoost (`objective='count:poisson'`) → `uv`-managed Python 3.12. The design is GBM-implementation-agnostic.

## 4. Honest framing (product requirement)

The dashboard and CLI must display W/D/L probabilities and top-5 scorelines with their probabilities — never a bare scoreline. A modal scoreline at ~10% is the realistic best case; the viral "AI predicted Mexico 2-0" was a modal pick (vs a 9-man side) landing, and this tool should be honest where that genre is not.
