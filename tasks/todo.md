# WC2026 Scoreline Predictor — task checklist

Full plan with code: `docs/superpowers/plans/2026-06-12-wc2026-score-predictor.md`
Spec: `docs/superpowers/specs/2026-06-12-wc2026-score-predictor-design.md`

- [x] Task 1: Project skeleton (requirements, pytest.ini, src/fifa pkg)
- [x] Task 2: Cached downloader with browser UA + stale-cache fallback
- [x] Task 3: Results loading, NA-fixture split, successor mapping
- [x] Task 4: FIFA→dataset team-name alias map + live completeness check
- [x] Task 5: Elo formulas (expectancy, K tiers, margin multiplier) + published-value validation
- [x] Task 6: Chronological Elo engine + real-data top-10 sanity
- [x] Task 7: Leakage-safe feature builder (Elo, decayed form, rest, H2H, confederations)
- [x] Task 8: Scoreline matrix + Dixon-Coles correction + LOCK/STRONG/LEAN/TOSS-UP tiers
- [x] Task 9: Dixon-Coles baseline (time-decayed Poisson GLM)
- [x] Task 10: Twin LightGBM Poisson goal model
- [x] Task 11: Metrics (RPS/log loss/Brier), report card with honesty gates, ensemble
- [x] Task 12: Walk-forward backtest + rho/blend tuning — ALL 5 GATES MUST PASS
- [x] Task 13: WC2026 live fixtures client + venue-host neutrality
- [x] Task 14: predict.py CLI (tiered scoreline output)
- [x] Task 15: FIFA group ranking + best-thirds logic
- [x] Task 16: Monte Carlo tournament simulation (+ bracket research step)
- [x] Task 17: simulate.py CLI (champion odds)
- [x] Task 18: Static dashboard + update.py daily refresh
- [x] Task 19: Final verification, README, vault update
- [x] Task 20: Intercontinental displacement feature
- [x] Task 21: Shootout-history-informed penalty resolution in sim
- [x] Task 22: Isotonic W/D/L calibration (protects the ≥70% LOCK contract)
- [x] Task 23: Bookmaker odds blend (needs ODDS_API_KEY in env or data/odds_api_key.txt; gracefully model-only without it)

## Accuracy contract (agreed 2026-06-12, LOCK bar relaxed to 70% same day)
- LOCK-tier picks (model ≥70% confident): target ≥70% — measured on real backtests (expect 75–85%)
- All-match W/D/L: 53–60% honest band; exact-score ≥ always-1-1 baseline (~11%)
- exact-score ≥15% = leakage alarm, NOT success

## Feature set (user-confirmed 2026-06-12)
Strength (Elo) · last-10 form (win rate + decayed GF/GA) · momentum (Elo trend 1y/2y)
· rest days · World Cup experience · head-to-head · confederation · match importance · venue/neutrality
