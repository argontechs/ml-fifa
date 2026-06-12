# WC2026 Scoreline Predictor — task checklist

Full plan with code: `docs/superpowers/plans/2026-06-12-wc2026-score-predictor.md`
Spec: `docs/superpowers/specs/2026-06-12-wc2026-score-predictor-design.md`

- [ ] Task 1: Project skeleton (requirements, pytest.ini, src/fifa pkg)
- [ ] Task 2: Cached downloader with browser UA + stale-cache fallback
- [ ] Task 3: Results loading, NA-fixture split, successor mapping
- [ ] Task 4: FIFA→dataset team-name alias map + live completeness check
- [ ] Task 5: Elo formulas (expectancy, K tiers, margin multiplier) + published-value validation
- [ ] Task 6: Chronological Elo engine + real-data top-10 sanity
- [ ] Task 7: Leakage-safe feature builder (Elo, decayed form, rest, H2H, confederations)
- [ ] Task 8: Scoreline matrix + Dixon-Coles correction + LOCK/STRONG/LEAN/TOSS-UP tiers
- [ ] Task 9: Dixon-Coles baseline (time-decayed Poisson GLM)
- [ ] Task 10: Twin LightGBM Poisson goal model
- [ ] Task 11: Metrics (RPS/log loss/Brier), report card with honesty gates, ensemble
- [ ] Task 12: Walk-forward backtest + rho/blend tuning — ALL 5 GATES MUST PASS
- [ ] Task 13: WC2026 live fixtures client + venue-host neutrality
- [ ] Task 14: predict.py CLI (tiered scoreline output)
- [ ] Task 15: FIFA group ranking + best-thirds logic
- [ ] Task 16: Monte Carlo tournament simulation (+ bracket research step)
- [ ] Task 17: simulate.py CLI (champion odds)
- [ ] Task 18: Static dashboard + update.py daily refresh
- [ ] Task 19: Final verification, README, vault update

## Accuracy contract (agreed 2026-06-12, LOCK bar relaxed to 70% same day)
- LOCK-tier picks (model ≥70% confident): target ≥70% — measured on real backtests (expect 75–85%)
- All-match W/D/L: 53–60% honest band; exact-score ≥ always-1-1 baseline (~11%)
- exact-score ≥15% = leakage alarm, NOT success

## Feature set (user-confirmed 2026-06-12)
Strength (Elo) · last-10 form (win rate + decayed GF/GA) · momentum (Elo trend 1y/2y)
· rest days · World Cup experience · head-to-head · confederation · match importance · venue/neutrality
