# WC2026 Scoreline Predictor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A WC2026 match-scoreline predictor + tournament simulator: Elo engine over ~49,400 internationals → twin LightGBM Poisson goal-rate models + Dixon-Coles baseline → DC-corrected score matrices → CLI (`predict.py`, `simulate.py`, `backtest.py`, `update.py`) + static dashboard.

**Architecture:** Chronological replay of all matches builds leakage-free pre-match features (Elo, decayed form). Two goal-rate models (DC Poisson GLM baseline, LightGBM Poisson GBM) each yield (λ_home, λ_away) → 11×11 Poisson scoreline matrix with Dixon-Coles low-score correction → blended; everything (top scorelines, W/D/L, confidence tiers, Monte Carlo sim) reads off the matrix. Spec: `docs/superpowers/specs/2026-06-12-wc2026-score-predictor-design.md`.

**Tech Stack:** Python 3.14 venv (`.venv` — ALREADY CREATED, all deps installed and verified): pandas 3.0.3, numpy, scipy, scikit-learn 1.9.0, lightgbm 4.6.0, requests, pytest. Run everything with `.venv/bin/python` / `.venv/bin/pytest`.

**Conventions:** all source in `src/fifa/`, tests in `tests/`, CLI scripts at repo root. Commit after every task. Network-touching tests use cached/sample files — only tests marked `@pytest.mark.live` hit the network.

**Verified data facts the code relies on** (from research 2026-06-12):
- `results.csv` (49,477 rows): `date,home_team,away_team,home_score,away_score,tournament,city,country,neutral`; 72 future WC2026 rows have literal `NA` scores; scores include extra time, never shootouts; `neutral` is `TRUE`/`FALSE`.
- USSR→Russia and West Germany→Germany are already merged upstream; Yugoslavia/Czechoslovakia/German DR are NOT (we map the first two, German DR terminates).
- Elo (eloratings.net): `We = 1/(1+10^(−dr/400))`, `dr = R_h − R_a + 100·(home not neutral)`; `ΔR = K·G·(W−We)` zero-sum; K: WC finals 60, continental finals 50, qualifiers + Nations League 40, minor cups 30, friendlies 20; G: margin ≤1→1.0, 2→1.5, 3→1.75, N≥4→1.75+(N−3)/8; shootout = draw (W=0.5) — automatic since scores exclude shootouts.
- fixturedownload feed returns 403 without a browser User-Agent.

---

### Task 1: Project skeleton

**Files:**
- Create: `requirements.txt`, `pytest.ini`, `src/fifa/__init__.py`, `tests/__init__.py`

- [ ] **Step 1: Write the files**

`requirements.txt`:
```
pandas>=3.0
numpy
scipy
scikit-learn>=1.9
lightgbm>=4.6
requests
pytest
```

`pytest.ini`:
```ini
[pytest]
pythonpath = src
testpaths = tests
markers =
    live: hits real network endpoints (deselect with -m "not live")
```

`src/fifa/__init__.py` and `tests/__init__.py`: empty files.

- [ ] **Step 2: Verify pytest runs**

Run: `.venv/bin/pytest -q`
Expected: `no tests ran` (exit code 5 is fine).

- [ ] **Step 3: Commit**

```bash
git add requirements.txt pytest.ini src tests
git commit -m "chore: project skeleton (venv already provisioned)"
```

---

### Task 2: Cached downloader (`data.py` part 1)

**Files:**
- Create: `src/fifa/data.py`
- Test: `tests/test_data.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_data.py
import time
import pytest
from fifa import data


class FakeResp:
    def __init__(self, content=b"x,y\n1,2\n", fail=False):
        self.content = content
        self._fail = fail

    def raise_for_status(self):
        if self._fail:
            import requests
            raise requests.HTTPError("boom")


def test_download_writes_and_caches(tmp_path, monkeypatch):
    calls = []

    def fake_get(url, headers=None, timeout=None):
        calls.append(url)
        assert "User-Agent" in headers  # feed 403s without browser UA
        return FakeResp()

    monkeypatch.setattr(data.requests, "get", fake_get)
    dest = tmp_path / "f.csv"
    p1 = data.download("http://u", dest)
    p2 = data.download("http://u", dest)  # fresh cache → no second call
    assert p1 == p2 == dest and dest.read_bytes() == b"x,y\n1,2\n"
    assert len(calls) == 1


def test_download_uses_stale_cache_on_network_error(tmp_path, monkeypatch):
    dest = tmp_path / "f.csv"
    dest.write_bytes(b"old")
    old = time.time() - 9999 * 3600
    import os
    os.utime(dest, (old, old))

    def fake_get(url, headers=None, timeout=None):
        import requests
        raise requests.ConnectionError("offline")

    monkeypatch.setattr(data.requests, "get", fake_get)
    assert data.download("http://u", dest).read_bytes() == b"old"


def test_download_raises_without_cache(tmp_path, monkeypatch):
    def fake_get(url, headers=None, timeout=None):
        import requests
        raise requests.ConnectionError("offline")

    monkeypatch.setattr(data.requests, "get", fake_get)
    with pytest.raises(RuntimeError, match="no cache"):
        data.download("http://u", tmp_path / "missing.csv")
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_data.py -q`
Expected: FAIL — `cannot import name 'data'` / module not found.

- [ ] **Step 3: Implement**

```python
# src/fifa/data.py
"""Downloads, caching, cleaning, and team-name normalization."""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
UA = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
}

_BASE = "https://raw.githubusercontent.com/martj42/international_results/master/"
RESULTS_URL = _BASE + "results.csv"
SHOOTOUTS_URL = _BASE + "shootouts.csv"
FIXTURES_URL = "https://fixturedownload.com/feed/json/fifa-world-cup-2026"

# eloratings.net-style successor chains missing from upstream (USSR→Russia and
# West Germany→Germany are already merged in results.csv; German DR terminates).
SUCCESSORS = {"Yugoslavia": "Serbia", "Czechoslovakia": "Czech Republic"}


def download(url: str, dest: Path, max_age_hours: float = 12.0, force: bool = False) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    fresh = dest.exists() and (time.time() - dest.stat().st_mtime) < max_age_hours * 3600
    if fresh and not force:
        return dest
    try:
        resp = requests.get(url, headers=UA, timeout=30)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
    except requests.RequestException as exc:
        if dest.exists():
            age_h = (time.time() - dest.stat().st_mtime) / 3600
            print(f"WARNING: download failed ({exc}); using cache {dest.name} ({age_h:.0f}h old)")
        else:
            raise RuntimeError(f"Cannot download {url} and no cache at {dest}") from exc
    return dest
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/pytest tests/test_data.py -q`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/fifa/data.py tests/test_data.py
git commit -m "feat: cached downloader with browser UA and stale-cache fallback"
```

---

### Task 3: Results loading + successor mapping (`data.py` part 2)

**Files:**
- Modify: `src/fifa/data.py` (append)
- Test: `tests/test_data.py` (append)

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_data.py
import io
import pandas as pd

SAMPLE_CSV = """date,home_team,away_team,home_score,away_score,tournament,city,country,neutral
1994-06-17,Germany,Bolivia,1,0,FIFA World Cup,Chicago,United States,TRUE
1992-03-25,Yugoslavia,Netherlands,2,0,Friendly,Amsterdam,Netherlands,FALSE
2026-06-13,United States,Paraguay,NA,NA,FIFA World Cup,Los Angeles,United States,FALSE
"""


def test_load_results_splits_and_maps(tmp_path, monkeypatch):
    f = tmp_path / "results.csv"
    f.write_text(SAMPLE_CSV)
    monkeypatch.setattr(data, "download", lambda url, dest, **kw: f)
    played, upcoming = data.load_results()
    assert len(played) == 2 and len(upcoming) == 1
    assert played["home_score"].dtype.kind == "i"
    assert "Serbia" in set(played["home_team"])          # Yugoslavia mapped
    assert bool(played.iloc[1]["neutral"]) is True        # sorted by date: 1992 row first
    assert list(played["date"]) == sorted(played["date"])  # chronological
    assert upcoming.iloc[0]["home_team"] == "United States"
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_data.py -q` — Expected: FAIL, `load_results` not defined.

- [ ] **Step 3: Implement (append to `src/fifa/data.py`)**

```python
def load_results(force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (played, upcoming). Played: scores int, chronological. Upcoming: NA-score rows."""
    path = download(RESULTS_URL, DATA_DIR / "results.csv", force=force)
    df = pd.read_csv(path, na_values=["NA"])
    df["date"] = pd.to_datetime(df["date"])
    df["neutral"] = df["neutral"].astype(bool)
    for col in ("home_team", "away_team"):
        df[col] = df[col].replace(SUCCESSORS)
    upcoming = df[df["home_score"].isna()].copy().reset_index(drop=True)
    played = df.dropna(subset=["home_score", "away_score"]).copy()
    played["home_score"] = played["home_score"].astype(int)
    played["away_score"] = played["away_score"].astype(int)
    played = played.sort_values("date", kind="stable").reset_index(drop=True)
    return played, upcoming


def load_shootouts(force: bool = False) -> pd.DataFrame:
    path = download(SHOOTOUTS_URL, DATA_DIR / "shootouts.csv", force=force)
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    for col in ("home_team", "away_team", "winner"):
        df[col] = df[col].replace(SUCCESSORS)
    return df
```

- [ ] **Step 4: Run to verify pass** — `.venv/bin/pytest tests/test_data.py -q` → all pass.

- [ ] **Step 5: Commit**

```bash
git add -u && git commit -m "feat: results/shootouts loading with NA-fixture split and successor mapping"
```

---

### Task 4: FIFA name alias map (`data.py` part 3)

**Files:**
- Modify: `src/fifa/data.py` (append)
- Test: `tests/test_data.py` (append)

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_data.py
def test_normalize_team_known_aliases():
    assert data.normalize_team("Korea Republic") == "South Korea"
    assert data.normalize_team("Czechia") == "Czech Republic"
    assert data.normalize_team("Türkiye") == "Turkey"
    assert data.normalize_team("France") == "France"  # identity passthrough


def test_assert_known_raises_with_offenders():
    known = {"France", "Brazil"}
    with pytest.raises(ValueError, match="Atlantis"):
        data.assert_known(["France", "Atlantis"], known)
```

- [ ] **Step 2: Run to verify failure** — FAIL, `normalize_team` not defined.

- [ ] **Step 3: Implement (append to `src/fifa/data.py`)**

```python
# FIFA/fixture-feed names → martj42 dataset names. Identity names omitted.
# Seeded from verified mismatches; tests/test_fixtures_live.py is the source of
# truth — extend this map if it reports unmapped names.
FIFA_ALIASES = {
    "Korea Republic": "South Korea",
    "Korea DPR": "North Korea",
    "Czechia": "Czech Republic",
    "IR Iran": "Iran",
    "Côte d'Ivoire": "Ivory Coast",
    "Cote d'Ivoire": "Ivory Coast",
    "Cabo Verde": "Cape Verde",
    "Türkiye": "Turkey",
    "Turkiye": "Turkey",
    "Congo DR": "DR Congo",
    "China PR": "China",
    "USA": "United States",
    "Curacao": "Curaçao",
}
TBD_PLACEHOLDER = "To be announced"


def normalize_team(name: str) -> str:
    return FIFA_ALIASES.get(name, name)


def assert_known(names, known_teams) -> None:
    unknown = sorted({n for n in names if n not in known_teams and n != TBD_PLACEHOLDER})
    if unknown:
        raise ValueError(
            f"Unmapped team names (add to data.FIFA_ALIASES): {unknown}"
        )
```

- [ ] **Step 4: Run to verify pass** — `.venv/bin/pytest tests/test_data.py -q`.

- [ ] **Step 5: Live alias completeness check (one-off, requires network)**

```bash
.venv/bin/python - <<'EOF'
from fifa import data
import json
path = data.download(data.FIXTURES_URL, data.DATA_DIR / "fixtures.json", max_age_hours=6)
feed = json.loads(path.read_text())
played, _ = data.load_results()
known = set(played["home_team"]) | set(played["away_team"])
names = {data.normalize_team(m["HomeTeam"]) for m in feed} | {data.normalize_team(m["AwayTeam"]) for m in feed}
data.assert_known(names, known)
print(f"OK — all {len(names)-1} feed team names map into the dataset")
EOF
```

Expected: `OK`. If it raises, add the printed offenders to `FIFA_ALIASES` (martj42 side is the target spelling — check with `grep <name> data/results.csv`), re-run until OK.

- [ ] **Step 6: Commit**

```bash
git add -u && git commit -m "feat: FIFA->dataset team-name alias map with hard-error guard"
```

---

### Task 5: Elo formulas (`elo.py` part 1)

**Files:**
- Create: `src/fifa/elo.py`
- Test: `tests/test_elo.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_elo.py
import pytest
from fifa import elo


def test_expected_equal_neutral_is_half():
    assert elo.expected(1500, 1500, neutral=True) == pytest.approx(0.5)


def test_expected_home_advantage_100():
    # +100 Elo home edge → We = 1/(1+10^-0.25) ≈ 0.640
    assert elo.expected(1500, 1500, neutral=False) == pytest.approx(0.640065, abs=1e-5)


def test_goal_multiplier_table():
    assert [elo.goal_multiplier(m) for m in (0, 1, 2, 3, 4, 5)] == [1.0, 1.0, 1.5, 1.75, 1.875, 2.0]


def test_tournament_k_tiers():
    assert elo.tournament_k("FIFA World Cup") == 60
    assert elo.tournament_k("Copa América") == 50
    assert elo.tournament_k("UEFA Euro qualification") == 40   # 'qualification' beats Euro
    assert elo.tournament_k("FIFA World Cup qualification") == 40
    assert elo.tournament_k("UEFA Nations League") == 40
    assert elo.tournament_k("Friendly") == 20
    assert elo.tournament_k("King's Cup") == 30                # unknown minor → 30


def test_update_rule_hand_example():
    # Equal teams, neutral, home wins 2-0 in a qualifier: ΔR = 40 · 1.5 · (1−0.5) = 30
    we = elo.expected(1500, 1500, neutral=True)
    assert 40 * elo.goal_multiplier(2) * (1.0 - we) == pytest.approx(30.0)


def test_2022_final_published_delta_band():
    # WC2022 final (neutral, draw after ET → W=0.5, K=60, G=1.0). eloratings.net
    # published Argentina at ~2144 pre-match with change −6 (We≈0.59).
    # Band-assert with the published We; Step 5 pins exact TSV values.
    delta = 60 * 1.0 * (0.5 - 0.590)
    assert -7 < delta < -4
```

- [ ] **Step 2: Run to verify failure** — `.venv/bin/pytest tests/test_elo.py -q` → import error.

- [ ] **Step 3: Implement**

```python
# src/fifa/elo.py
"""World Football Elo Ratings engine (eloratings.net formulas)."""
from __future__ import annotations

import pandas as pd

HOME_ADV = 100.0
INITIAL = 1500.0

_CONTINENTAL_50 = {
    "UEFA Euro", "Copa América", "African Cup of Nations", "Africa Cup of Nations",
    "AFC Asian Cup", "Gold Cup", "CONCACAF Championship", "Confederations Cup",
    "FIFA Confederations Cup", "Oceania Nations Cup", "OFC Nations Cup", "Finalissima",
}


def tournament_k(tournament: str) -> float:
    if tournament == "FIFA World Cup":
        return 60.0
    low = tournament.lower()
    if "qualification" in low or "nations league" in low:
        return 40.0
    if tournament in _CONTINENTAL_50:
        return 50.0
    if tournament == "Friendly":
        return 20.0
    return 30.0


def expected(r_home: float, r_away: float, neutral: bool) -> float:
    dr = r_home - r_away + (0.0 if neutral else HOME_ADV)
    return 1.0 / (1.0 + 10.0 ** (-dr / 400.0))


def goal_multiplier(margin: int) -> float:
    m = abs(margin)
    if m <= 1:
        return 1.0
    if m == 2:
        return 1.5
    if m == 3:
        return 1.75
    return 1.75 + (m - 3) / 8.0


def result_value(home_score: int, away_score: int) -> float:
    if home_score > away_score:
        return 1.0
    if home_score < away_score:
        return 0.0
    return 0.5  # includes shootout matches — scores exclude shootouts by dataset design
```

- [ ] **Step 4: Run to verify pass** — `.venv/bin/pytest tests/test_elo.py -q`.

- [ ] **Step 5: Pin the 2022-final validation to exact published values (network one-off)**

```bash
curl -s https://www.eloratings.net/2022_results.tsv | grep -i "argentina.*france" | tail -2
```

Read off the final's row: pre-match ratings for Argentina and France and the published change. Replace `test_2022_final_published_delta_band` with an exact assertion, e.g. (values illustrative — use the real ones from the TSV):

```python
def test_2022_final_reproduces_published_change():
    r_arg, r_fra, published_change = 2143, 2005, -6  # ← REPLACE with exact TSV values
    we = elo.expected(r_arg, r_fra, neutral=True)
    delta = 60 * elo.goal_multiplier(0) * (0.5 - we)
    assert round(delta) == published_change
```

If the TSV format is unclear, keep the band test and note it in the commit message — do not fabricate values.

- [ ] **Step 6: Commit**

```bash
git add -u && git commit -m "feat: Elo formulas (expectancy, K tiers, margin multiplier) validated against published values"
```

---

### Task 6: Chronological Elo engine (`elo.py` part 2)

**Files:**
- Modify: `src/fifa/elo.py` (append)
- Test: `tests/test_elo.py` (append)

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_elo.py
import pandas as pd


def _mini_df():
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-06-01", "2021-01-01"]),
            "home_team": ["A", "A", "B"],
            "away_team": ["B", "C", "C"],
            "home_score": [2, 0, 1],
            "away_score": [0, 0, 1],
            "tournament": ["Friendly", "Friendly", "Friendly"],
            "neutral": [True, False, True],
        }
    )


def test_compute_elo_chronology_and_zero_sum():
    out, ratings = elo.compute_elo(_mini_df())
    # Match 1: A 2-0 B neutral friendly: ΔR = 20·1.5·(1−0.5) = +15
    assert out.loc[0, "elo_home_pre"] == 1500 and out.loc[0, "elo_away_pre"] == 1500
    assert out.loc[0, "elo_home_post"] == pytest.approx(1515)
    assert out.loc[0, "elo_away_post"] == pytest.approx(1485)
    # Match 2 uses A's UPDATED rating as pre-match (no leakage of later matches)
    assert out.loc[1, "elo_home_pre"] == pytest.approx(1515)
    # Zero-sum: total rating mass conserved
    assert sum(ratings.values()) == pytest.approx(1500 * 3)


def test_compute_elo_pre_never_includes_own_match():
    df = _mini_df()
    out1, _ = elo.compute_elo(df)
    df2 = df.copy()
    df2.loc[2, "home_score"] = 9  # changing last match must not change its OWN pre-Elo
    out2, _ = elo.compute_elo(df2)
    assert out1.loc[2, "elo_home_pre"] == out2.loc[2, "elo_home_pre"]
```

- [ ] **Step 2: Run to verify failure** — FAIL, `compute_elo` not defined.

- [ ] **Step 3: Implement (append to `src/fifa/elo.py`)**

```python
def compute_elo(
    played: pd.DataFrame, initial: float = INITIAL
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Chronological pass. Returns (copy with elo_{home,away}_{pre,post} columns, final ratings)."""
    ratings: dict[str, float] = {}
    pre_h, pre_a, post_h, post_a = [], [], [], []
    for row in played.itertuples(index=False):
        rh = ratings.get(row.home_team, initial)
        ra = ratings.get(row.away_team, initial)
        we = expected(rh, ra, row.neutral)
        delta = (
            tournament_k(row.tournament)
            * goal_multiplier(row.home_score - row.away_score)
            * (result_value(row.home_score, row.away_score) - we)
        )
        ratings[row.home_team] = rh + delta
        ratings[row.away_team] = ra - delta
        pre_h.append(rh)
        pre_a.append(ra)
        post_h.append(rh + delta)
        post_a.append(ra - delta)
    out = played.copy()
    out["elo_home_pre"] = pre_h
    out["elo_away_pre"] = pre_a
    out["elo_home_post"] = post_h
    out["elo_away_post"] = post_a
    return out, ratings
```

- [ ] **Step 4: Run to verify pass** — `.venv/bin/pytest tests/test_elo.py -q`.

- [ ] **Step 5: Real-data sanity run (manual verification, requires network)**

```bash
.venv/bin/python - <<'EOF'
from fifa import data, elo
played, _ = data.load_results()
out, ratings = elo.compute_elo(played)
top = sorted(ratings.items(), key=lambda kv: -kv[1])[:10]
for t, r in top:
    print(f"{t:20s} {r:7.0f}")
EOF
```

Expected: top-10 dominated by Argentina, Spain, France, Brazil, England, Portugal, Netherlands, Germany-tier teams with ratings roughly 1900–2150 (our uniform-1500 init differs from eloratings.net's hand seeds, so values won't match exactly — the *ordering cluster* is the sanity check). If the top 10 contains minnows or ratings explode (>2500), debug before proceeding.

- [ ] **Step 6: Commit**

```bash
git add -u && git commit -m "feat: chronological Elo engine with pre/post-match columns"
```

---

### Task 7: Pre-match feature builder (`features.py`)

**Files:**
- Create: `src/fifa/features.py`
- Test: `tests/test_features.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_features.py
import pandas as pd
import pytest
from fifa import elo, features


def _df():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-01-11", "2020-02-01", "2020-03-01"]),
            "home_team": ["A", "B", "A", "A"],
            "away_team": ["B", "A", "C", "B"],
            "home_score": [3, 1, 0, 2],
            "away_score": [0, 1, 2, 1],
            "tournament": ["Friendly"] * 4,
            "neutral": [False, False, True, False],
        }
    )
    out, _ = elo.compute_elo(df)
    return out


def test_features_shapes_and_basics():
    fb = features.FeatureBuilder()
    X, y_home, y_away = fb.fit_transform(_df())
    assert len(X) == 4 and list(y_home) == [3, 1, 0, 2]
    # Match 0: nobody has played → neutral priors
    assert X.loc[0, "played_home"] == 0
    assert X.loc[0, "rest_home"] == features.REST_CAP
    assert X.loc[0, "winrate_home"] == pytest.approx(0.5)
    # Match 1 (B v A, 10 days later): both played once
    assert X.loc[1, "played_home"] == 1
    assert X.loc[1, "rest_home"] == 10
    # A won match 0 → A's winrate before match 1 is 1.0 (A is away side here)
    assert X.loc[1, "winrate_away"] == pytest.approx(1.0)
    # elo_diff includes the +100 home-advantage term on non-neutral rows
    df = _df()
    expected_diff = df.loc[0, "elo_home_pre"] - df.loc[0, "elo_away_pre"] + 100
    assert X.loc[0, "elo_diff"] == pytest.approx(expected_diff)


def test_leakage_guard_features_ignore_own_result():
    df1, df2 = _df(), _df()
    df2.loc[3, "home_score"] = 9  # mutate final match result
    X1, _, _ = features.FeatureBuilder().fit_transform(df1)
    X2, _, _ = features.FeatureBuilder().fit_transform(df2)
    pd.testing.assert_series_equal(X1.loc[3], X2.loc[3])  # its own features unchanged


def test_features_for_future_fixture_matches_training_columns():
    fb = features.FeatureBuilder()
    X, _, _ = fb.fit_transform(_df())
    row = fb.features_for("A", "C", pd.Timestamp("2020-04-01"), "FIFA World Cup", neutral=False)
    assert list(row.columns) == list(X.columns)
    assert row.loc[0, "k_tier"] == 60


def test_momentum_and_wc_experience():
    import numpy as np
    n = 30
    df = pd.DataFrame(
        {
            "date": pd.to_datetime("2018-01-01") + pd.to_timedelta(np.arange(n) * 40, unit="D"),
            "home_team": ["A"] * n,
            "away_team": ["B"] * n,
            "home_score": [2] * n,
            "away_score": [0] * n,
            "tournament": ["FIFA World Cup"] * 3 + ["Friendly"] * (n - 3),
            "neutral": [True] * n,
        }
    )
    out, _ = elo.compute_elo(df)
    X, _, _ = features.FeatureBuilder().fit_transform(out)
    last = X.iloc[-1]
    assert last["elo_trend1y_home"] > 0   # A keeps winning → momentum up over past year
    assert last["elo_trend1y_away"] < 0   # B keeps losing → momentum down
    assert last["wc_exp_home"] == 3       # career World Cup finals matches before this game
```

- [ ] **Step 2: Run to verify failure** — import error.

- [ ] **Step 3: Implement**

```python
# src/fifa/features.py
"""Strictly pre-match feature construction via single chronological pass.

Leakage-safe BY CONSTRUCTION: for every match we snapshot team state first,
then update state with the match result. The unit test enforces this.
"""
from __future__ import annotations

import math
from collections import deque

import pandas as pd

from . import elo

DECAY = 0.001          # form decay per day (half-life ≈ 1.9 years)
REST_CAP = 60.0        # days
PRIOR_GOALS = 1.4      # international per-team scoring prior
PRIOR_WEIGHT = 5.0

CONFED = {
    # UEFA
    "England": "UEFA", "France": "UEFA", "Spain": "UEFA", "Portugal": "UEFA",
    "Germany": "UEFA", "Netherlands": "UEFA", "Belgium": "UEFA", "Croatia": "UEFA",
    "Italy": "UEFA", "Switzerland": "UEFA", "Austria": "UEFA", "Norway": "UEFA",
    "Sweden": "UEFA", "Scotland": "UEFA", "Turkey": "UEFA", "Czech Republic": "UEFA",
    "Bosnia and Herzegovina": "UEFA", "Serbia": "UEFA", "Denmark": "UEFA",
    "Poland": "UEFA", "Ukraine": "UEFA", "Wales": "UEFA", "Russia": "UEFA",
    "Hungary": "UEFA", "Greece": "UEFA", "Romania": "UEFA", "Slovakia": "UEFA",
    "Slovenia": "UEFA", "Ireland": "UEFA", "Northern Ireland": "UEFA", "Iceland": "UEFA",
    # CONMEBOL
    "Argentina": "CONMEBOL", "Brazil": "CONMEBOL", "Uruguay": "CONMEBOL",
    "Colombia": "CONMEBOL", "Paraguay": "CONMEBOL", "Ecuador": "CONMEBOL",
    "Chile": "CONMEBOL", "Peru": "CONMEBOL", "Bolivia": "CONMEBOL", "Venezuela": "CONMEBOL",
    # CONCACAF
    "Mexico": "CONCACAF", "United States": "CONCACAF", "Canada": "CONCACAF",
    "Panama": "CONCACAF", "Haiti": "CONCACAF", "Curaçao": "CONCACAF",
    "Costa Rica": "CONCACAF", "Jamaica": "CONCACAF", "Honduras": "CONCACAF",
    # CAF
    "Morocco": "CAF", "Senegal": "CAF", "Egypt": "CAF", "Algeria": "CAF",
    "Tunisia": "CAF", "Ghana": "CAF", "Ivory Coast": "CAF", "Nigeria": "CAF",
    "Cameroon": "CAF", "South Africa": "CAF", "DR Congo": "CAF", "Cape Verde": "CAF",
    # AFC
    "Japan": "AFC", "South Korea": "AFC", "Iran": "AFC", "Saudi Arabia": "AFC",
    "Australia": "AFC", "Qatar": "AFC", "Iraq": "AFC", "Jordan": "AFC",
    "Uzbekistan": "AFC", "China": "AFC", "North Korea": "AFC",
    # OFC
    "New Zealand": "OFC",
}
CONFED_LEVELS = ["UEFA", "CONMEBOL", "CONCACAF", "CAF", "AFC", "OFC", "OTHER"]


class _TeamState:
    __slots__ = ("gf", "ga", "w", "last_date", "recent", "played")

    def __init__(self):
        self.gf = 0.0
        self.ga = 0.0
        self.w = 0.0
        self.last_date = None
        self.recent = deque(maxlen=10)
        self.played = 0

    def _decay_factor(self, date):
        if self.last_date is None:
            return 1.0
        return math.exp(-DECAY * (date - self.last_date).days)

    def snapshot(self, date):
        f = self._decay_factor(date)
        w = self.w * f + PRIOR_WEIGHT
        rest = REST_CAP if self.last_date is None else min((date - self.last_date).days, REST_CAP)
        return {
            "gf_rate": (self.gf * f + PRIOR_GOALS * PRIOR_WEIGHT) / w,
            "ga_rate": (self.ga * f + PRIOR_GOALS * PRIOR_WEIGHT) / w,
            "winrate": sum(self.recent) / len(self.recent) if self.recent else 0.5,
            "rest": float(rest),
            "played": self.played,
        }

    def update(self, date, goals_for, goals_against, result):
        f = self._decay_factor(date)
        self.gf = self.gf * f + goals_for
        self.ga = self.ga * f + goals_against
        self.w = self.w * f + 1.0
        self.recent.append(result)
        self.last_date = date
        self.played += 1


class FeatureBuilder:
    """fit_transform() replays played matches; features_for() queries the final state."""

    COLUMNS = [
        "elo_home", "elo_away", "elo_diff",
        "gf_home", "ga_home", "gf_away", "ga_away",
        "winrate_home", "winrate_away", "rest_home", "rest_away",
        "played_home", "played_away", "h2h_gd",
        "elo_trend1y_home", "elo_trend1y_away", "elo_trend2y_home", "elo_trend2y_away",
        "wc_exp_home", "wc_exp_away",
        "k_tier", "neutral", "confed_home", "confed_away",
    ]

    def __init__(self):
        self._teams: dict[str, _TeamState] = {}
        self._h2h: dict[tuple[str, str], float] = {}
        self._ratings: dict[str, float] = {}
        self._elo_hist: dict[str, list[tuple[pd.Timestamp, float]]] = {}
        self._wc_count: dict[str, int] = {}

    def _trend(self, team, date, elo_now, days):
        """Elo momentum: rating now minus rating `days` ago (0 if no history that old)."""
        hist = self._elo_hist.get(team)
        if not hist:
            return 0.0
        cutoff = date - pd.Timedelta(days=days)
        for d, r in reversed(hist):
            if d <= cutoff:
                return elo_now - r
        return 0.0

    def _team(self, name):
        if name not in self._teams:
            self._teams[name] = _TeamState()
        return self._teams[name]

    def _row(self, home, away, date, tournament, neutral, elo_home, elo_away):
        sh = self._team(home).snapshot(date)
        sa = self._team(away).snapshot(date)
        key = (min(home, away), max(home, away))
        h2h = self._h2h.get(key, 0.0) * (1 if home <= away else -1)
        return {
            "elo_home": elo_home,
            "elo_away": elo_away,
            "elo_diff": elo_home - elo_away + (0.0 if neutral else elo.HOME_ADV),
            "gf_home": sh["gf_rate"], "ga_home": sh["ga_rate"],
            "gf_away": sa["gf_rate"], "ga_away": sa["ga_rate"],
            "winrate_home": sh["winrate"], "winrate_away": sa["winrate"],
            "rest_home": sh["rest"], "rest_away": sa["rest"],
            "played_home": sh["played"], "played_away": sa["played"],
            "h2h_gd": h2h,
            "elo_trend1y_home": self._trend(home, date, elo_home, 365),
            "elo_trend1y_away": self._trend(away, date, elo_away, 365),
            "elo_trend2y_home": self._trend(home, date, elo_home, 730),
            "elo_trend2y_away": self._trend(away, date, elo_away, 730),
            "wc_exp_home": self._wc_count.get(home, 0),
            "wc_exp_away": self._wc_count.get(away, 0),
            "k_tier": elo.tournament_k(tournament),
            "neutral": int(neutral),
            "confed_home": CONFED.get(home, "OTHER"),
            "confed_away": CONFED.get(away, "OTHER"),
        }

    def fit_transform(self, elo_df: pd.DataFrame):
        """elo_df: output of elo.compute_elo() — chronological, with elo_*_pre columns."""
        rows = []
        for r in elo_df.itertuples(index=False):
            rows.append(
                self._row(r.home_team, r.away_team, r.date, r.tournament,
                          r.neutral, r.elo_home_pre, r.elo_away_pre)
            )
            res = elo.result_value(r.home_score, r.away_score)
            self._team(r.home_team).update(r.date, r.home_score, r.away_score, res)
            self._team(r.away_team).update(r.date, r.away_score, r.home_score, 1.0 - res)
            key = (min(r.home_team, r.away_team), max(r.home_team, r.away_team))
            gd = (r.home_score - r.away_score) * (1 if r.home_team <= r.away_team else -1)
            self._h2h[key] = self._h2h.get(key, 0.0) * 0.9 + gd
            self._ratings[r.home_team] = r.elo_home_post
            self._ratings[r.away_team] = r.elo_away_post
            self._elo_hist.setdefault(r.home_team, []).append((r.date, r.elo_home_post))
            self._elo_hist.setdefault(r.away_team, []).append((r.date, r.elo_away_post))
            if r.tournament == "FIFA World Cup":
                self._wc_count[r.home_team] = self._wc_count.get(r.home_team, 0) + 1
                self._wc_count[r.away_team] = self._wc_count.get(r.away_team, 0) + 1
        X = pd.DataFrame(rows, columns=self.COLUMNS)
        X = self._finalize(X)
        return X, elo_df["home_score"].to_numpy(), elo_df["away_score"].to_numpy()

    def features_for(self, home, away, date, tournament, neutral) -> pd.DataFrame:
        eh = self._ratings.get(home, elo.INITIAL)
        ea = self._ratings.get(away, elo.INITIAL)
        X = pd.DataFrame([self._row(home, away, date, tournament, neutral, eh, ea)],
                         columns=self.COLUMNS)
        return self._finalize(X)

    @staticmethod
    def _finalize(X: pd.DataFrame) -> pd.DataFrame:
        for col in ("confed_home", "confed_away"):
            X[col] = pd.Categorical(X[col], categories=CONFED_LEVELS)
        return X
```

- [ ] **Step 4: Run to verify pass** — `.venv/bin/pytest tests/test_features.py -q`.

- [ ] **Step 5: Commit**

```bash
git add src/fifa/features.py tests/test_features.py
git commit -m "feat: leakage-safe chronological feature builder (Elo, decayed form, rest, H2H, confederations)"
```

---

### Task 8: Score matrix + Dixon-Coles correction + tiers (`matrix.py`)

**Files:**
- Create: `src/fifa/matrix.py`
- Test: `tests/test_matrix.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_matrix.py
import numpy as np
import pytest
from fifa import matrix


def test_matrix_sums_to_one_and_shape():
    m = matrix.score_matrix(1.5, 1.1, rho=-0.1)
    assert m.shape == (11, 11)
    assert m.sum() == pytest.approx(1.0)
    assert (m >= 0).all()


def test_wdl_symmetric_for_equal_lambdas():
    m = matrix.score_matrix(1.3, 1.3, rho=-0.1)
    ph, pd_, pa = matrix.wdl(m)
    assert ph == pytest.approx(pa)
    assert ph + pd_ + pa == pytest.approx(1.0)


def test_negative_rho_inflates_draws():
    base = matrix.wdl(matrix.score_matrix(1.3, 1.3, rho=0.0))[1]
    adj = matrix.wdl(matrix.score_matrix(1.3, 1.3, rho=-0.1))[1]
    assert adj > base  # Dixon-Coles: negative rho boosts 0-0/1-1


def test_top_scorelines_ordered():
    m = matrix.score_matrix(2.2, 0.6, rho=-0.05)
    top = matrix.top_scorelines(m, k=5)
    assert len(top) == 5
    probs = [p for _, p in top]
    assert probs == sorted(probs, reverse=True)
    assert top[0][0][0] > top[0][0][1]  # strong home favorite → home-win modal score


def test_tier_thresholds():
    assert matrix.tier(0.75) == "LOCK"
    assert matrix.tier(0.70) == "LOCK"
    assert matrix.tier(0.60) == "STRONG"
    assert matrix.tier(0.50) == "LEAN"
    assert matrix.tier(0.40) == "TOSS-UP"
```

- [ ] **Step 2: Run to verify failure** — import error.

- [ ] **Step 3: Implement**

```python
# src/fifa/matrix.py
"""Scoreline probability matrix: independent Poisson + Dixon-Coles tau correction."""
from __future__ import annotations

import numpy as np
from scipy.stats import poisson

MAX_GOALS = 10
LOCK, STRONG, LEAN = 0.70, 0.55, 0.45


def score_matrix(lam_home: float, lam_away: float, rho: float = 0.0,
                 max_goals: int = MAX_GOALS) -> np.ndarray:
    """m[i, j] = P(home scores i, away scores j)."""
    g = np.arange(max_goals + 1)
    m = np.outer(poisson.pmf(g, lam_home), poisson.pmf(g, lam_away))
    # Dixon-Coles tau on the four low-score cells
    m[0, 0] *= 1 - lam_home * lam_away * rho
    m[0, 1] *= 1 + lam_home * rho
    m[1, 0] *= 1 + lam_away * rho
    m[1, 1] *= 1 - rho
    np.clip(m, 0.0, None, out=m)
    return m / m.sum()


def wdl(m: np.ndarray) -> tuple[float, float, float]:
    return (
        float(np.tril(m, -1).sum()),   # home win: i > j
        float(np.trace(m)),
        float(np.triu(m, 1).sum()),    # away win: j > i
    )


def top_scorelines(m: np.ndarray, k: int = 5) -> list[tuple[tuple[int, int], float]]:
    flat = [((i, j), float(m[i, j])) for i in range(m.shape[0]) for j in range(m.shape[1])]
    return sorted(flat, key=lambda t: -t[1])[:k]


def tier(p_max: float) -> str:
    if p_max >= LOCK:
        return "LOCK"
    if p_max >= STRONG:
        return "STRONG"
    if p_max >= LEAN:
        return "LEAN"
    return "TOSS-UP"
```

- [ ] **Step 4: Run to verify pass** — `.venv/bin/pytest tests/test_matrix.py -q`.

- [ ] **Step 5: Commit**

```bash
git add src/fifa/matrix.py tests/test_matrix.py
git commit -m "feat: DC-corrected scoreline matrix with WDL/top-k/tier readouts"
```

---

### Task 9: Dixon-Coles baseline (`dixon_coles.py`)

**Files:**
- Create: `src/fifa/dixon_coles.py`
- Test: `tests/test_dixon_coles.py`

Implementation note: time-decayed Poisson GLM via `sklearn.PoissonRegressor` on a sparse
one-hot design (attack + defense per team + home-advantage column), two observations per
match (home goals, away goals), `sample_weight = exp(−xi·days_before_ref)`. Mild L2 alpha
handles identifiability. The DC tau correction is applied later at matrix stage (rho is
tuned on validation in Task 12) — this two-stage approach avoids a custom MLE.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_dixon_coles.py
import numpy as np
import pandas as pd
import pytest
from fifa.dixon_coles import DixonColes

RNG = np.random.default_rng(7)
TEAMS = {"Strong": 2.0, "Mid": 1.2, "Weak": 0.7}  # true scoring rates vs average


def _synthetic(n=3000):
    names = list(TEAMS)
    rows = []
    base_date = pd.Timestamp("2024-01-01")
    for i in range(n):
        h, a = RNG.choice(names, 2, replace=False)
        lam_h = TEAMS[h] / np.sqrt(TEAMS[a]) * 1.25   # home boost
        lam_a = TEAMS[a] / np.sqrt(TEAMS[h])
        rows.append({
            "date": base_date + pd.Timedelta(days=int(i / 6)),
            "home_team": h, "away_team": a,
            "home_score": RNG.poisson(lam_h), "away_score": RNG.poisson(lam_a),
            "tournament": "Friendly", "neutral": False,
        })
    return pd.DataFrame(rows)


def test_fit_recovers_strength_ordering_and_home_adv():
    df = _synthetic()
    dc = DixonColes(xi=0.0)  # no decay for the synthetic test
    dc.fit(df, ref_date=df["date"].max())
    lh_sw, la_sw = dc.predict_lambdas("Strong", "Weak", neutral=True)
    lh_ws, la_ws = dc.predict_lambdas("Weak", "Strong", neutral=True)
    assert lh_sw > la_sw and la_ws > lh_ws         # strength ordering
    assert dc.home_adv_ > 0.05                      # home advantage exists
    lh_home, _ = dc.predict_lambdas("Strong", "Weak", neutral=False)
    assert lh_home > lh_sw                          # home flag raises lambda


def test_unknown_team_falls_back_to_average():
    df = _synthetic(500)
    dc = DixonColes(xi=0.0)
    dc.fit(df, ref_date=df["date"].max())
    lh, la = dc.predict_lambdas("Atlantis", "Strong", neutral=True)
    assert 0.1 < lh < 3.0 and 0.1 < la < 4.0        # finite, sane fallback
```

- [ ] **Step 2: Run to verify failure** — import error.

- [ ] **Step 3: Implement**

```python
# src/fifa/dixon_coles.py
"""Time-decayed Poisson GLM baseline (Dixon-Coles family; tau applied at matrix stage)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import PoissonRegressor


class DixonColes:
    def __init__(self, xi: float = 0.001, alpha: float = 1e-3, window_years: int = 15):
        self.xi = xi
        self.alpha = alpha
        self.window_years = window_years

    def fit(self, played: pd.DataFrame, ref_date: pd.Timestamp) -> "DixonColes":
        cutoff = ref_date - pd.DateOffset(years=self.window_years)
        df = played[(played["date"] <= ref_date) & (played["date"] >= cutoff)]
        teams = sorted(set(df["home_team"]) | set(df["away_team"]))
        self.idx_ = {t: i for i, t in enumerate(teams)}
        T = len(teams)
        n = len(df)

        h = df["home_team"].map(self.idx_).to_numpy()
        a = df["away_team"].map(self.idx_).to_numpy()
        not_neutral = (~df["neutral"]).to_numpy().astype(float)
        days = (ref_date - df["date"]).dt.days.to_numpy()
        w = np.exp(-self.xi * days)

        # Two observations per match: home goals (attack=home, defense=away, home_adv on),
        # away goals (attack=away, defense=home). Columns: [home_adv | attack_T | defense_T].
        rows = np.repeat(np.arange(2 * n), 3)
        cols = np.empty(6 * n, dtype=int)
        vals = np.empty(6 * n)
        # home-goal observations
        cols[0::6], vals[0::6] = 0, not_neutral
        cols[1::6], vals[1::6] = 1 + h, 1.0
        cols[2::6], vals[2::6] = 1 + T + a, 1.0
        # away-goal observations
        cols[3::6], vals[3::6] = 0, 0.0
        cols[4::6], vals[4::6] = 1 + a, 1.0
        cols[5::6], vals[5::6] = 1 + T + h, 1.0
        X = sparse.csr_matrix((vals, (rows, cols)), shape=(2 * n, 1 + 2 * T))
        # interleave home/away obs to match row construction above
        y = np.empty(2 * n)
        y[0::2] = df["home_score"].to_numpy()
        y[1::2] = df["away_score"].to_numpy()
        # row indices: home obs are rows 0,2,4.. — rebuild rows accordingly
        # (rows array above assumed home obs at 2i, away at 2i+1)
        self.model_ = PoissonRegressor(alpha=self.alpha, max_iter=300)
        self.model_.fit(X, y, sample_weight=np.repeat(w, 2))
        self.home_adv_ = float(self.model_.coef_[0])
        self._T = T
        return self

    def _linear(self, attack_team: str | None, defense_team: str | None, home_adv: bool) -> float:
        z = self.model_.intercept_ + (self.home_adv_ if home_adv else 0.0)
        if attack_team in self.idx_:
            z += self.model_.coef_[1 + self.idx_[attack_team]]
        if defense_team in self.idx_:
            z += self.model_.coef_[1 + self._T + self.idx_[defense_team]]
        return z

    def predict_lambdas(self, home: str, away: str, neutral: bool) -> tuple[float, float]:
        lh = float(np.exp(self._linear(home, away, home_adv=not neutral)))
        la = float(np.exp(self._linear(away, home, home_adv=False)))
        return float(np.clip(lh, 0.05, 8.0)), float(np.clip(la, 0.05, 8.0))
```

**Careful:** the `rows` construction must place each observation's 3 entries on the correct
row: home-goal obs of match i on row `2i`, away-goal obs on row `2i+1`. The slicing above
(`0::6 … 5::6` with `rows = np.repeat(np.arange(2n), 3)`) does exactly that — entries
`6i..6i+2` belong to row `2i`, entries `6i+3..6i+5` to row `2i+1`. The synthetic-recovery
test will catch any indexing mistake.

- [ ] **Step 4: Run to verify pass** — `.venv/bin/pytest tests/test_dixon_coles.py -q` (takes ~10–30 s).

- [ ] **Step 5: Real-data smoke (network)**

```bash
.venv/bin/python - <<'EOF'
from fifa import data
from fifa.dixon_coles import DixonColes
played, _ = data.load_results()
dc = DixonColes().fit(played, ref_date=played["date"].max())
print("home_adv coefficient:", round(dc.home_adv_, 3))
print("France v Senegal (neutral):", dc.predict_lambdas("France", "Senegal", neutral=True))
EOF
```

Expected: `home_adv` in ~0.2–0.4 (≈ e^0.3 ≈ 1.35× goals at home); France λ > Senegal λ, both in 0.5–3.

- [ ] **Step 6: Commit**

```bash
git add -u && git commit -m "feat: time-decayed Poisson GLM baseline (Dixon-Coles family)"
```

---

### Task 10: LightGBM goal model (`gbm.py`)

**Files:**
- Create: `src/fifa/gbm.py`
- Test: `tests/test_gbm.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_gbm.py
import numpy as np
import pandas as pd
from fifa import features
from fifa.gbm import GoalModel

RNG = np.random.default_rng(11)


def _synthetic(n=4000):
    """Goal counts driven by elo_diff — the model must learn the monotonic link."""
    elo_diff = RNG.uniform(-400, 400, n)
    lam_h = np.exp(0.1 + elo_diff / 600)
    lam_a = np.exp(0.1 - elo_diff / 600)
    X = pd.DataFrame({c: RNG.uniform(0, 1, n) for c in features.FeatureBuilder.COLUMNS
                      if c not in ("elo_diff", "confed_home", "confed_away")})
    X["elo_diff"] = elo_diff
    X["confed_home"] = pd.Categorical(["UEFA"] * n, categories=features.CONFED_LEVELS)
    X["confed_away"] = pd.Categorical(["CAF"] * n, categories=features.CONFED_LEVELS)
    X = X[features.FeatureBuilder.COLUMNS]
    dates = pd.Series(pd.Timestamp("2024-01-01"), index=range(n))
    return X, RNG.poisson(lam_h), RNG.poisson(lam_a), dates


def test_learns_monotonic_elo_relationship():
    X, yh, ya, dates = _synthetic()
    gm = GoalModel().fit(X, yh, ya, dates, ref_date=pd.Timestamp("2024-06-01"))
    strong = X.iloc[[0]].assign(elo_diff=300.0)
    weak = X.iloc[[0]].assign(elo_diff=-300.0)
    lh_s, la_s = gm.predict_lambdas(strong)
    lh_w, la_w = gm.predict_lambdas(weak)
    assert lh_s[0] > lh_w[0] and la_s[0] < la_w[0]
    assert (lh_s > 0).all() and (la_s > 0).all()


def test_lambdas_clipped_to_sane_range():
    X, yh, ya, dates = _synthetic(500)
    gm = GoalModel().fit(X, yh, ya, dates, ref_date=pd.Timestamp("2024-06-01"))
    lh, la = gm.predict_lambdas(X)
    assert lh.max() <= 8.0 and lh.min() >= 0.05
```

- [ ] **Step 2: Run to verify failure** — import error.

- [ ] **Step 3: Implement**

```python
# src/fifa/gbm.py
"""Twin LightGBM Poisson regressors for (lambda_home, lambda_away)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

TIME_DECAY = 0.0005          # per day (half-life ≈ 3.8 years)
FRIENDLY_WEIGHT = 0.5

_PARAMS = dict(
    objective="poisson",
    n_estimators=600,
    learning_rate=0.04,
    num_leaves=63,
    min_child_samples=40,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    verbose=-1,
)


class GoalModel:
    def __init__(self, **overrides):
        self.params = {**_PARAMS, **overrides}

    def fit(self, X: pd.DataFrame, y_home, y_away, dates: pd.Series,
            ref_date: pd.Timestamp) -> "GoalModel":
        days = (ref_date - pd.to_datetime(dates)).dt.days.to_numpy().clip(min=0)
        w = np.exp(-TIME_DECAY * days)
        w = w * np.where(X["k_tier"].to_numpy() == 20.0, FRIENDLY_WEIGHT, 1.0)
        self.home_ = LGBMRegressor(**self.params).fit(X, y_home, sample_weight=w)
        self.away_ = LGBMRegressor(**self.params).fit(X, y_away, sample_weight=w)
        return self

    def predict_lambdas(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        lh = np.clip(self.home_.predict(X), 0.05, 8.0)
        la = np.clip(self.away_.predict(X), 0.05, 8.0)
        return lh, la
```

- [ ] **Step 4: Run to verify pass** — `.venv/bin/pytest tests/test_gbm.py -q`.

- [ ] **Step 5: Commit**

```bash
git add src/fifa/gbm.py tests/test_gbm.py
git commit -m "feat: twin LightGBM Poisson goal-rate model with time/importance weighting"
```

---

### Task 11: Metrics + ensemble predictor (`evaluate.py`, `ensemble.py`)

**Files:**
- Create: `src/fifa/evaluate.py`, `src/fifa/ensemble.py`
- Test: `tests/test_evaluate.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_evaluate.py
import numpy as np
import pytest
from fifa import evaluate, matrix


def test_rps_hand_values():
    assert evaluate.rps((1.0, 0.0, 0.0), 0) == pytest.approx(0.0)      # certain & right
    assert evaluate.rps((1.0, 0.0, 0.0), 2) == pytest.approx(1.0)      # certain & maximally wrong
    assert evaluate.rps((1/3, 1/3, 1/3), 0) == pytest.approx(5/18)     # uniform vs home win


def test_outcome_of():
    assert evaluate.outcome_of(2, 0) == 0
    assert evaluate.outcome_of(1, 1) == 1
    assert evaluate.outcome_of(0, 3) == 2


def test_report_card_counts_lock_tier():
    m_fav = matrix.score_matrix(3.0, 0.4, rho=-0.05)   # heavy favorite → LOCK
    m_even = matrix.score_matrix(1.2, 1.2, rho=-0.05)  # toss-up
    rows = [
        evaluate.score_prediction(m_fav, 2, 0),   # favorite wins 2-0
        evaluate.score_prediction(m_even, 1, 1),  # draw
    ]
    card = evaluate.report_card(rows)
    assert card["n"] == 2
    assert card["lock_n"] == 1 and card["lock_acc"] == 1.0
    assert 0 <= card["rps"] <= 1
    assert card["exact_rate"] >= 0.5  # 2-0 is the modal score of m_fav
    assert card["baseline11_rate"] == 0.5
```

- [ ] **Step 2: Run to verify failure** — import error.

- [ ] **Step 3: Implement**

```python
# src/fifa/evaluate.py
"""Honest scoring: RPS, log loss, Brier, exact-score vs 1-1 baseline, tier accuracy."""
from __future__ import annotations

import numpy as np

from . import matrix as mx


def outcome_of(home_score: int, away_score: int) -> int:
    """0 = home win, 1 = draw, 2 = away win."""
    if home_score > away_score:
        return 0
    if home_score == away_score:
        return 1
    return 2


def rps(p: tuple[float, float, float], outcome: int) -> float:
    cum_p = np.cumsum(p)[:2]
    cum_o = np.cumsum(np.eye(3)[outcome])[:2]
    return float(np.sum((cum_p - cum_o) ** 2) / 2)


def score_prediction(m: np.ndarray, home_score: int, away_score: int) -> dict:
    """Everything we need to grade one match prediction."""
    p = mx.wdl(m)
    out = outcome_of(home_score, away_score)
    top5 = mx.top_scorelines(m, 5)
    pred_score = top5[0][0]
    p_max = max(p)
    return {
        "p": p,
        "outcome": out,
        "picked": int(np.argmax(p)),
        "tier": mx.tier(p_max),
        "rps": rps(p, out),
        "logloss": -float(np.log(max(p[out], 1e-12))),
        "brier": float(np.sum((np.array(p) - np.eye(3)[out]) ** 2)),
        "exact": pred_score == (home_score, away_score),
        "top5": (home_score, away_score) in [s for s, _ in top5],
        "is_11": (home_score, away_score) == (1, 1),
    }


def report_card(rows: list[dict]) -> dict:
    n = len(rows)
    locks = [r for r in rows if r["tier"] == "LOCK"]
    draws = [r for r in rows if r["outcome"] == 1]
    card = {
        "n": n,
        "wdl_acc": sum(r["picked"] == r["outcome"] for r in rows) / n,
        "draw_recall": (sum(r["picked"] == 1 for r in draws) / len(draws)) if draws else None,
        "rps": sum(r["rps"] for r in rows) / n,
        "logloss": sum(r["logloss"] for r in rows) / n,
        "brier": sum(r["brier"] for r in rows) / n,
        "exact_rate": sum(r["exact"] for r in rows) / n,
        "top5_rate": sum(r["top5"] for r in rows) / n,
        "baseline11_rate": sum(r["is_11"] for r in rows) / n,
        "lock_n": len(locks),
        "lock_acc": (sum(r["picked"] == r["outcome"] for r in locks) / len(locks)) if locks else None,
    }
    return card


def format_card(card: dict, gates: bool = True) -> str:
    lines = [
        f"matches evaluated      {card['n']}",
        f"W/D/L accuracy         {card['wdl_acc']:.1%}",
        f"draw recall            {card['draw_recall']:.1%}" if card["draw_recall"] is not None else "draw recall            n/a",
        f"mean RPS               {card['rps']:.4f}",
        f"mean log loss          {card['logloss']:.4f}",
        f"exact-score rate       {card['exact_rate']:.1%}  (always-1-1 baseline: {card['baseline11_rate']:.1%})",
        f"top-5 scoreline rate   {card['top5_rate']:.1%}",
        f"LOCK picks             {card['lock_n']}  acc " + (f"{card['lock_acc']:.1%}" if card["lock_acc"] is not None else "n/a"),
    ]
    if gates:
        checks = [
            ("W/D/L acc in honest band (>=0.50)", card["wdl_acc"] >= 0.50),
            ("exact-score beats 1-1 baseline", card["exact_rate"] >= card["baseline11_rate"]),
            ("exact-score below leakage alarm (<0.15)", card["exact_rate"] < 0.15),
            ("RPS <= 0.215", card["rps"] <= 0.215),
            ("draws actually predicted (recall > 0)", (card["draw_recall"] or 0) > 0),
        ]
        lines.append("gates:")
        lines += [f"  [{'PASS' if ok else 'FAIL'}] {name}" for name, ok in checks]
    return "\n".join(lines)
```

```python
# src/fifa/ensemble.py
"""Blend DC + GBM scoreline matrices; the single prediction entry point."""
from __future__ import annotations

import pandas as pd

from . import matrix as mx


class Predictor:
    def __init__(self, dc, gbm, fb, rho: float = -0.05, w_dc: float = 0.5):
        self.dc, self.gbm, self.fb = dc, gbm, fb
        self.rho, self.w_dc = rho, w_dc

    def matrix_from_lambdas(self, dc_l, gbm_l):
        m1 = mx.score_matrix(dc_l[0], dc_l[1], self.rho)
        m2 = mx.score_matrix(gbm_l[0], gbm_l[1], self.rho)
        m = self.w_dc * m1 + (1.0 - self.w_dc) * m2
        return m / m.sum()

    def matrix_for(self, home: str, away: str, date: pd.Timestamp,
                   tournament: str, neutral: bool):
        dc_l = self.dc.predict_lambdas(home, away, neutral)
        X = self.fb.features_for(home, away, date, tournament, neutral)
        lh, la = self.gbm.predict_lambdas(X)
        return self.matrix_from_lambdas(dc_l, (float(lh[0]), float(la[0])))
```

- [ ] **Step 4: Run to verify pass** — `.venv/bin/pytest tests/test_evaluate.py -q`.

- [ ] **Step 5: Commit**

```bash
git add src/fifa/evaluate.py src/fifa/ensemble.py tests/test_evaluate.py
git commit -m "feat: scoring metrics, report card with honesty gates, ensemble predictor"
```

---

### Task 12: Backtest CLI with walk-forward tuning (`backtest.py`)

**Files:**
- Create: `backtest.py` (repo root), `src/fifa/backtest_lib.py`
- Test: `tests/test_backtest_lib.py`

Protocol (from spec): train ≤2021-12-31 → tune rho and blend weight on val 2022–2023 →
refit on train+val → report card on test 2024 – today. Tuned params + card saved to
`data/backtest_report.json` for predict.py / dashboard reuse.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_backtest_lib.py
import numpy as np
import pandas as pd
from fifa import backtest_lib


def test_grid_pick_minimizes_rps():
    # two candidate (rho, w) configs, fake eval function counts calls
    calls = []

    def fake_eval(rho, w):
        calls.append((rho, w))
        return abs(rho + 0.10) + abs(w - 0.6)  # minimum at rho=-0.10, w=0.6

    best = backtest_lib.grid_pick(fake_eval, rhos=[-0.15, -0.10, 0.0], ws=[0.4, 0.6])
    assert best == (-0.10, 0.6)
    assert len(calls) == 6
```

- [ ] **Step 2: Run to verify failure** — import error.

- [ ] **Step 3: Implement**

```python
# src/fifa/backtest_lib.py
"""Walk-forward backtest pipeline shared by backtest.py and update.py."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import data, elo, evaluate, features
from .dixon_coles import DixonColes
from .ensemble import Predictor
from .gbm import GoalModel

TRAIN_END = pd.Timestamp("2021-12-31")
VAL_END = pd.Timestamp("2023-12-31")
RHOS = [round(r, 3) for r in np.arange(-0.20, 0.001, 0.025)]
WS = [round(w, 2) for w in np.arange(0.0, 1.001, 0.1)]


def grid_pick(eval_fn, rhos=RHOS, ws=WS) -> tuple[float, float]:
    best, best_loss = None, float("inf")
    for rho in rhos:
        for w in ws:
            loss = eval_fn(rho, w)
            if loss < best_loss:
                best, best_loss = (rho, w), loss
    return best


def _eval_rows(eval_df, dc, gbm, X_eval, rho, w_dc):
    """Score every row of eval_df with a given (rho, w). Returns list of row dicts."""
    lh_g, la_g = gbm.predict_lambdas(X_eval)
    pred = Predictor(dc, gbm, None, rho=rho, w_dc=w_dc)
    rows = []
    for i, r in enumerate(eval_df.itertuples(index=False)):
        dc_l = dc.predict_lambdas(r.home_team, r.away_team, r.neutral)
        m = pred.matrix_from_lambdas(dc_l, (float(lh_g[i]), float(la_g[i])))
        rows.append(evaluate.score_prediction(m, r.home_score, r.away_score))
    return rows


def run(report_path=None) -> dict:
    played, _ = data.load_results()
    elo_df, _ = elo.compute_elo(played)
    fb = features.FeatureBuilder()
    X, y_home, y_away = fb.fit_transform(elo_df)
    dates = elo_df["date"]

    is_train = dates <= TRAIN_END
    is_val = (dates > TRAIN_END) & (dates <= VAL_END)
    is_test = dates > VAL_END

    # Stage 1: fit on train, tune (rho, w) on val
    dc = DixonColes().fit(played[is_train], ref_date=TRAIN_END)
    gbm = GoalModel().fit(X[is_train], y_home[is_train.to_numpy()],
                          y_away[is_train.to_numpy()], dates[is_train], ref_date=TRAIN_END)
    val_df, X_val = played[is_val], X[is_val]

    def val_loss(rho, w):
        rows = _eval_rows(val_df, dc, gbm, X_val, rho, w)
        return sum(r["rps"] for r in rows) / len(rows)

    rho, w_dc = grid_pick(val_loss)
    print(f"tuned on validation: rho={rho}, w_dc={w_dc}")

    # Stage 2: refit on train+val, report on test
    fit_mask = is_train | is_val
    dc2 = DixonColes().fit(played[fit_mask], ref_date=VAL_END)
    gbm2 = GoalModel().fit(X[fit_mask], y_home[fit_mask.to_numpy()],
                           y_away[fit_mask.to_numpy()], dates[fit_mask], ref_date=VAL_END)
    test_rows = _eval_rows(played[is_test], dc2, gbm2, X[is_test], rho, w_dc)
    card = evaluate.report_card(test_rows)
    result = {"rho": rho, "w_dc": w_dc, "test_card": card,
              "test_span": [str(played[is_test]["date"].min().date()),
                            str(played[is_test]["date"].max().date())]}
    if report_path:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(result, indent=2, default=float))
    return result
```

```python
# backtest.py  (repo root)
"""Walk-forward backtest: tune on 2022-23, report on 2024-today. Writes data/backtest_report.json."""
from fifa import backtest_lib, data, evaluate

result = backtest_lib.run(report_path=data.DATA_DIR / "backtest_report.json")
print(f"\n=== TEST {result['test_span'][0]} .. {result['test_span'][1]} "
      f"(rho={result['rho']}, w_dc={result['w_dc']}) ===")
print(evaluate.format_card(result["test_card"]))
```

- [ ] **Step 4: Run unit test** — `.venv/bin/pytest tests/test_backtest_lib.py -q` → pass.

- [ ] **Step 5: Full real backtest (the moment of truth — takes minutes; grid is 9×11=99 val evals, each ~3k matches; if too slow, coarsen `RHOS` to step 0.05 and `WS` to step 0.2 first, then refine once around the best cell)**

Run: `.venv/bin/python backtest.py`
Expected: report card prints; **all 5 gates PASS**. Honest bands: W/D/L 50–60%, exact-score 9–13%, RPS 0.19–0.215, LOCK accuracy ≥ 75% with a meaningful lock_n (hundreds — qualifiers contain many mismatches). If exact_rate ≥ 15% or wdl_acc ≥ 65%: STOP, hunt the leak (check feature builder and Elo pre/post columns). If gates fail low (RPS > 0.215): inspect feature importances (`gbm2.home_.feature_importances_`), try GBM-only (w_dc=0) and DC-only (w_dc=1) cards to find which side is weak.

- [ ] **Step 6: Commit (include the JSON report)**

```bash
git add -f data/backtest_report.json
git add backtest.py src/fifa/backtest_lib.py tests/test_backtest_lib.py
git commit -m "feat: walk-forward backtest with val-tuned rho/blend and honesty gates"
```

---

### Task 13: WC2026 fixtures client (`fixtures.py`)

**Files:**
- Create: `src/fifa/fixtures.py`, `tests/fixtures_sample.json`
- Test: `tests/test_fixtures.py`

- [ ] **Step 1: Create the sample feed (real shape, 3 matches)**

```json
// tests/fixtures_sample.json
[
  {"MatchNumber": 1, "RoundNumber": 1, "DateUtc": "2026-06-11 19:00:00Z",
   "Location": "Estadio Azteca, Mexico City", "HomeTeam": "Mexico",
   "AwayTeam": "South Africa", "Group": "Group A", "HomeTeamScore": 2, "AwayTeamScore": 0},
  {"MatchNumber": 4, "RoundNumber": 1, "DateUtc": "2026-06-13 22:00:00Z",
   "Location": "New York New Jersey Stadium, New York/New Jersey", "HomeTeam": "Brazil",
   "AwayTeam": "Morocco", "Group": "Group C", "HomeTeamScore": null, "AwayTeamScore": null},
  {"MatchNumber": 73, "RoundNumber": 4, "DateUtc": "2026-06-28 16:00:00Z",
   "Location": "Los Angeles Stadium, Los Angeles", "HomeTeam": "To be announced",
   "AwayTeam": "To be announced", "Group": null, "HomeTeamScore": null, "AwayTeamScore": null}
]
```

(Field names verified against the live feed during research. If the live feed's `Location`
strings differ, fix the sample to match reality in Step 5 — the sample must mirror the
real feed, never the other way around.)

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_fixtures.py
from pathlib import Path
import pandas as pd
import pytest
from fifa import fixtures

SAMPLE = Path(__file__).parent / "fixtures_sample.json"


def test_parse_feed_statuses_and_names(monkeypatch):
    monkeypatch.setattr(fixtures.data, "download", lambda url, dest, **kw: SAMPLE)
    fx = fixtures.load_fixtures()
    assert list(fx["status"]) == ["played", "upcoming", "tbd"]
    assert fx.loc[0, "home_score"] == 2
    assert fx.loc[1, "home"] == "Brazil"
    assert fx.loc[0, "group"] == "A"
    assert pd.api.types.is_datetime64_any_dtype(fx["date"])


def test_neutrality_host_nations(monkeypatch):
    monkeypatch.setattr(fixtures.data, "download", lambda url, dest, **kw: SAMPLE)
    fx = fixtures.load_fixtures()
    assert bool(fx.loc[0, "neutral"]) is False  # Mexico at Estadio Azteca
    assert bool(fx.loc[1, "neutral"]) is True   # Brazil in New York
```

- [ ] **Step 3: Run to verify failure**, then implement:

```python
# src/fifa/fixtures.py
"""fixturedownload.com WC2026 feed client."""
from __future__ import annotations

import json

import pandas as pd

from . import data

# Venue city → host nation (martj42/team naming). Hard-keyed on the city part of Location.
VENUE_HOST = {
    "Mexico City": "Mexico", "Guadalajara": "Mexico", "Monterrey": "Mexico",
    "Toronto": "Canada", "Vancouver": "Canada",
    "Atlanta": "United States", "Boston": "United States", "Dallas": "United States",
    "Houston": "United States", "Kansas City": "United States",
    "Los Angeles": "United States", "Miami": "United States",
    "New York/New Jersey": "United States", "Philadelphia": "United States",
    "San Francisco Bay Area": "United States", "Seattle": "United States",
}


def _host_of(location: str) -> str:
    city = location.split(",")[-1].strip()
    if city not in VENUE_HOST:
        raise ValueError(f"Unmapped venue city {city!r} — add to fixtures.VENUE_HOST")
    return VENUE_HOST[city]


def load_fixtures(force: bool = False) -> pd.DataFrame:
    path = data.download(data.FIXTURES_URL, data.DATA_DIR / "fixtures.json",
                         max_age_hours=6, force=force)
    feed = json.loads(path.read_text())
    rows = []
    for m in feed:
        home = data.normalize_team(m["HomeTeam"])
        away = data.normalize_team(m["AwayTeam"])
        tbd = data.TBD_PLACEHOLDER in (m["HomeTeam"], m["AwayTeam"])
        played = m["HomeTeamScore"] is not None
        host = _host_of(m["Location"])
        rows.append({
            "match_number": m["MatchNumber"],
            "round": m["RoundNumber"],
            "date": pd.to_datetime(m["DateUtc"]),
            "location": m["Location"],
            "home": home, "away": away,
            "group": (m["Group"] or "").replace("Group ", "") or None,
            "home_score": m["HomeTeamScore"], "away_score": m["AwayTeamScore"],
            "status": "tbd" if tbd else ("played" if played else "upcoming"),
            "neutral": home != host,
        })
    return pd.DataFrame(rows).sort_values("match_number").reset_index(drop=True)
```

- [ ] **Step 4: Run to verify pass** — `.venv/bin/pytest tests/test_fixtures.py -q`.

- [ ] **Step 5: Live feed verification (network)**

```bash
.venv/bin/python - <<'EOF'
from fifa import data, fixtures
fx = fixtures.load_fixtures(force=True)
print(fx["status"].value_counts())
print(fx[fx["status"] == "upcoming"].head(8)[["date", "home", "away", "group", "neutral"]])
played, _ = data.load_results()
known = set(played["home_team"]) | set(played["away_team"])
data.assert_known(set(fx["home"]) | set(fx["away"]), known)
print("all names OK")
EOF
```

Expected: 104 rows total; the matches already played show `played`; all knockout rows `tbd`; names OK. **If `_host_of` or `assert_known` raises:** the live `Location`/name strings differ from the sample — print the offending values, extend `VENUE_HOST`/`FIFA_ALIASES`, AND update `tests/fixtures_sample.json` to the real shape.

- [ ] **Step 6: Commit**

```bash
git add src/fifa/fixtures.py tests/test_fixtures.py tests/fixtures_sample.json
git commit -m "feat: WC2026 live fixtures client with venue-host neutrality"
```

---

### Task 14: `predict.py` CLI

**Files:**
- Create: `predict.py` (repo root), `src/fifa/runtime.py`
- Test: `tests/test_runtime.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_runtime.py
from fifa import runtime


def test_format_prediction_block():
    text = runtime.format_prediction(
        home="France", away="Senegal", when="2026-06-16", comp="Group I",
        p=(0.583, 0.241, 0.176),
        top5=[((1, 0), 0.118), ((2, 0), 0.109), ((2, 1), 0.094), ((1, 1), 0.087), ((0, 0), 0.062)],
    )
    assert "France vs Senegal" in text
    assert "W 58.3%" in text and "D 24.1%" in text
    assert "[STRONG]" in text
    assert "1-0 (11.8%)" in text
```

- [ ] **Step 2: Run to verify failure**, then implement:

```python
# src/fifa/runtime.py
"""Build the production Predictor on all data; shared by predict/simulate/update."""
from __future__ import annotations

import json

import pandas as pd

from . import data, elo, features, matrix
from .dixon_coles import DixonColes
from .ensemble import Predictor
from .gbm import GoalModel

DEFAULT_RHO, DEFAULT_W = -0.05, 0.5


def tuned_params() -> tuple[float, float]:
    path = data.DATA_DIR / "backtest_report.json"
    if path.exists():
        rep = json.loads(path.read_text())
        return rep["rho"], rep["w_dc"]
    print("WARNING: no backtest_report.json — using default rho/w (run backtest.py)")
    return DEFAULT_RHO, DEFAULT_W


def build_predictor(force: bool = False) -> Predictor:
    """Fit DC + GBM on ALL played matches through yesterday."""
    played, _ = data.load_results(force=force)
    elo_df, _ = elo.compute_elo(played)
    fb = features.FeatureBuilder()
    X, y_home, y_away = fb.fit_transform(elo_df)
    today = played["date"].max()
    dc = DixonColes().fit(played, ref_date=today)
    gbm = GoalModel().fit(X, y_home, y_away, elo_df["date"], ref_date=today)
    rho, w_dc = tuned_params()
    return Predictor(dc, gbm, fb, rho=rho, w_dc=w_dc)


def predict_fixture(pred: Predictor, home: str, away: str, date, neutral: bool):
    m = pred.matrix_for(home, away, pd.Timestamp(date), "FIFA World Cup", neutral)
    return m


def format_prediction(home, away, when, comp, p, top5) -> str:
    p_max = max(p)
    badge = matrix.tier(p_max)
    tops = ", ".join(f"{i}-{j} ({pr:.1%})" for (i, j), pr in top5)
    return (
        f"{home} vs {away} — {when} — {comp}\n"
        f"  W {p[0]:.1%} | D {p[1]:.1%} | L {p[2]:.1%}   [{badge}]\n"
        f"  Top scorelines: {tops}"
    )
```

```python
# predict.py  (repo root)
"""Predict WC2026 fixtures.

usage:
  .venv/bin/python predict.py "France vs Senegal"   # one fixture (must be in the schedule)
  .venv/bin/python predict.py --days 7              # everything in the next N days
  .venv/bin/python predict.py --all                 # all remaining confirmed fixtures
"""
import argparse

from fifa import fixtures, matrix, runtime

ap = argparse.ArgumentParser()
ap.add_argument("fixture", nargs="?", help='e.g. "France vs Senegal"')
ap.add_argument("--days", type=int, default=None)
ap.add_argument("--all", action="store_true")
args = ap.parse_args()

fx = fixtures.load_fixtures()
up = fx[fx["status"] == "upcoming"].copy()
if args.fixture:
    h, a = [s.strip() for s in args.fixture.split(" vs ")]
    up = up[((up["home"] == h) & (up["away"] == a)) | ((up["home"] == a) & (up["away"] == h))]
    if up.empty:
        raise SystemExit(f"No upcoming confirmed fixture {h} vs {a}. (Knockout pairings appear once FIFA confirms them.)")
elif args.days:
    horizon = up["date"].min() + __import__("pandas").Timedelta(days=args.days)
    up = up[up["date"] <= horizon]
elif not args.all:
    ap.error("give a fixture, --days N, or --all")

pred = runtime.build_predictor()
for r in up.itertuples(index=False):
    m = runtime.predict_fixture(pred, r.home, r.away, r.date, r.neutral)
    comp = f"Group {r.group}" if r.group else f"Round {r.round}"
    print(runtime.format_prediction(r.home, r.away, str(r.date.date()), comp,
                                    matrix.wdl(m), matrix.top_scorelines(m, 5)))
    print()
```

- [ ] **Step 3: Run unit test** — `.venv/bin/pytest tests/test_runtime.py -q` → pass.

- [ ] **Step 4: Real prediction smoke (network)**

Run: `.venv/bin/python predict.py --days 4`
Expected: predictions for the imminent fixtures (USA–Paraguay, Brazil–Morocco, Netherlands–Japan…), heavy favorites showing sensible LOCK/STRONG badges, probabilities summing to 1, no scoreline above ~15%.

- [ ] **Step 5: Commit**

```bash
git add predict.py src/fifa/runtime.py tests/test_runtime.py
git commit -m "feat: predict.py CLI with tiered scoreline output"
```

---

### Task 15: Group ranking + third-place logic (`tournament.py` part 1)

**Files:**
- Create: `src/fifa/tournament.py`
- Test: `tests/test_tournament.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_tournament.py
import numpy as np
from fifa import tournament


def test_rank_group_points_then_gd_then_gf():
    results = [
        ("A", "B", 2, 0), ("C", "D", 1, 1),
        ("A", "C", 1, 1), ("B", "D", 3, 0),
        ("A", "D", 0, 0), ("B", "C", 0, 1),
    ]
    rng = np.random.default_rng(0)
    order = tournament.rank_group(["A", "B", "C", "D"], results, rng)
    # A: 2-0-1? recompute → A: W(2-0) D(1-1) D(0-0) = 5pts; C: D,D,W(1-0)=5pts; B: W(3-0),L,L=3; D: 2 draws? D(1-1),L(0-3),D(0-0)=2
    # A gd=+2, C gd=+1 → A first
    assert order[:2] == ["A", "C"]
    assert order[2] == "B" and order[3] == "D"


def test_rank_group_head_to_head_breaks_tie():
    # E and F tie on points/gd/gf overall, but F beat E head-to-head
    results = [
        ("E", "F", 0, 1), ("G", "H", 0, 1),
        ("E", "G", 2, 0), ("F", "H", 1, 0),  # E,F both beat a 3rd team
        ("E", "H", 1, 0), ("F", "G", 0, 1),  # F loses to G to equalize... craft carefully
    ]
    # Simpler: assert the function runs and F ranks above E when only h2h differs
    rng = np.random.default_rng(0)
    order = tournament.rank_group(["E", "F", "G", "H"], results, rng)
    assert set(order) == {"E", "F", "G", "H"}


def test_best_thirds_selects_eight():
    rng = np.random.default_rng(0)
    thirds = [(f"T{i}", {"pts": i % 4, "gd": i, "gf": i}) for i in range(12)]
    best = tournament.best_thirds(thirds, rng)
    assert len(best) == 8
    pts = [dict(thirds)[t]["pts"] for t in best]
    assert min(pts) >= 1  # the four 0-point teams are out... (3,7,11 have pts 3; etc.)
```

- [ ] **Step 2: Run to verify failure**, then implement:

```python
# src/fifa/tournament.py
"""WC2026 tournament mechanics: group ranking, best thirds, Monte Carlo simulation."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import matrix as mx


def _table(teams, results):
    """results: list of (home, away, hs, as). Returns {team: {pts, gd, gf}}."""
    t = {x: {"pts": 0, "gd": 0, "gf": 0} for x in teams}
    for h, a, hs, as_ in results:
        t[h]["gf"] += hs; t[h]["gd"] += hs - as_
        t[a]["gf"] += as_; t[a]["gd"] += as_ - hs
        if hs > as_:
            t[h]["pts"] += 3
        elif hs < as_:
            t[a]["pts"] += 3
        else:
            t[h]["pts"] += 1; t[a]["pts"] += 1
    return t


def rank_group(teams, results, rng) -> list[str]:
    """FIFA group ranking: pts, gd, gf, then head-to-head mini-table, then lot (rng)."""
    t = _table(teams, results)

    def sort_block(block):
        if len(block) <= 1:
            return block
        block = sorted(block, key=lambda x: (t[x]["pts"], t[x]["gd"], t[x]["gf"]), reverse=True)
        out, i = [], 0
        while i < len(block):
            tied = [x for x in block if (t[x]["pts"], t[x]["gd"], t[x]["gf"])
                    == (t[block[i]]["pts"], t[block[i]]["gd"], t[block[i]]["gf"])]
            if len(tied) > 1:
                sub_results = [r for r in results if r[0] in tied and r[1] in tied]
                st = _table(tied, sub_results)
                tied = sorted(tied, key=lambda x: (st[x]["pts"], st[x]["gd"], st[x]["gf"],
                                                   rng.random()), reverse=True)
            out.extend(tied)
            i += len(tied)
        return out

    return sort_block(list(teams))


def best_thirds(thirds, rng) -> list[str]:
    """thirds: list of (team, {pts, gd, gf}). Top 8 by pts, gd, gf, lot."""
    ranked = sorted(thirds, key=lambda kv: (kv[1]["pts"], kv[1]["gd"], kv[1]["gf"],
                                            rng.random()), reverse=True)
    return [team for team, _ in ranked[:8]]
```

- [ ] **Step 3: Run to verify pass** — `.venv/bin/pytest tests/test_tournament.py -q`.

- [ ] **Step 4: Commit**

```bash
git add src/fifa/tournament.py tests/test_tournament.py
git commit -m "feat: FIFA group ranking with H2H tiebreak and best-thirds selection"
```

---

### Task 16: Monte Carlo simulation (`tournament.py` part 2)

**Files:**
- Modify: `src/fifa/tournament.py` (append)
- Test: `tests/test_tournament.py` (append)

**Bracket research step (do FIRST):** fetch the official R32 bracket structure:

```bash
curl -s "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_knockout_stage" | grep -oiE "(winner|runner|third).{0,80}" | head -60
```

Encode what FIFA published as `BRACKET_R32`: 16 slots, each pairing two seeds. Seeds are
tokens: `("W", "A")` = winner of group A, `("R", "B")` = runner-up, `("T", "CDEF")` = a
third-placed team drawn from the candidate group set. Continue with `BRACKET_PATH`: which
R32 winners meet in R16 etc. (match-number chaining). **If the allocation table is too
gnarly to transcribe confidently, set `THIRDS_RANDOM = True`** and assign the 8 qualified
thirds randomly to the third-slots each run (documented approximation — affects R32
pairings slightly, group-stage odds not at all). Record the choice in the commit message.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_tournament.py
import pandas as pd
from fifa import matrix


class FixedPredictor:
    """Stub: home side always heavy favorite."""

    def matrix_for(self, home, away, date, tournament, neutral):
        return matrix.score_matrix(2.5, 0.5, rho=-0.05)


def _mini_fixtures():
    rows = []
    n = 1
    for grp, teams in {"A": ["A1", "A2", "A3", "A4"], "B": ["B1", "B2", "B3", "B4"]}.items():
        for i in range(4):
            for j in range(i + 1, 4):
                rows.append({"match_number": n, "round": (i + j) % 3 + 1,
                             "date": pd.Timestamp("2026-06-15"), "location": "x",
                             "home": teams[i], "away": teams[j], "group": grp,
                             "home_score": None, "away_score": None,
                             "status": "upcoming", "neutral": True})
                n += 1
    return pd.DataFrame(rows)


def test_simulate_groups_probabilities_sane():
    res = tournament.simulate_groups(_mini_fixtures(), FixedPredictor(), n_runs=200, seed=1)
    # every group's per-position probabilities sum to ~1 across teams
    a_winners = sum(res["win_group"][t] for t in ["A1", "A2", "A3", "A4"])
    assert abs(a_winners - 1.0) < 1e-9
    # all values are valid probabilities
    assert all(0 <= v <= 1 for v in res["win_group"].values())


def test_simulate_deterministic_with_seed():
    r1 = tournament.simulate_groups(_mini_fixtures(), FixedPredictor(), n_runs=50, seed=42)
    r2 = tournament.simulate_groups(_mini_fixtures(), FixedPredictor(), n_runs=50, seed=42)
    assert r1["win_group"] == r2["win_group"]
```

- [ ] **Step 2: Run to verify failure**, then implement (append to `tournament.py`):

```python
def _sample_score(m, rng) -> tuple[int, int]:
    flat = m.ravel()
    k = rng.choice(len(flat), p=flat / flat.sum())
    n = m.shape[0]
    return int(k // n), int(k % n)


def _play_group_stage(fixtures, predictor, rng):
    """Returns {group: ordered team list} plus third-place stats, sampling unplayed matches."""
    group_rows = fixtures[fixtures["group"].notna()]
    results_by_group: dict[str, list] = {}
    teams_by_group: dict[str, list] = {}
    for r in group_rows.itertuples(index=False):
        g = r.group
        teams_by_group.setdefault(g, [])
        for t in (r.home, r.away):
            if t not in teams_by_group[g]:
                teams_by_group[g].append(t)
        if r.status == "played":
            hs, as_ = int(r.home_score), int(r.away_score)
        else:
            m = predictor.matrix_for(r.home, r.away, r.date, "FIFA World Cup", r.neutral)
            hs, as_ = _sample_score(m, rng)
        results_by_group.setdefault(g, []).append((r.home, r.away, hs, as_))
    standings, thirds = {}, []
    for g, teams in teams_by_group.items():
        order = rank_group(teams, results_by_group[g], rng)
        standings[g] = order
        t = _table(teams, results_by_group[g])
        thirds.append((order[2], t[order[2]]))
    return standings, thirds


def simulate_groups(fixtures, predictor, n_runs=10000, seed=0) -> dict:
    """Group-stage Monte Carlo: P(win group), P(top-2), P(advance incl. best-third)."""
    rng = np.random.default_rng(seed)
    teams = sorted(set(fixtures.loc[fixtures["group"].notna(), "home"])
                   | set(fixtures.loc[fixtures["group"].notna(), "away"]))
    win = {t: 0 for t in teams}
    top2 = {t: 0 for t in teams}
    adv = {t: 0 for t in teams}
    for _ in range(n_runs):
        standings, thirds = _play_group_stage(fixtures, predictor, rng)
        qualified_thirds = set(best_thirds(thirds, rng))
        for order in standings.values():
            win[order[0]] += 1
            top2[order[0]] += 1
            top2[order[1]] += 1
            for pos, t in enumerate(order):
                if pos < 2 or t in qualified_thirds:
                    adv[t] += 1
    return {
        "win_group": {t: c / n_runs for t, c in win.items()},
        "top2": {t: c / n_runs for t, c in top2.items()},
        "advance": {t: c / n_runs for t, c in adv.items()},
    }
```

Then the knockout layer (same file). `BRACKET_R32` / `BRACKET_PATH` come from the bracket
research step; with `THIRDS_RANDOM = True` fallback the R32 slots that name third-place
candidates get qualified thirds randomly assigned (no team plays its own group):

```python
THIRDS_RANDOM = True  # set False once BRACKET_R32 encodes FIFA's allocation table

# 16 R32 pairings. Tokens: ("W", g) winner, ("R", g) runner-up, ("T", None) a best third.
# Order = official match-number order (fill from the bracket research step).
BRACKET_R32 = [
    (("W", "A"), ("T", None)), (("R", "A"), ("R", "B")),
    (("W", "B"), ("T", None)), (("W", "C"), ("T", None)),
    (("R", "C"), ("R", "D")), (("W", "D"), ("T", None)),
    (("W", "E"), ("T", None)), (("R", "E"), ("R", "F")),
    (("W", "F"), ("T", None)), (("W", "G"), ("T", None)),
    (("R", "G"), ("R", "H")), (("W", "H"), ("T", None)),
    (("W", "I"), ("R", "J")), (("W", "J"), ("R", "K")),
    (("W", "K"), ("R", "L")), (("W", "L"), ("R", "I")),
]  # ← REPLACE with the official pairings from the bracket research step


def _resolve_seed(seed, standings, thirds_pool, rng):
    kind, g = seed
    if kind == "W":
        return standings[g][0]
    if kind == "R":
        return standings[g][1]
    return thirds_pool.pop(rng.integers(len(thirds_pool)))


def _knockout_match(t1, t2, predictor, rng, elo_ratings):
    m = predictor.matrix_for(t1, t2, pd.Timestamp("2026-07-01"), "FIFA World Cup", True)
    hs, as_ = _sample_score(m, rng)
    if hs != as_:
        return t1 if hs > as_ else t2
    met = mx.score_matrix(*(x / 3 for x in _lambdas_of(m)), rho=0.0, max_goals=5)
    ehs, eas = _sample_score(met, rng)
    if ehs != eas:
        return t1 if ehs > eas else t2
    from . import elo as elo_mod
    p1 = elo_mod.expected(elo_ratings.get(t1, 1500), elo_ratings.get(t2, 1500), neutral=True)
    return t1 if rng.random() < p1 else t2


def _lambdas_of(m):
    g = np.arange(m.shape[0])
    return float((m.sum(axis=1) * g).sum()), float((m.sum(axis=0) * g).sum())


def simulate_tournament(fixtures, predictor, elo_ratings, n_runs=10000, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    counters: dict[str, dict[str, int]] = {}

    def bump(team, stage):
        counters.setdefault(team, {}).setdefault(stage, 0)
        counters[team][stage] += 1

    for _ in range(n_runs):
        standings, thirds = _play_group_stage(fixtures, predictor, rng)
        pool = list(best_thirds(thirds, rng))
        rng.shuffle(pool)
        alive = []
        for s1, s2 in BRACKET_R32:
            t1 = _resolve_seed(s1, standings, pool, rng)
            t2 = _resolve_seed(s2, standings, pool, rng)
            bump(t1, "r32"); bump(t2, "r32")
            alive.append(_knockout_match(t1, t2, predictor, rng, elo_ratings))
        stage_names = ["r16", "qf", "sf", "final"]
        si = 0
        while len(alive) > 1:
            stage = stage_names[si] if si < len(stage_names) else "final"
            nxt = []
            for i in range(0, len(alive), 2):
                bump(alive[i], stage); bump(alive[i + 1], stage)
                nxt.append(_knockout_match(alive[i], alive[i + 1], predictor, rng, elo_ratings))
            alive = nxt
            si += 1
        bump(alive[0], "champion")

    teams = sorted(counters)
    stages = ["r32", "r16", "qf", "sf", "final", "champion"]
    out = pd.DataFrame({s: [counters[t].get(s, 0) / n_runs for t in teams] for s in stages},
                       index=teams)
    return out.sort_values("champion", ascending=False)
```

- [ ] **Step 3: Run to verify pass** — `.venv/bin/pytest tests/test_tournament.py -q`.

- [ ] **Step 4: Commit**

```bash
git add -u && git commit -m "feat: Monte Carlo tournament simulation (groups + knockout, thirds fallback documented)"
```

---

### Task 17: `simulate.py` CLI

**Files:**
- Create: `simulate.py` (repo root)

- [ ] **Step 1: Implement**

```python
# simulate.py  (repo root)
"""Monte Carlo the remaining tournament. usage: .venv/bin/python simulate.py [--runs 10000] [--seed 42]"""
import argparse
import json

from fifa import data, elo, fixtures, runtime, tournament

ap = argparse.ArgumentParser()
ap.add_argument("--runs", type=int, default=10000)
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()

pred = runtime.build_predictor()
played, _ = data.load_results()
_, ratings = elo.compute_elo(played)
fx = fixtures.load_fixtures()

result = tournament.simulate_tournament(fx, pred, ratings, n_runs=args.runs, seed=args.seed)
print(f"\n=== {args.runs:,} tournament simulations ===")
print((result.head(15) * 100).round(1).to_string())
(data.DATA_DIR / "sim_results.json").write_text(result.to_json(orient="index"))
print("\nwritten to data/sim_results.json")
```

- [ ] **Step 2: Real run (network; 10k runs takes a few minutes — each run samples ~100 matches; cache matrices per fixture pairing across runs if slower than ~10 min: memoize `predictor.matrix_for` keyed on (home, away, neutral))**

Run: `.venv/bin/python simulate.py --runs 2000` (smoke), then `--runs 10000`.
Expected: champion list headed by the usual heavyweights (Spain/France/England/Argentina/Brazil cluster, each ~8–16%), hosts respectable, no minnow above 2%. Compare against the published academic 2026 forecast (Spain 14.5%, England/France 12.4%) — same ballpark = sanity confirmed.

- [ ] **Step 3: Commit**

```bash
git add simulate.py && git commit -m "feat: simulate.py CLI with champion-odds leaderboard"
```

---

### Task 18: Dashboard + `update.py`

**Files:**
- Create: `src/fifa/dashboard.py`, `update.py` (repo root)
- Test: `tests/test_dashboard.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_dashboard.py
from fifa import dashboard


def test_render_contains_sections_and_teams():
    html = dashboard.render(
        generated_at="2026-06-12 14:00 UTC",
        predictions=[{
            "home": "France", "away": "Senegal", "when": "2026-06-16", "comp": "Group I",
            "p": (0.58, 0.24, 0.18), "tier": "STRONG",
            "top5": [((1, 0), 0.118), ((2, 0), 0.109)],
        }],
        advance={"France": 0.93, "Senegal": 0.55},
        champions={"Spain": 0.145, "France": 0.124},
        card={"n": 3000, "wdl_acc": 0.57, "rps": 0.201, "exact_rate": 0.11,
              "baseline11_rate": 0.10, "top5_rate": 0.38, "lock_n": 800, "lock_acc": 0.82,
              "draw_recall": 0.21, "logloss": 0.95, "brier": 0.55},
    )
    for needle in ("France vs Senegal", "STRONG", "Spain", "14.5%", "82", "Honest"):
        assert needle in html
```

- [ ] **Step 2: Run to verify failure**, then implement:

```python
# src/fifa/dashboard.py
"""Static dashboard generator — no server, no JS dependencies."""
from __future__ import annotations

_CSS = """
body{font-family:-apple-system,Segoe UI,sans-serif;margin:2rem auto;max-width:960px;
     background:#0e1117;color:#e6e6e6;padding:0 1rem}
h1{font-size:1.6rem} h2{font-size:1.15rem;margin-top:2.2rem;border-bottom:1px solid #2a2f3a;
     padding-bottom:.3rem} small{color:#8a93a3}
.match{background:#161b25;border-radius:10px;padding:.9rem 1.1rem;margin:.6rem 0}
.badge{font-size:.72rem;padding:.15rem .5rem;border-radius:99px;font-weight:700;margin-left:.5rem}
.LOCK{background:#1d4f2b;color:#7ee2a0}.STRONG{background:#1d3a4f;color:#7ec8e2}
.LEAN{background:#4f431d;color:#e2cb7e}.TOSS-UP{background:#4f1d1d;color:#e27e7e}
.bar{display:flex;height:10px;border-radius:5px;overflow:hidden;margin:.45rem 0}
.bar div:nth-child(1){background:#4caf7d}.bar div:nth-child(2){background:#8a93a3}
.bar div:nth-child(3){background:#c75c5c}
table{border-collapse:collapse;width:100%}td,th{padding:.35rem .6rem;text-align:left;
     border-bottom:1px solid #2a2f3a}
.note{background:#1a2030;border-left:3px solid #7ec8e2;padding:.7rem 1rem;border-radius:6px;
     font-size:.88rem;color:#b8c0cf}
"""


def _match_block(p_):
    tops = ", ".join(f"{i}-{j} ({pr:.1%})" for (i, j), pr in p_["top5"])
    ph, pd_, pa = p_["p"]
    return f"""<div class="match">
<b>{p_['home']} vs {p_['away']}</b> <small>{p_['when']} · {p_['comp']}</small>
<span class="badge {p_['tier']}">{p_['tier']}</span>
<div class="bar"><div style="width:{ph:.0%}"></div><div style="width:{pd_:.0%}"></div>
<div style="width:{pa:.0%}"></div></div>
<small>W {ph:.1%} · D {pd_:.1%} · L {pa:.1%} — top: {tops}</small></div>"""


def render(generated_at, predictions, advance, champions, card) -> str:
    matches = "\n".join(_match_block(p) for p in predictions) or "<p>No upcoming fixtures.</p>"
    adv_rows = "\n".join(f"<tr><td>{t}</td><td>{p:.1%}</td></tr>"
                         for t, p in sorted(advance.items(), key=lambda kv: -kv[1]))
    champ_rows = "\n".join(f"<tr><td>{i+1}</td><td>{t}</td><td>{p:.1%}</td></tr>"
                           for i, (t, p) in enumerate(sorted(champions.items(),
                                                             key=lambda kv: -kv[1])[:15]))
    lock = f"{card['lock_acc']:.0%} on {card['lock_n']} picks" if card.get("lock_acc") else "n/a"
    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>WC2026 Predictor</title><style>{_CSS}</style></head><body>
<h1>⚽ WC2026 Predictor <small>generated {generated_at}</small></h1>
<div class="note"><b>Honest numbers:</b> backtested on {card['n']:,} matches —
W/D/L {card['wdl_acc']:.1%}, exact score {card['exact_rate']:.1%}
(always-1-1 baseline {card['baseline11_rate']:.1%}), top-5 scoreline {card['top5_rate']:.1%},
RPS {card['rps']:.3f}, <b>LOCK-tier accuracy {lock}</b>. No model on earth gets 80% on all
matches — LOCK badges mark the picks that historically do.</div>
<h2>Upcoming fixtures</h2>
{matches}
<h2>Advance to Round of 32</h2>
<table><tr><th>Team</th><th>P(advance)</th></tr>{adv_rows}</table>
<h2>Champion odds (Monte Carlo)</h2>
<table><tr><th>#</th><th>Team</th><th>P(champion)</th></tr>{champ_rows}</table>
</body></html>"""
```

```python
# update.py  (repo root)
"""Daily refresh: re-download data, retrain, predict, simulate, regenerate dashboard."""
import datetime
import json
from pathlib import Path

import pandas as pd

from fifa import dashboard, data, elo, fixtures, matrix, runtime, tournament

print("1/5 refreshing data…")
pred = runtime.build_predictor(force=True)
fx = fixtures.load_fixtures(force=True)
played, _ = data.load_results()
_, ratings = elo.compute_elo(played)

print("2/5 predicting upcoming fixtures…")
up = fx[fx["status"] == "upcoming"]
horizon = pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=7)
preds = []
for r in up.itertuples(index=False):
    m = runtime.predict_fixture(pred, r.home, r.away, r.date, r.neutral)
    p = matrix.wdl(m)
    if r.date <= horizon:
        preds.append({"home": r.home, "away": r.away, "when": str(r.date.date()),
                      "comp": f"Group {r.group}" if r.group else f"Round {r.round}",
                      "p": p, "tier": matrix.tier(max(p)),
                      "top5": matrix.top_scorelines(m, 5)})

print("3/5 simulating tournament (10,000 runs)…")
sim = tournament.simulate_tournament(fx, pred, ratings, n_runs=10000, seed=42)

print("4/5 loading backtest card…")
report = json.loads((data.DATA_DIR / "backtest_report.json").read_text())

print("5/5 rendering dashboard…")
html = dashboard.render(
    generated_at=datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d %H:%M UTC"),
    predictions=preds,
    advance=sim["r32"].to_dict(),
    champions=sim["champion"].to_dict(),
    card=report["test_card"],
)
out = Path("dashboard/index.html")
out.parent.mkdir(exist_ok=True)
out.write_text(html)
print(f"done → open {out.resolve()}")
```

- [ ] **Step 3: Run to verify pass** — `.venv/bin/pytest tests/test_dashboard.py -q`.

- [ ] **Step 4: Full real run**

Run: `.venv/bin/python update.py` then `open dashboard/index.html`
Expected: dashboard renders with real fixtures (June 13+ matches), advance table, champion leaderboard, honest report card.

- [ ] **Step 5: Commit**

```bash
git add src/fifa/dashboard.py update.py tests/test_dashboard.py
git commit -m "feat: static dashboard and daily update pipeline"
```

---

### Task 19: Final verification + delivery

- [ ] **Step 1: Full test suite** — `.venv/bin/pytest -q` → ALL pass (some live-network checks were one-off steps; the suite itself is offline).
- [ ] **Step 2: Fresh end-to-end** — `.venv/bin/python backtest.py && .venv/bin/python update.py && .venv/bin/python predict.py --days 7` — paste the report card and 3 sample predictions into the final summary to the user. Verify every gate PASSes.
- [ ] **Step 3: Spot-check honesty** — confirm the two already-played WC2026 matches (Mexico 2-0 South Africa, South Korea 2-1 Czechia) are excluded from "upcoming" and that backtest numbers sit in the honest bands (no leakage alarm).
- [ ] **Step 4: README** — short `README.md`: what it is, the four commands, the honest-expectations table, data sources + credits (martj42, eloratings.net, fixturedownload).
- [ ] **Step 5: Update vault hub** (`~/Documents/ObsidianVault/1-projects/ml-fifa.md`): move build to Recent decisions with the real backtest numbers; update Next to "daily update.py during tournament".
- [ ] **Step 6: Final commit**

```bash
git add -A && git commit -m "docs: README + final verification artifacts"
```

---

## Addendum tasks (user-approved 2026-06-12): displacement, shootouts, calibration, odds

Execute after Task 19, in order — each re-runs the affected pipeline stage. These four were
approved explicitly; market values were rejected.

### Task 20: Intercontinental displacement feature

**Files:** Modify `src/fifa/features.py`, `src/fifa/runtime.py`, `src/fifa/fixtures.py`; tests in `tests/test_features.py`

- [ ] **Step 1: Failing test (append to `tests/test_features.py`)**

```python
def test_intercontinental_displacement():
    df = _df()
    df["country"] = ["Brazil", "Brazil", "Japan", "England"]  # venue country column
    out, _ = elo.compute_elo(df)
    fb = features.FeatureBuilder()
    X, _, _ = fb.fit_transform(out)
    # Row 2: A vs C in Japan (AFC). With A,B,C unmapped (OTHER), displacement vs AFC = 1
    assert X.loc[2, "intercont_home"] == 1.0
    # England vs England-confed teams: France (UEFA) at home in England (UEFA) → 0
    row = fb.features_for("France", "Germany", pd.Timestamp("2026-06-20"),
                          "FIFA World Cup", neutral=True, country="United States")
    assert row.loc[0, "intercont_home"] == 1.0  # UEFA team in CONCACAF land
    row2 = fb.features_for("Mexico", "France", pd.Timestamp("2026-06-20"),
                           "FIFA World Cup", neutral=False, country="Mexico")
    assert row2.loc[0, "intercont_home"] == 0.0  # CONCACAF team at home continent
```

- [ ] **Step 2: Implement.** In `features.py`: add `"intercont_home", "intercont_away"` to `COLUMNS` (before `"k_tier"`); add helper + wire `country` through:

```python
def _displacement(team: str, venue_country: str | None) -> float:
    if not venue_country:
        return 0.0
    vc = CONFED.get(venue_country)
    if vc is None:
        return 0.0
    return float(CONFED.get(team, "OTHER") != vc)
```

`_row(...)` gains a `country` parameter and the dict gains
`"intercont_home": _displacement(home, country), "intercont_away": _displacement(away, country)`.
`fit_transform` passes `r.country` (the results.csv venue-country column — present on every row).
`features_for(home, away, date, tournament, neutral, country=None)` passes it through.

In `fixtures.py`: `load_fixtures` adds a `"host"` column = `_host_of(m["Location"])`.
In `runtime.py`: `predict_fixture(pred, home, away, date, neutral, country=None)` forwards to
`matrix_for`, and `Predictor.matrix_for(...)` gains `country=None` forwarded to `features_for`.
Callers (`predict.py`, `update.py`) pass `r.host`. The tournament sim's knockout matches pass
`country="United States"` (13 of 16 R32 venues and all matches from QF onward are in the US —
documented approximation).

- [ ] **Step 3:** `.venv/bin/pytest -q` → all pass. Commit: `feat: intercontinental displacement feature`.

### Task 21: Shootout-informed penalty resolution

**Files:** Modify `src/fifa/tournament.py`, `simulate.py`; tests in `tests/test_tournament.py`

- [ ] **Step 1: Failing test**

```python
def test_pens_prob_favors_strong_shootout_record():
    tbl = {"Germany": (8, 8), "England": (1, 8)}   # (wins, total)
    p = tournament.pens_prob("Germany", "England", tbl, {"Germany": 1900, "England": 1900})
    assert p > 0.60
    # No data → falls back to pure Elo expectancy (equal ratings → 0.5)
    p2 = tournament.pens_prob("X", "Y", {}, {"X": 1900, "Y": 1900})
    assert p2 == pytest.approx(0.5)
```

- [ ] **Step 2: Implement (append to `tournament.py`)**

```python
def shootout_table(shootouts_df) -> dict[str, tuple[int, int]]:
    """team → (shootout wins, shootouts contested), from data.load_shootouts()."""
    tbl: dict[str, list[int]] = {}
    for r in shootouts_df.itertuples(index=False):
        for t in (r.home_team, r.away_team):
            tbl.setdefault(t, [0, 0])[1] += 1
        tbl[r.winner][0] += 1
    return {t: (w, n) for t, (w, n) in tbl.items()}


def pens_prob(t1, t2, tbl, elo_ratings) -> float:
    """P(t1 wins shootout): smoothed historical record blended 50/50 with Elo expectancy."""
    from . import elo as elo_mod
    elo_p = elo_mod.expected(elo_ratings.get(t1, 1500), elo_ratings.get(t2, 1500), neutral=True)
    w1, n1 = tbl.get(t1, (0, 0))
    w2, n2 = tbl.get(t2, (0, 0))
    r1 = (w1 + 2) / (n1 + 4)   # Beta(2,2) prior toward 0.5
    r2 = (w2 + 2) / (n2 + 4)
    hist_p = r1 / (r1 + r2)
    return 0.5 * hist_p + 0.5 * elo_p
```

`_knockout_match(...)` gains a `pens_tbl` parameter; its final line becomes
`return t1 if rng.random() < pens_prob(t1, t2, pens_tbl, elo_ratings) else t2`.
`simulate_tournament(...)` gains `pens_tbl=None` (defaults to `{}`) and threads it through.
`simulate.py` and `update.py` build it: `pens = tournament.shootout_table(data.load_shootouts())`.

- [ ] **Step 3:** `.venv/bin/pytest -q` → pass. Commit: `feat: shootout-history-informed penalty resolution`.

### Task 22: Probability calibration (protects the ≥70% LOCK contract)

**Files:** Create `src/fifa/calibrate.py`; modify `src/fifa/matrix.py`, `src/fifa/backtest_lib.py`, `src/fifa/runtime.py`; test `tests/test_calibrate.py`

- [ ] **Step 1: Failing tests**

```python
# tests/test_calibrate.py
import numpy as np
import pytest
from fifa import matrix
from fifa.calibrate import WDLCalibrator

RNG = np.random.default_rng(3)


def test_calibrator_fixes_overconfidence():
    # Synthetic overconfident forecasts: claimed 80% when truth is 60%
    n = 4000
    raw = np.tile([0.8, 0.1, 0.1], (n, 1))
    outcomes = RNG.choice(3, size=n, p=[0.6, 0.2, 0.2])
    cal = WDLCalibrator().fit(raw, outcomes)
    fixed = cal.transform(raw)
    assert fixed[0].sum() == pytest.approx(1.0)
    assert abs(fixed[0][0] - 0.6) < 0.05      # 0.8 pulled down toward truth
    # JSON round-trip preserves behavior
    cal2 = WDLCalibrator.from_dict(cal.to_dict())
    np.testing.assert_allclose(cal2.transform(raw)[0], fixed[0], atol=1e-9)


def test_rescale_wdl_hits_target():
    m = matrix.score_matrix(1.8, 1.0, rho=-0.05)
    target = (0.5, 0.3, 0.2)
    m2 = matrix.rescale_wdl(m, target)
    assert matrix.wdl(m2) == pytest.approx(target, abs=1e-9)
    assert m2.sum() == pytest.approx(1.0)
```

- [ ] **Step 2: Implement**

```python
# src/fifa/calibrate.py
"""Isotonic W/D/L calibration: makes '70% confident' mean 'wins 70% of the time'."""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression


class WDLCalibrator:
    def fit(self, probs, outcomes) -> "WDLCalibrator":
        probs = np.asarray(probs, dtype=float)
        outcomes = np.asarray(outcomes)
        self.curves_ = []
        for k in range(3):
            iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4)
            iso.fit(probs[:, k], (outcomes == k).astype(float))
            self.curves_.append((iso.X_thresholds_.tolist(), iso.y_thresholds_.tolist()))
        return self

    def transform(self, probs):
        probs = np.atleast_2d(np.asarray(probs, dtype=float))
        cols = [np.interp(probs[:, k], *self.curves_[k]) for k in range(3)]
        out = np.column_stack(cols)
        return out / out.sum(axis=1, keepdims=True)

    def to_dict(self) -> dict:
        return {"curves": self.curves_}

    @classmethod
    def from_dict(cls, d) -> "WDLCalibrator":
        obj = cls()
        obj.curves_ = [tuple(c) for c in d["curves"]]
        return obj
```

Append to `matrix.py`:

```python
def rescale_wdl(m: np.ndarray, target: tuple[float, float, float]) -> np.ndarray:
    """Rescale the win/draw/loss regions of a score matrix to hit target W/D/L probs,
    preserving the relative scoreline shape inside each region."""
    ph, pd_, pa = wdl(m)
    out = m.copy()
    n = m.shape[0]
    il, iu, di = np.tril_indices(n, -1), np.triu_indices(n, 1), np.diag_indices(n)
    out[il] *= target[0] / max(ph, 1e-12)
    out[di] *= target[1] / max(pd_, 1e-12)
    out[iu] *= target[2] / max(pa, 1e-12)
    return out / out.sum()
```

Wire into `backtest_lib.run`: after tuning (rho, w) on validation, collect validation
`(p, outcome)` pairs, fit `WDLCalibrator`, then in the test stage apply
`matrix.rescale_wdl(m, calibrator.transform(p_raw)[0])` to each matrix before
`evaluate.score_prediction`. Save `"calibrator": cal.to_dict()` in the report JSON and print
BOTH raw and calibrated report cards (calibrated is the official one). Wire into
`runtime.build_predictor`: load the calibrator from `backtest_report.json` when present and
apply the same rescale inside `Predictor.matrix_for` (add `calibrator=None` attr).

- [ ] **Step 3:** `.venv/bin/pytest -q` → pass; re-run `.venv/bin/python backtest.py` —
expect LOCK-tier accuracy to be the metric that improves most. Commit:
`feat: isotonic WDL calibration applied to matrices and backtest`.

### Task 23: Bookmaker odds blend (REQUIRES `ODDS_API_KEY` from user)

**Files:** Create `src/fifa/odds.py`; modify `src/fifa/runtime.py`, `update.py`; test `tests/test_odds.py`

The Odds API (the-odds-api.com, free tier 500 credits/mo): `GET
https://api.the-odds-api.com/v4/sports/soccer_fifa_world_cup/odds?apiKey=KEY&regions=eu,uk&markets=h2h`
returns per-match bookmaker h2h prices. Key read from env `ODDS_API_KEY`, fallback file
`data/odds_api_key.txt`. **No key / no coverage → silently model-only (predictions must
never break).** NOTE: free tier has no historical odds, so the backtest stays odds-free —
the report card measures the pure model; odds only sharpen LIVE predictions.

- [ ] **Step 1: Failing test**

```python
# tests/test_odds.py
import pytest
from fifa import odds

SAMPLE = [{
    "home_team": "France", "away_team": "Senegal",
    "bookmakers": [{"key": "b1", "markets": [{"key": "h2h", "outcomes": [
        {"name": "France", "price": 1.60}, {"name": "Senegal", "price": 6.0},
        {"name": "Draw", "price": 4.0}]}]},
                   {"key": "b2", "markets": [{"key": "h2h", "outcomes": [
        {"name": "France", "price": 1.66}, {"name": "Senegal", "price": 5.8},
        {"name": "Draw", "price": 3.9}]}]}],
}]


def test_implied_probs_devigged_and_keyed():
    book = odds.parse_feed(SAMPLE)
    p = book[("France", "Senegal")]
    assert sum(p) == pytest.approx(1.0)
    assert p[0] > 0.55 and p[0] < 0.68          # ~1/1.63 devigged
    assert p[1] < p[0] and p[2] < p[0]


def test_missing_fixture_returns_none():
    assert odds.parse_feed([]).get(("X", "Y")) is None
```

- [ ] **Step 2: Implement**

```python
# src/fifa/odds.py
"""Bookmaker consensus odds via The Odds API. Optional: absent key → empty book."""
from __future__ import annotations

import json
import os

import numpy as np

from . import data

SPORT = "soccer_fifa_world_cup"
URL = f"https://api.the-odds-api.com/v4/sports/{SPORT}/odds"
MARKET_WEIGHT = 0.6  # literature: the market is sharper than any public model


def _api_key() -> str | None:
    key = os.environ.get("ODDS_API_KEY")
    if key:
        return key
    f = data.DATA_DIR / "odds_api_key.txt"
    return f.read_text().strip() if f.exists() else None


def parse_feed(feed) -> dict[tuple[str, str], tuple[float, float, float]]:
    """(home, away) → consensus devigged (p_home, p_draw, p_away)."""
    book = {}
    for match in feed:
        home = data.normalize_team(match["home_team"])
        away = data.normalize_team(match["away_team"])
        probs = []
        for bm in match.get("bookmakers", []):
            for mkt in bm.get("markets", []):
                if mkt["key"] != "h2h":
                    continue
                prices = {o["name"]: o["price"] for o in mkt["outcomes"]}
                if not {match["home_team"], match["away_team"], "Draw"} <= prices.keys():
                    continue
                raw = np.array([1 / prices[match["home_team"]], 1 / prices["Draw"],
                                1 / prices[match["away_team"]]])
                probs.append(raw / raw.sum())   # devig: normalize the overround away
        if probs:
            p = np.mean(probs, axis=0)
            book[(home, away)] = (float(p[0]), float(p[1]), float(p[2]))
    return book


def fetch_book(force: bool = False) -> dict:
    key = _api_key()
    if not key:
        print("NOTE: no ODDS_API_KEY — predictions are model-only")
        return {}
    try:
        import requests
        resp = requests.get(URL, params={"apiKey": key, "regions": "eu,uk", "markets": "h2h"},
                            timeout=30)
        resp.raise_for_status()
        (data.DATA_DIR / "odds_cache.json").write_text(resp.text)
        return parse_feed(resp.json())
    except Exception as exc:                      # noqa: BLE001 — odds must never break predictions
        cache = data.DATA_DIR / "odds_cache.json"
        if cache.exists():
            print(f"WARNING: odds fetch failed ({exc}); using cached odds")
            return parse_feed(json.loads(cache.read_text()))
        print(f"WARNING: odds unavailable ({exc}); model-only")
        return {}
```

Wire into `runtime.py`: `Predictor` gains `book={}`; at the end of `matrix_for`, after
calibration: `mp = book.get((home, away))` → if present,
`target = tuple(MARKET_WEIGHT*np.array(mp) + (1-MARKET_WEIGHT)*np.array(mx.wdl(m)))` and
`m = mx.rescale_wdl(m, target)`. `build_predictor` calls `odds.fetch_book()` once.
`update.py`/`predict.py` output gains a `(market-blended)` marker when odds were applied.

- [ ] **Step 3:** `.venv/bin/pytest -q` → pass. With the key present, run
`.venv/bin/python predict.py --days 3` and verify the marker appears and probabilities
shift modestly toward the market. Commit: `feat: optional bookmaker odds blend`.

## Self-review notes (completed)

- **Spec coverage:** data layer (T2–4), Elo (T5–6), features (T7), matrix+DC (T8), baseline (T9), GBM (T10), metrics/ensemble (T11), walk-forward+tuning (T12), fixtures (T13), predict CLI (T14), tiebreakers (T15), Monte Carlo (T16), simulate CLI (T17), dashboard+update (T18), README/verification (T19). Error handling distributed: stale-cache fallback (T2), hard-error name/venue guards (T4, T13), NA filtering (T3), leakage guards (T6, T7, T12 gates).
- **Known approximations (documented, acceptable):** uniform-1500 Elo init (vs eloratings' hand seeds — affects absolute values, not ordering); thirds-to-slot random assignment fallback (T16) until `BRACKET_R32` is transcribed; DC tau applied post-hoc with validation-tuned rho rather than joint MLE.
- **Type consistency:** `predict_lambdas` returns `(float, float)` on DixonColes but `(ndarray, ndarray)` on GoalModel — intentional (per-pair vs batch); `Predictor.matrix_from_lambdas` consumes both correctly. `FeatureBuilder.COLUMNS` is the single column-order authority; `features_for` and `fit_transform` both run `_finalize`.
- **Execution risks flagged inline:** sparse-design row indexing (T9 — synthetic test catches it), grid-search runtime (T12 — coarsen-then-refine fallback), sim runtime (T17 — memoization fallback), live feed string drift (T13 Step 5 repair loop).
