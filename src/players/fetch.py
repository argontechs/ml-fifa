"""FBref player-season data via soccerdata; columns pinned from the 2026-06-12 probe.

Identity model (audit fix): a player is keyed by (player, nation_raw, born) — bare
(player, nation) merged genuinely different people (both Brazilian Alissons), and
keep-max-minutes deduping dropped 70+ real players whose seasons were split across
transfer stints. Stints are now SUMMED per identity before per-90s are computed.
"""
from __future__ import annotations

import pandas as pd

from sentiment.match_window import TRIGRAPH

CANONICAL = ["npg90", "ast90", "sh90", "sot_pct", "conv", "crs90", "int90",
             "tklw90", "fld90", "fls90", "off90", "card90"]

_CODE_TO_NATION = {code: name for name, code in TRIGRAPH.items()}

LEAGUE = "Big 5 European Leagues Combined"
SEASON = "2025-2026"

_COUNT_COLS = ["gls", "ast", "pk", "yc", "rc", "sh", "sot", "crs", "int_",
               "tklw", "fld", "fls", "off", "minutes", "nineties"]
IDENTITY = ["player", "nation_raw", "born"]


def nation_of(raw) -> str | None:
    """FBref nation strings look like 'fr FRA' — map the trigraph to our dataset name."""
    if not isinstance(raw, str) or not raw.strip():
        return None
    code = raw.split()[-1].upper()
    return _CODE_TO_NATION.get(code)


def _flat(df: pd.DataFrame, mapping: dict[tuple, str]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for col, name in mapping.items():
        out[name] = df[col] if col in df.columns else pd.NA
    return out


def _extract_stints(readers: dict) -> pd.DataFrame:
    """One row per (team, player) stint with RAW season totals — no per-90s yet."""
    std = _flat(readers["standard"](), {
        ("nation", ""): "nation_raw", ("pos", ""): "pos", ("age", ""): "age",
        ("born", ""): "born",
        ("Playing Time", "Min"): "minutes", ("Playing Time", "90s"): "nineties",
        ("Performance", "Gls"): "gls", ("Performance", "Ast"): "ast",
        ("Performance", "PK"): "pk", ("Performance", "CrdY"): "yc",
        ("Performance", "CrdR"): "rc",
    })
    sht = _flat(readers["shooting"](), {
        ("Standard", "Sh"): "sh", ("Standard", "SoT"): "sot",
    })
    msc = _flat(readers["misc"](), {
        ("Performance", "Crs"): "crs", ("Performance", "Int"): "int_",
        ("Performance", "TklW"): "tklw", ("Performance", "Fld"): "fld",
        ("Performance", "Fls"): "fls", ("Performance", "Off"): "off",
    })
    df = std.join(sht, how="left").join(msc, how="left").reset_index()
    for c in _COUNT_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    df["pos"] = df["pos"].fillna("")
    df["born"] = df["born"].astype(str).str.strip().replace({"<NA>": "", "nan": ""})
    return df[["player", "team", "nation_raw", "pos", "age", "born"] + _COUNT_COLS]


def aggregate_stints(stints: pd.DataFrame) -> pd.DataFrame:
    """Sum stints per identity, then compute per-90s/ratios from season totals.
    Display columns (team/pos/age) come from the biggest-minutes stint."""
    stints = stints.copy()
    stints["born"] = stints["born"].fillna("")
    summed = stints.groupby(IDENTITY, dropna=False)[_COUNT_COLS].sum().reset_index()
    display = (stints.sort_values("minutes", ascending=False)
               .drop_duplicates(subset=IDENTITY)[IDENTITY + ["team", "pos", "age"]])
    df = summed.merge(display, on=IDENTITY, how="left")

    n = df["nineties"].clip(lower=0.1)
    sh = df["sh"]
    out = pd.DataFrame({
        "player": df["player"], "team": df["team"],
        "nation": df["nation_raw"].map(nation_of),
        "pos": df["pos"], "age": df["age"],
        "minutes": df["minutes"], "nineties": df["nineties"],
        "npg90": (df["gls"] - df["pk"]) / n,
        "ast90": df["ast"] / n,
        "sh90": sh / n,
        "sot_pct": (df["sot"] / sh * 100).where(sh > 0),
        "conv": (df["gls"] / sh).where(sh > 0),
        "crs90": df["crs"] / n,
        "int90": df["int_"] / n,
        "tklw90": df["tklw"] / n,
        "fld90": df["fld"] / n,
        "fls90": df["fls"] / n,
        "off90": df["off"] / n,
        "card90": (df["yc"] + 2 * df["rc"]) / n,
    })
    return out


def assemble(readers: dict) -> pd.DataFrame:
    """Single-league pull → identity-aggregated canonical frame."""
    return aggregate_stints(_extract_stints(readers))


def default_readers() -> dict:
    """Real network readers (cached on disk by soccerdata)."""
    import soccerdata

    fb = soccerdata.FBref(leagues=LEAGUE, seasons=SEASON)
    return {st: (lambda st=st: fb.read_player_season_stats(stat_type=st))
            for st in ("standard", "shooting", "misc")}


# WC-relevant leagues beyond the Big 5 (custom entries in ~/soccerdata/config/league_dict.json).
# Calendar-year leagues use their latest COMPLETE season for full-season profiles.
EXTRA_LEAGUES: list[tuple[str, str]] = [
    ("BRA-Serie A", "2025"),
    ("USA-MLS", "2025"),
    ("SAU-Pro League", "2025-2026"),
    ("MEX-Liga MX", "2025-2026"),
    ("NED-Eredivisie", "2025-2026"),
    ("POR-Primeira Liga", "2025-2026"),
]


def assemble_multi(extra: list[tuple[str, str]] = None) -> pd.DataFrame:
    """All leagues' stints concatenated, then ONE identity aggregation — so a player
    who moved between leagues mid-season gets a single summed profile."""
    import soccerdata

    frames = [_extract_stints(default_readers())]
    for lg, season in (EXTRA_LEAGUES if extra is None else extra):
        try:
            fb = soccerdata.FBref(leagues=lg, seasons=season)
            readers = {st: (lambda st=st, fb=fb: fb.read_player_season_stats(stat_type=st))
                       for st in ("standard", "shooting", "misc")}
            part = _extract_stints(readers)
            frames.append(part)
            print(f"   + {lg} {season}: {len(part)} stints")
        except Exception as exc:  # noqa: BLE001 — one broken league must not sink the page
            print(f"   ! {lg} skipped ({exc})")
    return aggregate_stints(pd.concat(frames, ignore_index=True))
