"""FBref player-season data via soccerdata; columns pinned from the 2026-06-12 probe."""
from __future__ import annotations

import pandas as pd

from sentiment.match_window import TRIGRAPH

CANONICAL = ["npg90", "ast90", "sh90", "sot_pct", "conv", "crs90", "int90",
             "tklw90", "fld90", "fls90", "off90", "card90"]

_CODE_TO_NATION = {code: name for name, code in TRIGRAPH.items()}

LEAGUE = "Big 5 European Leagues Combined"
SEASON = "2025-2026"


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


def assemble(readers: dict) -> pd.DataFrame:
    """readers: stat_type → callable returning the raw FBref frame. Returns canonical rows."""
    std = _flat(readers["standard"](), {
        ("nation", ""): "nation_raw", ("pos", ""): "pos", ("age", ""): "age",
        ("Playing Time", "Min"): "minutes", ("Playing Time", "90s"): "nineties",
        ("Performance", "Gls"): "gls", ("Performance", "Ast"): "ast",
        ("Performance", "PK"): "pk", ("Performance", "CrdY"): "yc",
        ("Performance", "CrdR"): "rc",
    })
    sht = _flat(readers["shooting"](), {
        ("Standard", "Sh"): "sh", ("Standard", "SoT%"): "sot_pct", ("Standard", "G/Sh"): "conv",
    })
    msc = _flat(readers["misc"](), {
        ("Performance", "Crs"): "crs", ("Performance", "Int"): "int_",
        ("Performance", "TklW"): "tklw", ("Performance", "Fld"): "fld",
        ("Performance", "Fls"): "fls", ("Performance", "Off"): "off",
    })
    df = std.join(sht, how="left").join(msc, how="left").reset_index()

    n = df["nineties"].astype(float).clip(lower=0.1)
    per90 = {
        "npg90": (df["gls"].fillna(0) - df["pk"].fillna(0)) / n,
        "ast90": df["ast"].fillna(0) / n,
        "sh90": df["sh"].fillna(0) / n,
        "crs90": df["crs"].fillna(0) / n,
        "int90": df["int_"].fillna(0) / n,
        "tklw90": df["tklw"].fillna(0) / n,
        "fld90": df["fld"].fillna(0) / n,
        "fls90": df["fls"].fillna(0) / n,
        "off90": df["off"].fillna(0) / n,
        "card90": (df["yc"].fillna(0) + 2 * df["rc"].fillna(0)) / n,
    }
    out = pd.DataFrame({
        "player": df["player"], "team": df["team"],
        "nation": df["nation_raw"].map(nation_of),
        "pos": df["pos"].fillna(""), "age": df["age"],
        "minutes": pd.to_numeric(df["minutes"], errors="coerce").fillna(0),
        "nineties": df["nineties"].astype(float),
        "sot_pct": pd.to_numeric(df["sot_pct"], errors="coerce"),
        "conv": pd.to_numeric(df["conv"], errors="coerce"),
        **per90,
    })
    return out


def default_readers() -> dict:
    """Real network readers (cached on disk by soccerdata)."""
    import soccerdata

    fb = soccerdata.FBref(leagues=LEAGUE, seasons=SEASON)
    return {st: (lambda st=st: fb.read_player_season_stats(stat_type=st))
            for st in ("standard", "shooting", "misc")}
