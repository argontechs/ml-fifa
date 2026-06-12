"""Feature matrix: minutes filter, GK exclusion, position-grouped standardization."""
from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import StandardScaler

from .fetch import CANONICAL

MIN_MINUTES = 900


def pos_group(pos: str) -> str:
    p = (pos or "").upper()
    if p.startswith("GK"):
        return "GK"
    for g in ("DF", "MF", "FW"):
        if g in p.split(",")[0]:
            return g
    return "MF"


def build_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (X_scaled, meta) for outfield players with enough minutes."""
    d = df.copy()
    d["pos_group"] = d["pos"].map(pos_group)
    d = d[(d["minutes"] >= MIN_MINUTES) & (d["pos_group"] != "GK")].reset_index(drop=True)
    X = d[CANONICAL].astype(float)
    # impute within position group, then scale within position group
    for g, idx in d.groupby("pos_group").groups.items():
        block = X.loc[idx]
        block = block.fillna(block.median(numeric_only=True)).fillna(0.0)
        X.loc[idx] = StandardScaler().fit_transform(block)
    meta = d[["player", "team", "nation", "pos", "pos_group", "age", "minutes", "nineties"]
             + CANONICAL]
    return X, meta
