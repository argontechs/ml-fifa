"""Playing-style archetypes: k-means WITHIN each position group (features are
position-relative z-scores, so cross-position clusters mislead — a goal-scoring
centre-back is not a striker). Role-specific naming tables."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# per-role: centroid signature overlap (≥2 of top-3 features) → archetype name
GROUP_ARCHETYPES: dict[str, list[tuple[set, str]]] = {
    "FW": [
        ({"npg90", "sh90", "conv"}, "Penalty-box striker"),
        ({"npg90", "sh90", "sot_pct"}, "Penalty-box striker"),
        ({"ast90", "crs90", "fld90"}, "Wide creator"),
        ({"ast90", "crs90", "off90"}, "Wide creator"),
        ({"ast90", "npg90", "fld90"}, "Complete forward"),
        ({"tklw90", "int90", "fls90"}, "Pressing forward"),
        ({"fld90", "fls90", "card90"}, "Physical forward"),
    ],
    "MF": [
        ({"npg90", "sh90", "conv"}, "Goal-scoring midfielder"),
        ({"npg90", "sh90", "sot_pct"}, "Goal-scoring midfielder"),
        ({"ast90", "crs90", "fld90"}, "Creator"),
        ({"ast90", "crs90", "off90"}, "Wide midfielder"),
        ({"tklw90", "int90", "fls90"}, "Ball-winner"),
        ({"tklw90", "int90", "card90"}, "Ball-winner"),
        ({"fls90", "card90", "fld90"}, "Destroyer"),
    ],
    "DF": [
        ({"crs90", "ast90", "off90"}, "Attacking full-back"),
        ({"crs90", "ast90", "fld90"}, "Attacking full-back"),
        ({"npg90", "conv", "sh90"}, "Set-piece threat"),
        ({"npg90", "conv", "sot_pct"}, "Set-piece threat"),
        ({"tklw90", "int90", "fls90"}, "Stopper"),
        ({"tklw90", "int90", "card90"}, "Stopper"),
        ({"int90", "card90", "fls90"}, "Last-line defender"),
    ],
}


def fit(X: pd.DataFrame, k_range=range(6, 13), seed: int = 42) -> dict:
    best = None
    arr = X.to_numpy()
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(arr)
        score = silhouette_score(arr, km.labels_)
        if best is None or score > best["sil"]:
            best = {"k": k, "sil": score, "labels": km.labels_, "centers": km.cluster_centers_}
    return best


def centroid_signature(center: np.ndarray, cols: list[str], top: int = 3) -> tuple[str, ...]:
    order = np.argsort(center)[::-1][:top]
    return tuple(cols[i] for i in order)


_FEATURE_PHRASES = {
    "npg90": "goal threat", "ast90": "creator", "sh90": "shot volume",
    "sot_pct": "shot accuracy", "conv": "clinical finishing", "crs90": "crossing",
    "int90": "interceptor", "tklw90": "tackler", "fld90": "foul magnet",
    "fls90": "physical edge", "off90": "line-runner", "card90": "aggressor",
}


def label_for(signature: tuple[str, ...], group: str = "MF") -> str:
    sig = set(signature)
    for proto, name in GROUP_ARCHETYPES.get(group, []):
        if len(sig & proto) >= 2:
            return name
    # human fallback — raw column names must never reach the public page (audit)
    a, b = (_FEATURE_PHRASES.get(s, s.replace("90", "")) for s in signature[:2])
    return f"{a.capitalize()} · {b}"


def name_clusters(centers: np.ndarray, cols: list[str], group: str = "MF") -> dict[int, str]:
    names: dict[int, str] = {}
    for i, c in enumerate(centers):
        sig = centroid_signature(c, cols)
        base = label_for(sig, group=group)
        k = 1
        while base in names.values():  # loop until unique (audit: single-pass collided)
            extra = _FEATURE_PHRASES.get(sig[min(k, len(sig) - 1)], "variant")
            base = f"{label_for(sig, group=group)} ({extra})"
            k += 1
            if k > 4:
                base = f"{base} {i}"
        names[i] = base
    return names


def _registry_match(centers, local_names, group, registry):
    """Carry stable names across builds: a new centroid close (cosine) to a stored
    one inherits the stored name, so routine refreshes don't rename archetypes."""
    stored = registry.get(group, [])
    out = dict(local_names)
    taken = set()
    for i, c in enumerate(centers):
        best, best_d = None, 0.35  # cosine distance threshold
        cn = c / (np.linalg.norm(c) + 1e-12)
        for s in stored:
            sv = np.asarray(s["center"])
            sv = sv / (np.linalg.norm(sv) + 1e-12)
            d = 1.0 - float(cn @ sv)
            if d < best_d and s["name"] not in taken:
                best, best_d = s["name"], d
        if best is not None:
            out[i] = best
            taken.add(best)
    registry[group] = [{"name": out[i], "center": list(map(float, centers[i]))}
                       for i in range(len(centers))]
    return out


def fit_by_group(X: pd.DataFrame, meta: pd.DataFrame, k_range=range(3, 6),
                 seed: int = 42, registry_path=None) -> tuple[np.ndarray, dict[int, str]]:
    """Cluster each position group separately. Returns (global labels, label→name)."""
    import json
    from pathlib import Path

    registry = {}
    if registry_path is not None and Path(registry_path).exists():
        registry = json.loads(Path(registry_path).read_text())

    labels = np.full(len(X), -1, dtype=int)
    names: dict[int, str] = {}
    offset = 0
    for group in ("DF", "MF", "FW"):
        mask = (meta["pos_group"] == group).to_numpy()
        if mask.sum() < max(k_range) * 5:
            continue
        res = fit(X[mask], k_range=k_range, seed=seed)
        local = name_clusters(res["centers"], list(X.columns), group=group)
        local = _registry_match(res["centers"], local, group, registry)
        labels[mask] = res["labels"] + offset
        for i, name in local.items():
            names[i + offset] = f"{group} · {name}"
        offset += res["k"]

    if registry_path is not None:
        Path(registry_path).write_text(json.dumps(registry))
    return labels, names


def project2d(X: pd.DataFrame, seed: int = 42) -> np.ndarray:
    return PCA(n_components=2, random_state=seed).fit_transform(X.to_numpy())
