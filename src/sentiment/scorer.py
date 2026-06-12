"""Sentiment scoring loop: multilingual XLM-RoBERTa over unscored posts."""
from __future__ import annotations

import time

from . import db

MODEL_ID = "cardiffnlp/twitter-xlm-roberta-base-sentiment"


def load_model():
    import torch
    from transformers import pipeline

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"loading {MODEL_ID} on {device} …")
    return pipeline("text-classification", model=MODEL_ID, device=device,
                    top_k=None, truncation=True, max_length=128)


def score_texts(model, texts: list[str]) -> list[float]:
    """P(positive) − P(negative) per text, in [−1, 1]."""
    out = []
    for label_probs in model(texts, batch_size=32):
        probs = {d["label"].lower(): d["score"] for d in label_probs}
        out.append(probs.get("positive", 0.0) - probs.get("negative", 0.0))
    return out


def run_once(conn, model, batch: int = 128) -> int:
    rows = db.unscored_batch(conn, batch)
    if not rows:
        return 0
    ids = [r[0] for r in rows]
    texts = [r[1] for r in rows]
    db.set_scores(conn, ids, score_texts(model, texts))
    return len(rows)


def run_loop(conn, model, poll: float = 2.0) -> None:
    print("scorer running — Ctrl-C to stop")
    while True:
        n = run_once(conn, model)
        if n:
            print(f"scored {n}")
        else:
            time.sleep(poll)
