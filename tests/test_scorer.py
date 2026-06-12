import pytest

from sentiment import db, scorer


def stub_model(texts, **kw):
    out = []
    for t in texts:
        if "goal" in t.lower():
            out.append([{"label": "positive", "score": 0.8},
                        {"label": "neutral", "score": 0.1},
                        {"label": "negative", "score": 0.1}])
        else:
            out.append([{"label": "positive", "score": 0.1},
                        {"label": "neutral", "score": 0.2},
                        {"label": "negative", "score": 0.7}])
    return out


def test_score_texts_maps_to_signed_scale():
    scores = scorer.score_texts(stub_model, ["GOAL what a strike", "we were robbed"])
    assert scores[0] == pytest.approx(0.7)   # 0.8 − 0.1
    assert scores[1] == pytest.approx(-0.6)  # 0.1 − 0.7


def test_run_once_scores_batch_and_returns_count(tmp_path):
    conn = db.connect(tmp_path / "s.db")
    for text in ("goal!", "terrible ref", "goal again"):
        db.insert_post(conn, 1.0, "bsky", 1, "home", text)
    assert scorer.run_once(conn, stub_model) == 3
    assert scorer.run_once(conn, stub_model) == 0  # nothing left unscored
    frame = db.posts_frame(conn, 1)
    assert frame["score"].notna().all()
    assert (frame["score"] > 0).sum() == 2


@pytest.mark.live
def test_real_model_multilingual_signs():
    model = scorer.load_model()
    scores = scorer.score_texts(model, [
        "GOOOOAL what a strike, incredible!",
        "embarrassing performance, we were robbed",
        "¡Qué golazo, vamos!",
        "quelle défaite honteuse",
    ])
    assert scores[0] > 0 and scores[2] > 0
    assert scores[1] < 0 and scores[3] < 0


def poison_model(texts, **kw):
    if any("POISON" in t for t in texts):
        raise RuntimeError("model exploded")
    return stub_model(texts)


def test_poison_post_cannot_stall_the_queue(tmp_path):
    conn = db.connect(tmp_path / "s.db")
    for text in ("goal!", "POISON \x00 frame", "another goal!"):
        db.insert_post(conn, 1.0, "bsky", 1, "home", text)
    n = scorer.run_once(conn, poison_model)
    assert n == 3  # batch completes despite the poison item
    frame = db.posts_frame(conn, 1)
    assert frame["score"].notna().all()  # nothing left to crash-loop on
    assert (frame["score"] > 0).sum() == 2  # the two good posts scored normally
