import pandas as pd

from fifa import ledger


def _pred(mn, score=(2, 0)):
    return {
        "match_number": mn, "home": "H", "away": "A", "kickoff": "2026-06-13T19:00",
        "predicted": list(score), "p": [0.6, 0.25, 0.15], "tier": "STRONG",
    }


def test_record_freezes_first_prediction(tmp_path):
    path = tmp_path / "ledger.jsonl"
    ledger.record([_pred(1, (2, 0))], path)
    ledger.record([_pred(1, (5, 5)), _pred(2, (1, 1))], path)  # mn=1 must NOT change
    book = ledger.load(path)
    assert book[1]["predicted"] == [2, 0]  # frozen
    assert book[2]["predicted"] == [1, 1]


def test_tracker_scores_played_matches(tmp_path):
    path = tmp_path / "ledger.jsonl"
    ledger.record([_pred(1, (2, 0)), _pred(2, (1, 1)), _pred(3, (0, 2))], path)
    fx = pd.DataFrame(
        {
            "match_number": [1, 2, 3, 4],
            "home": ["H"] * 4, "away": ["A"] * 4,
            "home_score": [2, 0, 1, None], "away_score": [0, 3, 1, None],
            "status": ["played", "played", "played", "upcoming"],
        }
    )
    rows, tally = ledger.tracker(ledger.load(path), fx)
    assert len(rows) == 3
    assert tally["n"] == 3


def test_outcome_judged_on_wdl_probs_not_modal_scoreline(tmp_path):
    # Modal scoreline can be 1-1 (a draw) while home win is the most likely OUTCOME —
    # the Korea-Czechia case. Outcome must be judged on argmax(p).
    path = tmp_path / "ledger.jsonl"
    ledger.record([{"match_number": 9, "home": "South Korea", "away": "Czech Republic",
                    "kickoff": "x", "predicted": [1, 1], "p": [0.42, 0.23, 0.35],
                    "tier": "TOSS-UP"}], path)
    fx = pd.DataFrame({"match_number": [9], "home": ["South Korea"],
                       "away": ["Czech Republic"], "home_score": [2], "away_score": [1],
                       "status": ["played"]})
    rows, tally = ledger.tracker(ledger.load(path), fx)
    assert rows[0]["outcome"] is True   # model picked Korea (42% > 35%) — correct
    assert rows[0]["exact"] is False    # 1-1 ≠ 2-1
    assert tally["outcome_hits"] == 1  # match 1 outcome right (H win); 2 wrong; 3 was draw pred? (1,1) draw vs 0-3 away → wrong; (0,2) away pred vs 1-1 draw → wrong
    assert tally["exact_hits"] == 1  # match 1 exactly 2-0
    assert rows[0]["exact"] is True and rows[1]["outcome"] is False
