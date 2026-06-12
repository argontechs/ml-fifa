import numpy as np
import pytest

from fifa import evaluate, matrix


def test_rps_hand_values():
    assert evaluate.rps((1.0, 0.0, 0.0), 0) == pytest.approx(0.0)  # certain & right
    assert evaluate.rps((1.0, 0.0, 0.0), 2) == pytest.approx(1.0)  # certain & maximally wrong
    assert evaluate.rps((1 / 3, 1 / 3, 1 / 3), 0) == pytest.approx(5 / 18)  # uniform vs home win


def test_outcome_of():
    assert evaluate.outcome_of(2, 0) == 0
    assert evaluate.outcome_of(1, 1) == 1
    assert evaluate.outcome_of(0, 3) == 2


def test_report_card_counts_lock_tier():
    m_fav = matrix.score_matrix(3.0, 0.4, rho=-0.05)  # heavy favorite → LOCK
    m_even = matrix.score_matrix(1.2, 1.2, rho=-0.05)  # toss-up
    rows = [
        evaluate.score_prediction(m_fav, 2, 0),  # favorite wins 2-0
        evaluate.score_prediction(m_even, 1, 1),  # draw
    ]
    card = evaluate.report_card(rows)
    assert card["n"] == 2
    assert card["lock_n"] == 1 and card["lock_acc"] == 1.0
    assert 0 <= card["rps"] <= 1
    assert card["exact_rate"] >= 0.5  # 2-0 is the modal score of m_fav
    assert card["baseline11_rate"] == 0.5
