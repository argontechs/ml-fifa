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


def test_p_over_lines():
    m = matrix.score_matrix(1.4, 1.2, rho=0.0)
    p15, p25, p35 = (matrix.p_over(m, x) for x in (1.5, 2.5, 3.5))
    assert 1 > p15 > p25 > p35 > 0  # monotone in the line
    # hand-check: P(over 0.5) = 1 − P(0-0) = 1 − e^−1.4·e^−1.2
    import math
    assert matrix.p_over(m, 0.5) == pytest.approx(1 - math.exp(-2.6), abs=1e-3)


def test_double_chance_drops_least_likely():
    # away least likely → 1X (home or draw)
    assert matrix.double_chance((0.60, 0.25, 0.15)) == ("1X", pytest.approx(0.85))
    # draw least likely → 12 (home or away)
    assert matrix.double_chance((0.45, 0.10, 0.45)) == ("12", pytest.approx(0.90))
    # home least likely → X2 (draw or away)
    assert matrix.double_chance((0.15, 0.30, 0.55)) == ("X2", pytest.approx(0.85))


def test_tier_thresholds():
    assert matrix.tier(0.75) == "LOCK"
    assert matrix.tier(0.70) == "LOCK"
    assert matrix.tier(0.60) == "STRONG"
    assert matrix.tier(0.50) == "LEAN"
    assert matrix.tier(0.40) == "TOSS-UP"
