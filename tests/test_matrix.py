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
