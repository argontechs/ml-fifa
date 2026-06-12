import numpy as np
import pytest

from fifa import matrix
from fifa.calibrate import WDLCalibrator

RNG = np.random.default_rng(3)


def test_calibrator_fixes_overconfidence():
    # Synthetic overconfident forecasts: claimed 80% when truth is 60%
    n = 4000
    raw = np.tile([0.8, 0.1, 0.1], (n, 1))
    outcomes = RNG.choice(3, size=n, p=[0.6, 0.2, 0.2])
    cal = WDLCalibrator().fit(raw, outcomes)
    fixed = cal.transform(raw)
    assert fixed[0].sum() == pytest.approx(1.0)
    assert abs(fixed[0][0] - 0.6) < 0.05  # 0.8 pulled down toward truth
    # JSON round-trip preserves behavior
    cal2 = WDLCalibrator.from_dict(cal.to_dict())
    np.testing.assert_allclose(cal2.transform(raw)[0], fixed[0], atol=1e-9)


def test_rescale_wdl_hits_target():
    m = matrix.score_matrix(1.8, 1.0, rho=-0.05)
    target = (0.5, 0.3, 0.2)
    m2 = matrix.rescale_wdl(m, target)
    assert matrix.wdl(m2) == pytest.approx(target, abs=1e-9)
    assert m2.sum() == pytest.approx(1.0)
