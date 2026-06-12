from fifa import backtest_lib


def test_grid_pick_minimizes_rps():
    calls = []

    def fake_eval(rho, w):
        calls.append((rho, w))
        return abs(rho + 0.10) + abs(w - 0.6)  # minimum at rho=-0.10, w=0.6

    best = backtest_lib.grid_pick(fake_eval, rhos=[-0.15, -0.10, 0.0], ws=[0.4, 0.6])
    assert best == (-0.10, 0.6)
    assert len(calls) == 6
