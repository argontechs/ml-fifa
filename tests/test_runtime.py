import json

from fifa import runtime


def test_tuned_params_prefers_rolling_production_calibrator(tmp_path, monkeypatch):
    from fifa import data as fdata

    monkeypatch.setattr(fdata, "DATA_DIR", tmp_path)
    val_only = {"curves": [[[0.0, 1.0], [0.1, 0.9]]] * 3}
    production = {"curves": [[[0.0, 1.0], [0.2, 0.8]]] * 3}
    (tmp_path / "backtest_report.json").write_text(json.dumps({
        "rho": -0.05, "w_dc": 0.3,
        "calibrator": val_only, "calibrator_production": production,
    }))
    rho, w, cal = runtime.tuned_params()
    assert (rho, w) == (-0.05, 0.3)
    assert cal.curves_[0][1] == [0.2, 0.8]  # production curves chosen, not val-only


def test_format_prediction_block():
    text = runtime.format_prediction(
        home="France", away="Senegal", when="2026-06-16", comp="Group I",
        p=(0.583, 0.241, 0.176),
        top5=[((1, 0), 0.118), ((2, 0), 0.109), ((2, 1), 0.094), ((1, 1), 0.087), ((0, 0), 0.062)],
    )
    assert "France vs Senegal" in text
    assert "W 58.3%" in text and "D 24.1%" in text
    assert "[STRONG]" in text
    assert "1-0 (11.8%)" in text
