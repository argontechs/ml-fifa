from fifa import runtime


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
