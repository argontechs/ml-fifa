from fifa import dashboard


def test_render_contains_sections_and_teams():
    html = dashboard.render(
        generated_at="2026-06-12 14:00 UTC",
        predictions=[
            {
                "home": "France", "away": "Senegal", "when": "2026-06-16", "comp": "Group I",
                "p": (0.58, 0.24, 0.18), "tier": "STRONG",
                "top5": [((1, 0), 0.118), ((2, 0), 0.109)],
            }
        ],
        advance={"France": 0.93, "Senegal": 0.55},
        champions={"Spain": 0.145, "France": 0.124},
        card={
            "n": 3000, "wdl_acc": 0.57, "rps": 0.201, "exact_rate": 0.11,
            "baseline11_rate": 0.10, "top5_rate": 0.38, "lock_n": 800, "lock_acc": 0.82,
            "draw_recall": 0.21, "logloss": 0.95, "brier": 0.55,
        },
    )
    for needle in ("France vs Senegal", "STRONG", "Spain", "14.5%", "82", "Honest"):
        assert needle in html
