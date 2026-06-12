"""Static dashboard generator — no server, no JS dependencies."""
from __future__ import annotations

_CSS = """
body{font-family:-apple-system,Segoe UI,sans-serif;margin:2rem auto;max-width:960px;
     background:#0e1117;color:#e6e6e6;padding:0 1rem}
h1{font-size:1.6rem} h2{font-size:1.15rem;margin-top:2.2rem;border-bottom:1px solid #2a2f3a;
     padding-bottom:.3rem} small{color:#8a93a3}
.match{background:#161b25;border-radius:10px;padding:.9rem 1.1rem;margin:.6rem 0}
.badge{font-size:.72rem;padding:.15rem .5rem;border-radius:99px;font-weight:700;margin-left:.5rem}
.LOCK{background:#1d4f2b;color:#7ee2a0}.STRONG{background:#1d3a4f;color:#7ec8e2}
.LEAN{background:#4f431d;color:#e2cb7e}.TOSS-UP{background:#4f1d1d;color:#e27e7e}
.bar{display:flex;height:10px;border-radius:5px;overflow:hidden;margin:.45rem 0}
.bar div:nth-child(1){background:#4caf7d}.bar div:nth-child(2){background:#8a93a3}
.bar div:nth-child(3){background:#c75c5c}
table{border-collapse:collapse;width:100%}td,th{padding:.35rem .6rem;text-align:left;
     border-bottom:1px solid #2a2f3a}
.note{background:#1a2030;border-left:3px solid #7ec8e2;padding:.7rem 1rem;border-radius:6px;
     font-size:.88rem;color:#b8c0cf}
"""


def _match_block(p_):
    tops = ", ".join(f"{i}-{j} ({pr:.1%})" for (i, j), pr in p_["top5"])
    ph, pd_, pa = p_["p"]
    return f"""<div class="match">
<b>{p_['home']} vs {p_['away']}</b> <small>{p_['when']} · {p_['comp']}</small>
<span class="badge {p_['tier']}">{p_['tier']}</span>
<div class="bar"><div style="width:{ph:.0%}"></div><div style="width:{pd_:.0%}"></div>
<div style="width:{pa:.0%}"></div></div>
<small>W {ph:.1%} · D {pd_:.1%} · L {pa:.1%} — top: {tops}</small></div>"""


def render(generated_at, predictions, advance, champions, card) -> str:
    matches = "\n".join(_match_block(p) for p in predictions) or "<p>No upcoming fixtures.</p>"
    adv_rows = "\n".join(
        f"<tr><td>{t}</td><td>{p:.1%}</td></tr>"
        for t, p in sorted(advance.items(), key=lambda kv: -kv[1])
    )
    champ_rows = "\n".join(
        f"<tr><td>{i + 1}</td><td>{t}</td><td>{p:.1%}</td></tr>"
        for i, (t, p) in enumerate(sorted(champions.items(), key=lambda kv: -kv[1])[:15])
    )
    lock = f"{card['lock_acc']:.0%} on {card['lock_n']} picks" if card.get("lock_acc") else "n/a"
    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>WC2026 Predictor</title><style>{_CSS}</style></head><body>
<h1>⚽ WC2026 Predictor <small>generated {generated_at}</small></h1>
<div class="note"><b>Honest numbers:</b> backtested on {card['n']:,} matches —
W/D/L {card['wdl_acc']:.1%}, exact score {card['exact_rate']:.1%}
(always-1-1 baseline {card['baseline11_rate']:.1%}), top-5 scoreline {card['top5_rate']:.1%},
RPS {card['rps']:.3f}, <b>LOCK-tier accuracy {lock}</b>. No model on earth gets 80% on all
matches — LOCK badges mark the picks that historically do.</div>
<h2>Upcoming fixtures</h2>
{matches}
<h2>Advance to Round of 32</h2>
<table><tr><th>Team</th><th>P(advance)</th></tr>{adv_rows}</table>
<h2>Champion odds (Monte Carlo)</h2>
<table><tr><th>#</th><th>Team</th><th>P(champion)</th></tr>{champ_rows}</table>
</body></html>"""
