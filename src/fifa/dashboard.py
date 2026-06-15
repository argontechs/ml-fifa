"""Static dashboard generator — floodlit-pitch telemetry aesthetic, zero JS."""
from __future__ import annotations

from . import matrix

# flagcdn.com ISO codes for all 48 WC2026 teams (+ a few likely names)
FLAGS = {
    "Mexico": "mx", "South Africa": "za", "South Korea": "kr", "Czech Republic": "cz",
    "Canada": "ca", "Bosnia and Herzegovina": "ba", "Qatar": "qa", "Switzerland": "ch",
    "Brazil": "br", "Morocco": "ma", "Haiti": "ht", "Scotland": "gb-sct",
    "United States": "us", "Paraguay": "py", "Australia": "au", "Turkey": "tr",
    "Germany": "de", "Curaçao": "cw", "Ivory Coast": "ci", "Ecuador": "ec",
    "Netherlands": "nl", "Japan": "jp", "Sweden": "se", "Tunisia": "tn",
    "Belgium": "be", "Egypt": "eg", "Iran": "ir", "New Zealand": "nz",
    "Spain": "es", "Cape Verde": "cv", "Saudi Arabia": "sa", "Uruguay": "uy",
    "France": "fr", "Senegal": "sn", "Iraq": "iq", "Norway": "no",
    "Argentina": "ar", "Algeria": "dz", "Austria": "at", "Jordan": "jo",
    "Portugal": "pt", "DR Congo": "cd", "Uzbekistan": "uz", "Colombia": "co",
    "England": "gb-eng", "Croatia": "hr", "Ghana": "gh", "Panama": "pa",
}

_CSS = """
:root{
  --pitch:#0a0e0b; --panel:#10160f; --panel2:#141b12; --line:#222b1e;
  --ink:#e9eee4; --dim:#7d8a74; --faint:#48543f;
  --volt:#b8f53d; --volt-dim:#6f9420;
  --win:#b8f53d; --draw:#5b6878; --loss:#f0573f;
}
*{box-sizing:border-box;margin:0;padding:0}
body{
  background:
    radial-gradient(1200px 500px at 50% -200px, rgba(184,245,61,.07), transparent 70%),
    repeating-linear-gradient(90deg, transparent 0 110px, rgba(255,255,255,.013) 110px 220px),
    var(--pitch);
  color:var(--ink);
  font-family:'IBM Plex Mono',ui-monospace,monospace;
  font-size:13px; line-height:1.45;
  max-width:1080px; margin:0 auto; padding:2.2rem 1.2rem 4rem;
}
h1,h2,.team,.hero{font-family:'Saira Condensed','Arial Narrow',sans-serif}
h1{font-size:2.4rem;letter-spacing:.04em;text-transform:uppercase;line-height:1}
h1 .dot{color:var(--volt)}
.sub{color:var(--dim);font-size:11px;letter-spacing:.14em;text-transform:uppercase;margin-top:.35rem}
h2{font-size:1.25rem;letter-spacing:.12em;text-transform:uppercase;margin:3rem 0 1rem;
   display:flex;align-items:center;gap:.7rem;color:var(--ink)}
h2::after{content:"";flex:1;height:1px;background:var(--line)}
.ticker{margin-top:1.4rem;border:1px solid var(--line);border-left:3px solid var(--volt);
  background:var(--panel);padding:.75rem 1rem;color:var(--dim);font-size:11.5px;border-radius:4px}
.ticker b{color:var(--volt)}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:.9rem}
@media(max-width:760px){.grid{grid-template-columns:1fr}}
.card{background:linear-gradient(180deg,var(--panel2),var(--panel));border:1px solid var(--line);
  border-radius:6px;padding:1rem 1.1rem;animation:rise .45s both}
.card:hover{border-color:var(--volt-dim)}
@keyframes rise{from{opacity:0;transform:translateY(10px)}}
.grid .card:nth-child(2){animation-delay:.05s}.grid .card:nth-child(3){animation-delay:.1s}
.grid .card:nth-child(4){animation-delay:.15s}.grid .card:nth-child(5){animation-delay:.2s}
.grid .card:nth-child(6){animation-delay:.25s}.grid .card:nth-child(n+7){animation-delay:.3s}
.meta{display:flex;justify-content:space-between;align-items:center;color:var(--dim);
  font-size:10.5px;letter-spacing:.1em;text-transform:uppercase}
.badge{font-weight:600;font-size:10px;padding:.18rem .55rem;border-radius:3px;letter-spacing:.12em}
.badge.LOCK{background:var(--volt);color:#10160f}
.badge.STRONG{background:#1e3a4d;color:#7ec8e2}
.badge.LEAN{background:#46401c;color:#e2cb7e}
.badge.TOSS-UP{background:#4d211c;color:#f0998c}
.face{display:grid;grid-template-columns:1fr auto 1fr;align-items:center;gap:.8rem;margin:.8rem 0 .2rem}
.side{display:flex;flex-direction:column;align-items:center;gap:.3rem;min-width:0}
.side img{width:34px;height:auto;border-radius:2px;box-shadow:0 0 0 1px var(--line)}
.team{font-size:1.05rem;font-weight:600;letter-spacing:.06em;text-transform:uppercase;
  text-align:center;line-height:1.1}
.hero{font-size:3.4rem;font-weight:700;letter-spacing:.02em;line-height:.9;color:var(--ink);
  text-shadow:0 0 24px rgba(184,245,61,.25)}
.hero .dash{color:var(--faint);padding:0 .05em}
.ctx{color:var(--dim);font-size:10.5px;text-align:center}
.ctx .up{color:var(--volt)} .ctx .dn{color:var(--loss)}
.dots{letter-spacing:.18em;font-size:10px}
.dots .w{color:var(--win)}.dots .d{color:var(--draw)}.dots .l{color:var(--loss)}
.bar{display:flex;height:16px;border-radius:3px;overflow:hidden;margin:.7rem 0 .45rem;
  font-size:9.5px;font-weight:600;color:#0c1209}
.bar div{display:flex;align-items:center;justify-content:center;overflow:hidden;white-space:nowrap}
.bar .w{background:var(--win)} .bar .d{background:var(--draw);color:#d6dde6}
.bar .l{background:var(--loss);color:#2b0d08}
.chips{display:flex;flex-wrap:wrap;gap:.35rem;margin-top:.5rem}
.chip{border:1px solid var(--line);border-radius:3px;padding:.12rem .45rem;color:var(--dim);font-size:10.5px}
.chip.first{border-color:var(--volt-dim);color:var(--volt)}
.mkt{color:var(--faint);font-size:9.5px;letter-spacing:.1em;text-transform:uppercase;margin-top:.55rem}
.stars{color:var(--dim);font-size:10.5px;margin-top:.5rem}
.stars a{color:var(--ink);text-decoration:none;border-bottom:1px dotted var(--faint)}
.stars .v{color:var(--faint);padding:0 .4rem}
.tally{display:flex;gap:2.2rem;margin:.4rem 0 1rem;flex-wrap:wrap}
.tally .num{font-family:'Saira Condensed';font-size:2.2rem;font-weight:700;color:var(--volt);line-height:1}
.tally .lbl{color:var(--dim);font-size:10px;letter-spacing:.14em;text-transform:uppercase}
table{border-collapse:collapse;width:100%;font-size:12px}
td,th{padding:.4rem .55rem;text-align:left;border-bottom:1px solid var(--line)}
th{color:var(--faint);font-size:10px;letter-spacing:.12em;text-transform:uppercase;font-weight:400}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
.tick{color:var(--volt);font-weight:600}.cross{color:var(--loss);font-weight:600}
.ggrid{display:grid;grid-template-columns:repeat(3,1fr);gap:.9rem}
@media(max-width:900px){.ggrid{grid-template-columns:repeat(2,1fr)}}
@media(max-width:600px){.ggrid{grid-template-columns:1fr}}
.gcard{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:.7rem .8rem}
.gcard h3{font-family:'Saira Condensed';font-size:1rem;letter-spacing:.16em;color:var(--volt);
  text-transform:uppercase;margin-bottom:.4rem}
.grow{display:grid;grid-template-columns:auto 1fr repeat(3,2rem) 2.1rem 3rem;gap:.3rem;
  align-items:center;padding:.22rem 0;font-size:11px;
  border-bottom:1px solid rgba(255,255,255,.03);font-variant-numeric:tabular-nums}
.grow img{width:18px;border-radius:1px}
.grow .c{color:var(--dim);text-align:right}
.grow .pts{color:var(--ink);font-weight:600;text-align:right}
.grow .adv{color:var(--dim);font-size:10px;text-align:right}
.grow.q .adv{color:var(--volt)}
.grow.hd{color:var(--faint);font-size:9px;letter-spacing:.1em;text-transform:uppercase;
  border-bottom:1px solid var(--line)}
.grow.hd span{text-align:right}.grow.hd .nm{text-align:left}
.cbar{display:grid;grid-template-columns:9rem 1fr 3.4rem;gap:.6rem;align-items:center;
  padding:.28rem 0;font-size:12px}
.cbar .fill{height:13px;border-radius:2px;background:linear-gradient(90deg,var(--volt-dim),var(--volt));
  box-shadow:0 0 12px rgba(184,245,61,.25);animation:grow .8s both}
@keyframes grow{from{transform:scaleX(0);transform-origin:left}}
.cbar .pc{color:var(--volt);text-align:right;font-variant-numeric:tabular-nums}
.cbar .nm{display:flex;align-items:center;gap:.45rem}.cbar img{width:20px;border-radius:1px}
footer{margin-top:3.5rem;color:var(--faint);font-size:10.5px;line-height:1.8}
footer a{color:var(--dim)}
nav{display:flex;gap:1.4rem;margin-top:1.2rem;font-size:11px;letter-spacing:.16em;
  text-transform:uppercase}
nav a{color:var(--dim);text-decoration:none;padding-bottom:.25rem}
nav a.on{color:var(--volt);border-bottom:2px solid var(--volt)}
nav a:hover{color:var(--ink)}
.retro{font-size:9px;letter-spacing:.12em;color:var(--faint);border:1px solid var(--line);
  border-radius:3px;padding:.14rem .4rem;margin-left:.4rem;text-transform:uppercase}
.duo{display:flex;flex-direction:column;align-items:center;gap:.15rem}
.lbl2{color:var(--faint);font-size:9px;letter-spacing:.18em;text-transform:uppercase}
.hero.dim{font-size:1.6rem;color:var(--dim);text-shadow:none}
.hero.hit{color:var(--volt)} .hero.miss{color:var(--loss);text-shadow:none}
.verdict{display:flex;gap:1.2rem;margin-top:.55rem;font-size:10.5px;color:var(--dim);
  letter-spacing:.1em;text-transform:uppercase}
"""

_FONTS = (
    # social link previews: without og:image, messengers grab the first <img>
    # (a 40px flag) and blur it to card size
    '<meta property="og:title" content="WC26 Predictor — honest ML World Cup predictions">'
    '<meta property="og:description" content="Machine-learned scorelines from 49,000 '
    'internationals. Every pick frozen before kickoff, every miss on the record. '
    'Updated every 3 hours.">'
    '<meta property="og:image" content="https://wc2026.argontechs.dev/og.png">'
    '<meta property="og:image:width" content="1200">'
    '<meta property="og:image:height" content="630">'
    '<meta property="og:type" content="website">'
    '<meta name="twitter:card" content="summary_large_image">'
    '<link rel="icon" href="data:image/svg+xml,'
    '%3Csvg xmlns=%27http://www.w3.org/2000/svg%27 viewBox=%270 0 100 100%27%3E'
    '%3Ctext y=%27.9em%27 font-size=%2790%27%3E%E2%9A%BD%3C/text%3E%3C/svg%3E">'
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link href="https://fonts.googleapis.com/css2?family=Saira+Condensed:wght@600;700'
    '&family=IBM+Plex+Mono:wght@400;600&display=swap" rel="stylesheet">'
)


def _flag(team: str, px: int = 34) -> str:
    code = FLAGS.get(team)
    if not code:
        return ""
    # alt="" because the team name is always printed adjacent (flag is decorative);
    # onerror collapses the element if flagcdn drops a request, so a transient miss
    # shows nothing rather than a broken-image glyph + alt text
    return (f'<img src="https://flagcdn.com/w{40 if px > 24 else 20}/{code}.png" '
            f'alt="" loading="lazy" onerror="this.style.display=\'none\'">')


def _form_dots(last5) -> str:
    sym = {1.0: ('●', 'w'), 0.5: ('●', 'd'), 0.0: ('●', 'l')}
    return '<span class="dots">' + "".join(
        f'<span class="{c}">{s}</span>' for s, c in (sym[v] for v in last5)
    ) + '</span>'


def _trend(v: float) -> str:
    if abs(v) < 5:
        return '<span>→0</span>'
    cls = "up" if v > 0 else "dn"
    arrow = "↑" if v > 0 else "↓"
    return f'<span class="{cls}">{arrow}{abs(v):.0f}</span>'


def _side(team: str, ctx: dict | None) -> str:
    c = ""
    if ctx:
        c = (f'<div class="ctx">ELO {ctx["elo"]:.0f} {_trend(ctx["trend1y"])}</div>'
             f'<div class="ctx">{_form_dots(ctx["last5"])}</div>')
    return f'<div class="side">{_flag(team)}<div class="team">{team}</div>{c}</div>'


def _seg(cls: str, label: str, p: float) -> str:
    txt = f"{label} {p:.0%}" if p >= 0.14 else (f"{p:.0%}" if p >= 0.07 else "")
    return f'<div class="{cls}" style="width:{p:.1%}">{txt}</div>'


def _match_card(m: dict) -> str:
    ph, pd_, pa = m["p"]
    # Hero is the OUTCOME CALL — the thing the model is actually confident about —
    # not the fragile exact score. Index matches ledger.py's outcome scoring so the
    # card and the track record can never disagree about who we picked.
    idx = max(range(3), key=lambda k: m["p"][k])
    fav = m["home"] if idx == 0 else (m["away"] if idx == 2 else "DRAW")
    pwin = m["p"][idx]
    call = (f'<div class="duo"><div class="team" style="font-size:1.3rem">'
            f'{fav.upper() if fav != "DRAW" else "DRAW"}</div>'
            f'<div class="hero" style="font-size:2.6rem">{pwin:.0%}</div>'
            f'<div class="lbl2">{"to win" if fav != "DRAW" else "most likely"}</div></div>')

    # the single most-likely scoreline is demoted to a clearly-labelled low-confidence
    # chip (exact-score is ~14% even at the model's best — never a confidence signal)
    (hs, as_), pr0 = m["top5"][0]
    likeliest = (f'<span class="chip first">likeliest {hs}–{as_} · {pr0:.0%} · '
                 f'low-confidence</span>')
    rest = "".join(f'<span class="chip">{i_}-{j_} {pr:.0%}</span>'
                   for (i_, j_), pr in m["top5"][1:])

    # double-chance: the "safer" two-way cover, with a plain-English gloss so a casual
    # reader isn't left decoding "1X". Purely a readout of our own p — not a market call.
    code, dc_p = matrix.double_chance((ph, pd_, pa))
    drop = {"1X": 2, "12": 1, "X2": 0}[code]
    gloss = (f'{m["home"]} or draw' if drop == 2 else
             f'{m["away"]} or draw' if drop == 0 else
             f'{m["home"]} or {m["away"]} (no draw)')
    dc = f'<span class="chip">{code} · {gloss} · {dc_p:.0%}</span>'

    ou = ""
    if m.get("p_over25") is not None:
        po = m["p_over25"]
        side = "O" if po >= 0.5 else "U"
        pct = po if po >= 0.5 else 1 - po
        ou = f'<span class="chip">{side}2.5 {pct:.0%}</span>'

    def _star_links(entries):
        return " · ".join(
            f'<a href="players.html" title="{e["arch"]}">{e["player"]}</a>'
            for e in entries
        )

    stars = ""
    if m.get("stars_home") or m.get("stars_away"):
        stars = (f'<div class="stars">★ {_star_links(m.get("stars_home", []))}'
                 f'<span class="v">vs</span>{_star_links(m.get("stars_away", []))}</div>')
    draw_note = ('<div class="mkt">model rarely calls draws — see track record</div>'
                 if idx == 1 else "")
    mkt = '<div class="mkt">● market-blended</div>' if m.get("market") else ""
    return f"""<div class="card">
<div class="meta"><span>{m['when']} · {m['comp']}</span>
<span class="badge {m['tier']}">{m['tier']}</span></div>
<div class="face">{_side(m['home'], m.get('ctx_home'))}
{call}
{_side(m['away'], m.get('ctx_away'))}</div>
<div class="bar">{_seg('w', 'W', ph)}{_seg('d', 'D', pd_)}{_seg('l', 'L', pa)}</div>
<div class="chips">{likeliest}{dc}{ou}{rest}</div>{stars}{draw_note}{mkt}
</div>"""


def _nav(page: str) -> str:
    return (f'<nav><a href="index.html" class="{"on" if page == "home" else ""}">Upcoming</a>'
            f'<a href="past.html" class="{"on" if page == "past" else ""}">Past matches</a>'
            f'<a href="leaderboard.html" class="{"on" if page == "leaderboard" else ""}">Leaderboard</a>'
            f'<a href="players.html" class="{"on" if page == "players" else ""}">Players</a>'
            f'<a href="sentiment.html" class="{"on" if page == "sentiment" else ""}">Sentiment</a>'
            f'</nav>')


def _past_card(r: dict) -> str:
    ph, pa = r["predicted"]
    hs, as_ = r["actual"]
    hit_cls = "hit" if r["outcome"] else "miss"
    retro = '<span class="retro">retro</span>' if r.get("retro") else ""
    bar = ""
    if r.get("p"):
        pw, pd_, pl = r["p"]
        bar = f'<div class="bar">{_seg("w", "W", pw)}{_seg("d", "D", pd_)}{_seg("l", "L", pl)}</div>'
    tick = lambda ok: '<span class="tick">✓</span>' if ok else '<span class="cross">✗</span>'  # noqa: E731
    return f"""<div class="card">
<div class="meta"><span>{r.get('when', '')}</span>
<span><span class="badge {r['tier']}">{r['tier']}</span>{retro}</span></div>
<div class="face">{_side(r['home'], None)}
<div class="duo"><div class="lbl2">predicted</div>
<div class="hero dim">{ph}<span class="dash">–</span>{pa}</div>
<div class="lbl2">full time</div>
<div class="hero {hit_cls}">{hs}<span class="dash">–</span>{as_}</div></div>
{_side(r['away'], None)}</div>
{bar}
<div class="verdict"><span>outcome {tick(r['outcome'])}</span>
<span>exact score {tick(r['exact'])}</span></div>
</div>"""


def render_past(view: dict) -> str:
    rows = view["tracker_rows"]
    cards = "\n".join(_past_card(r) for r in reversed(rows)) or \
        "<p>No completed predictions yet — first picks settle after the next matchday.</p>"
    tally = view["tracker_tally"]
    lock = f"{tally['lock_hits']}/{tally['lock_n']}" if tally["lock_n"] else "—"
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>WC26 Predictor · Past matches</title>{_FONTS}<style>{_CSS}</style></head><body>
<header>
<h1>WC26<span class="dot">·</span>Predictor</h1>
<div class="sub">past matches · prediction vs full time · generated {view['generated_at']}</div>
{_nav('past')}
<div class="ticker"><b>Reading this honestly:</b> picks frozen before kickoff are unmarked;
<b>RETRO</b> rows are walk-forward backtest predictions for matches played before this system
existed (model refit on data strictly before that matchday — never trained on the result, never
using bookmaker odds).</div>
</header>
<div class="tally" style="margin-top:1.6rem">
<div><div class="num">{tally['outcome_hits']}/{tally['n']}</div><div class="lbl">outcomes</div></div>
<div><div class="num">{tally['exact_hits']}/{tally['n']}</div><div class="lbl">exact scores</div></div>
<div><div class="num">{lock}</div><div class="lbl">lock picks</div></div>
</div>
<div class="grid">{cards}</div>
<footer>Back to <a href="index.html">upcoming predictions</a>.</footer>
</body></html>"""


def render_leaderboard(view: dict) -> str:
    """Dedicated standings tab: all 12 groups in full detail + the third-place race."""
    standings, advance = view["standings"], view["advance"]
    group_cards = []
    for g in sorted(standings):
        rows = ""
        for pos, r in enumerate(standings[g]):
            zone = "q" if pos < 2 else ""
            rows += (
                f'<div class="grow {zone}">{_flag(r["team"], 18)}<span>{r["team"]}</span>'
                f'<span class="c">{r["p"]}</span><span class="c">{r["w"]}</span>'
                f'<span class="c">{r["d"]}</span><span class="c">{r["l"]}</span>'
                f'<span class="c">{r["gf"]}</span><span class="c">{r["ga"]}</span>'
                f'<span class="c">{r["gf"] - r["ga"]:+d}</span>'
                f'<span class="pts">{r["pts"]}</span>'
                f'<span class="adv">{advance.get(r["team"], 0):.0%}</span></div>'
            )
        header = ('<div class="grow hd lb"><span></span><span class="nm">team</span>'
                  '<span>P</span><span>W</span><span>D</span><span>L</span>'
                  '<span>GF</span><span>GA</span><span>GD</span><span>Pts</span>'
                  '<span>Adv</span></div>')
        group_cards.append(f'<div class="gcard"><h3>Group {g}</h3>{header}{rows}</div>')

    # a group only has a meaningful "third" once it has played — unplayed groups'
    # order is feed-listing noise (Brazil was showing as a 'third' pre-kickoff)
    thirds = sorted(
        ((g, s[2]) for g, s in standings.items()
         if len(s) >= 3 and any(r["p"] > 0 for r in s)),
        key=lambda kv: (kv[1]["pts"], kv[1]["gf"] - kv[1]["ga"], kv[1]["gf"]),
        reverse=True,
    )
    third_rows = ""
    for i, (g, r) in enumerate(thirds):
        cut = ('<tr><td colspan="7" style="border-bottom:2px solid var(--volt);'
               'padding:.1rem"></td></tr>') if i == 8 else ""
        third_rows += (
            f"{cut}<tr><td class='num'>{i + 1}</td>"
            f"<td>{_flag(r['team'], 18)} {r['team']}</td><td>{g}</td>"
            f"<td class='num'>{r['p']}</td><td class='num'>{r['pts']}</td>"
            f"<td class='num'>{r['gf'] - r['ga']:+d}</td><td class='num'>{r['gf']}</td></tr>"
        )
    pending = sorted(set(standings) - {g for g, _ in thirds})
    if pending:
        third_rows += (
            f'<tr><td colspan="7" style="color:var(--faint)">groups '
            f'{", ".join(pending)} join the race once they have played</td></tr>'
        )
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>WC26 Predictor · Leaderboard</title>{_FONTS}<style>{_CSS}
/* leaderboard needs wider cards: 2-across, not the default 3-across */
.ggrid{{grid-template-columns:repeat(2,1fr)}}
@media(max-width:820px){{.ggrid{{grid-template-columns:1fr}}}}
.gcard .grow{{grid-template-columns:auto minmax(0,1fr) repeat(7,1.5rem) 1.9rem 2.7rem;
  font-size:10.5px;gap:.25rem}}
.gcard .grow span:nth-child(2){{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
.grow.q{{background:rgba(184,245,61,.08)}}
.grow.q span:nth-child(2){{color:var(--volt)}}
</style></head><body>
<header>
<h1>WC26<span class="dot">·</span>Leaderboard</h1>
<div class="sub">live group standings · generated {view['generated_at']}</div>
{_nav('leaderboard')}
<div class="ticker"><b>How to read it:</b> green rows are the current top-2 (direct Round-of-32
qualification); the <b>Adv</b> column is our simulation's probability of reaching the R32 by any
route. Below the groups: the third-place race — the best 8 of 12 third-placed teams also qualify;
the volt line marks the cut.</div>
</header>
<h2>Groups</h2>
<div class="ggrid">{''.join(group_cards)}</div>
<h2>Third-place race · top 8 qualify</h2>
<table><tr><th class="num">#</th><th>Team</th><th>Grp</th><th class="num">P</th>
<th class="num">Pts</th><th class="num">GD</th><th class="num">GF</th></tr>{third_rows}</table>
<footer><a href="index.html">back to predictions</a></footer>
</body></html>"""


def _tracker_html(rows: list[dict], tally: dict, card: dict) -> str:
    # Lead with what the model is actually built to do (outcomes) + an honest
    # calibration read, with the tiny sample welded into every label so a live
    # scoreboard can never be mistaken for a verdict. Exact-score is demoted to a
    # "for-fun" footnote (see the design synthesis behind this block).
    n, card_n = tally["n"], card["n"]
    if n == 0:
        sample = (f'<div class="ticker"><b>No frozen picks have settled yet.</b> The only '
                  f'honest performance number on this page is the backtest above '
                  f'({card_n:,} matches) — everything here fills in after the first matchday.</div>')
    else:
        sample = (f'<div class="ticker"><b>Sample so far: {n} settled picks.</b> Far too few to '
                  f'judge this model — treat every number below as a live scoreboard, not a verdict. '
                  f'The honest performance estimate is the backtest above ({card_n:,} matches). '
                  f'A LOCK-tier claim alone needs ~30+ settled picks before it means anything.</div>')
    if not rows:
        return (sample + '<p class="ctx" style="text-align:left;margin-top:.8rem">No completed '
                'predictions yet — first frozen picks settle after the next matchday.</p>')

    frozen = [r for r in rows if not r.get("retro")]
    lock = f"{tally['lock_hits']}/{tally['lock_n']}" if tally["lock_n"] else "—"
    draw_ticker = exact_fun = ""
    if n:
        realized = tally["outcome_hits"] / n
        confs = [max(r["p"]) for r in frozen if r.get("p")]
        mean_conf = f"{sum(confs) / len(confs):.0%}" if confs else "—"
        misses = sum(1 for r in frozen if not r["outcome"])
        draw_misses = sum(1 for r in frozen
                          if not r["outcome"] and r["actual"][0] == r["actual"][1])
        # every backtest figure comes from the live card (single source of truth) so the
        # block can never contradict the header ticker or go stale on a retrain
        cell_a = (f'<div><div class="num">{tally["outcome_hits"]}/{n}</div>'
                  f'<div class="lbl">frozen outcomes · n={n}</div>'
                  f'<div class="ctx" style="text-align:left">{realized:.0%} — vs 33% coin-flip · '
                  f'{card["wdl_acc"]:.1%} model long-run</div></div>')
        cell_b = (f'<div><div class="num">{mean_conf}</div>'
                  f'<div class="lbl">avg stated confidence</div>'
                  f'<div class="ctx" style="text-align:left">right {realized:.0%} so far — '
                  f'too few picks (n={n}) to read into the gap</div></div>')
        if draw_misses:
            dr = card.get("draw_recall")  # None when a backtest test set had no draws
            recall_clause = f" (backtested draw recall {dr:.1%})" if dr is not None else ""
            draw_ticker = (f'<div class="ticker"><b>Known blind spot — draws:</b> {draw_misses} of '
                           f'our {misses} live misses were matches that ended level. The model almost '
                           f'never names a draw as its single most likely result{recall_clause} — so a '
                           f'favourite that draws reads as a miss here. We are an outcome-leaning model, '
                           f'not a draw detector.</div>')
        exact_fun = (f'<div class="mkt">Exact scoreline nailed: {tally["exact_hits"]}/{n} — for fun '
                     f'only. Even at its backtested best the model hits the precise score '
                     f'~{card["exact_rate"]:.0%} (always-1-1 baseline {card["baseline11_rate"]:.1%}): '
                     f'the hardest-possible metric, never a confidence signal.</div>')
    else:
        cell_a = cell_b = ""
    cell_c = (f'<div><div class="num" style="color:var(--dim)">{lock}</div>'
              f'<div class="lbl">lock picks</div>'
              f'<div class="ctx" style="text-align:left">long-run ≥70% contract · '
              f'needs ~30+ to judge</div></div>')

    body = "".join(
        f"<tr><td>{_flag(r['home'], 18)} {r['home']} v {r['away']} {_flag(r['away'], 18)}</td>"
        f"<td class='num'>{r['predicted'][0]}-{r['predicted'][1]}</td>"
        f"<td class='num'>{r['actual'][0]}-{r['actual'][1]}</td>"
        f"<td><span class='badge {r['tier']}'>{r['tier']}</span>"
        f"{'<span class=retro>retro</span>' if r.get('retro') else ''}</td>"
        f"<td>{'<span class=tick>✓</span>' if r['outcome'] else '<span class=cross>✗</span>'}</td>"
        f"<td>{'<span class=tick>✓</span>' if r['exact'] else '<span class=cross>✗</span>'}</td></tr>"
        for r in rows
    )
    retro_line = ""
    if tally.get("retro_n"):
        retro_line = (f'<div class="ctx" style="text-align:left;margin:.6rem 0">RETRO '
                      f'(walk-forward, pre-system matches): outcomes '
                      f'{tally["retro_outcome_hits"]}/{tally["retro_n"]} · exact '
                      f'{tally["retro_exact_hits"]}/{tally["retro_n"]} — counted separately, '
                      f'never in the headline numbers.</div>')
    return f"""{sample}
<div class="tally" style="margin-top:1rem">{cell_a}{cell_b}{cell_c}</div>
{draw_ticker}{exact_fun}{retro_line}
<table><tr><th>Match</th><th class="num">Pred</th><th class="num">Actual</th><th>Tier</th><th>Out</th><th>Exact (fun)</th></tr>{body}</table>"""


def _standings_html(standings: dict, advance: dict) -> str:
    header = ('<div class="grow hd"><span></span><span class="nm">team</span>'
              '<span>W</span><span>D</span><span>L</span><span>GD</span><span>pts·adv</span></div>')
    cards = []
    for g in sorted(standings):
        rows = "".join(
            f'<div class="grow{" q" if advance.get(r["team"], 0) >= .5 else ""}">'
            f'{_flag(r["team"], 18)}<span>{r["team"]}</span>'
            f'<span class="c">{r["w"]}</span><span class="c">{r["d"]}</span>'
            f'<span class="c">{r["l"]}</span>'
            f'<span class="c">{r["gf"] - r["ga"]:+d}</span>'
            f'<span class="pts">{r["pts"]} <span class="adv">{advance.get(r["team"], 0):.0%}</span></span>'
            f'</div>'
            for r in standings[g]
        )
        cards.append(f'<div class="gcard"><h3>Group {g}</h3>{header}{rows}</div>')
    return f'<div class="ggrid">{"".join(cards)}</div>'


def _champions_html(champions: dict) -> str:
    top = sorted(champions.items(), key=lambda kv: -kv[1])[:12]
    peak = top[0][1] if top else 1
    return "".join(
        f'<div class="cbar"><span class="nm">{_flag(t, 20)} {t}</span>'
        f'<div><div class="fill" style="width:{p / peak:.0%}"></div></div>'
        f'<span class="pc">{p:.1%}</span></div>'
        for t, p in top
    )


def render(view: dict) -> str:
    card = view["card"]
    matches = "\n".join(_match_card(m) for m in view["matches"]) or "<p>No upcoming fixtures.</p>"
    lock = f"{card['lock_acc']:.0%} on {card['lock_n']} picks" if card.get("lock_acc") else "n/a"
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>WC26 Predictor</title>{_FONTS}<style>{_CSS}</style></head><body>
<header>
<h1>WC26<span class="dot">·</span>Predictor</h1>
<div class="sub">machine-learned scorelines · 49,000 internationals · generated {view['generated_at']}</div>
{_nav('home')}
<div class="ticker"><b>Honest numbers</b> — backtested on {card['n']:,} matches:
W/D/L {card['wdl_acc']:.1%} · exact score {card['exact_rate']:.1%} (always-1-1 baseline
{card['baseline11_rate']:.1%}) · top-5 hit {card['top5_rate']:.1%} · RPS {card['rps']:.3f} ·
<b>LOCK-tier accuracy {lock}</b>. No model on earth gets 80% on all matches —
LOCK marks the picks that historically do.</div>
</header>
<h2>Track record</h2>
{_tracker_html(view['tracker_rows'], view['tracker_tally'], card)}
<h2>Upcoming · next 7 days</h2>
<div class="grid">{matches}</div>
<h2>Groups · advance odds</h2>
{_standings_html(view['standings'], view['advance'])}
<h2>Champion odds · 10,000 simulations</h2>
{_champions_html(view['champions'])}
<footer>Model: Elo (eloratings.net formulas) → LightGBM Poisson goal rates + Dixon-Coles,
isotonic-calibrated, bookmaker-blended where available.<br>
Data: <a href="https://github.com/martj42/international_results">martj42/international_results</a> (CC0)
· fixturedownload.com · flags by flagcdn.com · Built with Claude Code.</footer>
</body></html>"""
