"""Build players.html: fetch (cached) → features → cluster → render. usage:
  .venv/bin/python players_build.py [--deploy]
"""
import argparse
import datetime
from pathlib import Path

from players import cluster, features, fetch, page

ap = argparse.ArgumentParser()
ap.add_argument("--deploy", action="store_true")
args = ap.parse_args()

print("1/4 fetching FBref tables (disk-cached by soccerdata)…")
df = fetch.assemble_multi()
print(f"   {len(df)} players in raw pull")

print("2/4 building position-normalized matrix…")
X, meta = features.build_matrix(df)
print(f"   {len(X)} outfield players with ≥{features.MIN_MINUTES} minutes")

print("3/4 clustering…")
labels, names = cluster.fit_by_group(X, meta)
xy = cluster.project2d(X)
print(f"   {len(set(names.values()))} archetypes: " + ", ".join(sorted(set(names.values()))))

print("4/4 rendering…")
generated = (datetime.datetime.now(datetime.UTC)
             + datetime.timedelta(hours=8)).strftime("%Y-%m-%d %H:%M MYT")
html = page.render(meta, labels, xy, names, generated)
out = Path(__file__).resolve().parent / "dashboard" / "players.html"
out.write_text(html)
print(f"done → {out}")

if args.deploy:
    from fifa.deploy import deploy_dashboard

    deploy_dashboard()
