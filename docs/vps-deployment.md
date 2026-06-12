# Production on the Hostinger VPS (CloudPanel)

Architecture stays split: **the website remains on Cloudflare Pages** (free, global, fast);
the VPS replaces your Mac as the always-on *brain* — retraining/refreshing every 3 h, running
the sentiment collector + scorer 24/7, and pushing near-live sentiment snapshots during
matches. CloudPanel itself is barely involved: everything runs over SSH; CloudPanel is only
used (optionally, step 9) to expose the live Dash app.

## 0. Check the box first

```bash
ssh root@YOUR_VPS_IP
free -h          # want ≥4 GB total (the sentiment model needs ~2 GB)
nproc            # 2+ cores fine
df -h /          # want ≥15 GB free (torch ≈ 2.5 GB, model ≈ 1.1 GB, data < 1 GB)
```

If RAM is 2–3 GB, add swap before anything else:

```bash
fallocate -l 4G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile
echo '/swapfile none swap sw 0 0' >> /etc/fstab
```

## 1. Get the code onto the VPS (via private GitHub repo)

On the **Mac** (one-time):

```bash
cd ~/Desktop/ML-FIFA
gh repo create ml-fifa --private --source . --push
```

On the **VPS**:

```bash
apt update && apt install -y git python3-venv python3-pip curl
curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && apt install -y nodejs  # wrangler needs Node ≥18
cd /opt && git clone https://github.com/YOUR_GH_USERNAME/ml-fifa.git wc2026 && cd wc2026
```

## 2. Python environment (CPU-only torch — much smaller, no GPU on the VPS)

```bash
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -r requirements.txt -r requirements-sentiment.txt soccerdata
.venv/bin/pip install -e .
```

## 3. Secrets + state that aren't in git (copy from the Mac)

```bash
# run on the MAC:
scp ~/Desktop/ML-FIFA/data/odds_api_key.txt ~/Desktop/ML-FIFA/data/cf_token.txt \
    root@YOUR_VPS_IP:/opt/wc2026/data/
scp ~/soccerdata/config/league_dict.json root@YOUR_VPS_IP:/root/soccerdata/config/ 2>/dev/null \
  || ssh root@YOUR_VPS_IP "mkdir -p /root/soccerdata/config" \
     && scp ~/soccerdata/config/league_dict.json root@YOUR_VPS_IP:/root/soccerdata/config/
```

The prediction ledger, tuned-model report and sim results ARE in git (force-added), so the
frozen-picks history carries over automatically.

## 4. Smoke test (manual, once)

```bash
cd /opt/wc2026 && .venv/bin/pytest -q -m "not live"     # all green
.venv/bin/python update.py                               # full refresh + deploy (~5-10 min on VPS CPU)
```

Watch for `Deployment complete` and check https://wc2026.argontechs.dev shows the new
`generated …` timestamp.

## 5. The two always-on sentiment services (systemd)

```bash
cat > /etc/systemd/system/wc-collector.service <<'EOF'
[Unit]
Description=WC2026 sentiment collector (Bluesky firehose + goal polling)
After=network-online.target
[Service]
WorkingDirectory=/opt/wc2026
ExecStart=/opt/wc2026/.venv/bin/python sentiment_collect.py
Restart=always
RestartSec=10
[Install]
WantedBy=multi-user.target
EOF

cat > /etc/systemd/system/wc-scorer.service <<'EOF'
[Unit]
Description=WC2026 sentiment scorer (multilingual transformer)
After=network-online.target
[Service]
WorkingDirectory=/opt/wc2026
ExecStart=/opt/wc2026/.venv/bin/python sentiment_score.py
Restart=always
RestartSec=10
[Install]
WantedBy=multi-user.target
EOF

cat > /etc/systemd/system/wc-publish.service <<'EOF'
[Unit]
Description=WC2026 sentiment page publisher (deploys only when new posts exist)
After=network-online.target
[Service]
WorkingDirectory=/opt/wc2026
ExecStart=/opt/wc2026/.venv/bin/python sentiment_publish.py --loop 300
Restart=always
RestartSec=30
[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable --now wc-collector wc-scorer wc-publish
journalctl -u wc-collector -f     # watch it find tonight's matches
```

The publisher is change-gated: between matches nothing new arrives, so nothing deploys.
During a match the live site's Sentiment tab refreshes within ~5 minutes of the crowd.

## 6. Scheduled refreshes (cron)

```bash
crontab -e
# every 3 hours: results, retrain, predictions, sim, all four pages, deploy
0 */3 * * * cd /opt/wc2026 && .venv/bin/python update.py >> data/update.log 2>&1
# Mondays 06:00: refresh player profiles (club seasons move slowly)
0 6 * * 1 cd /opt/wc2026 && .venv/bin/python players_build.py --deploy >> data/players.log 2>&1
```

## 7. Retire the Mac as the brain (IMPORTANT — one writer only)

Two machines both freezing predictions and deploying means diverging ledgers. On the **Mac**:

```bash
launchctl unload ~/Library/LaunchAgents/wc2026-update.plist
```

The Mac stays your dev machine; `git push` from Mac, `git pull` on the VPS to ship changes.

## 8. Health checks

```bash
systemctl status wc-collector wc-scorer wc-publish
tail -50 /opt/wc2026/data/update.log
sqlite3 /opt/wc2026/data/sentiment.db 'SELECT COUNT(*), MAX(ts) FROM posts;'
```

## 9. Optional: expose the LIVE Dash app via CloudPanel

The static Sentiment tab updates every ~5 min; if you want the true 3-second live view on
the web: run `sentiment_app.py` as a fourth systemd unit (same pattern, port 8050), then in
CloudPanel: **Sites → Add Site → Reverse Proxy**, domain e.g. `live.argontechs.dev`,
target `http://127.0.0.1:8050`. Add a DNS record for that subdomain pointing at the VPS IP.
⚠️ Dash has no built-in auth — enable CloudPanel's Basic Auth on that site unless you're
happy with it being public.

## Rollback

The Mac setup keeps working at any time: `launchctl load ~/Library/LaunchAgents/wc2026-update.plist`
and stop the VPS units. The ledger is append-only JSONL in git — merge conflicts are visible,
not silent.
