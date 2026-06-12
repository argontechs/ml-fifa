#!/bin/bash
# One-shot VPS production setup: systemd services + cron schedule. Idempotent.
set -e
cd "$(dirname "$0")/.."

cp systemd/wc-collector.service systemd/wc-scorer.service systemd/wc-publish.service \
   /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now wc-collector wc-scorer wc-publish

( crontab -l 2>/dev/null | grep -v wc2026 || true
  echo '0 */3 * * * cd /opt/wc2026 && .venv/bin/python update.py >> data/update.log 2>&1'
  echo '0 6 * * 1 cd /opt/wc2026 && .venv/bin/python players_build.py --deploy >> data/players.log 2>&1'
) | crontab -

echo "--- services ---"
systemctl --no-pager --output=short status wc-collector wc-scorer wc-publish | grep -E "wc-|Active"
echo "--- cron ---"
crontab -l | grep wc2026
echo "SETUP COMPLETE"
