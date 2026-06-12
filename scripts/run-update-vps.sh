#!/bin/bash
# VPS cron wrapper: refresh + deploy, with heartbeat alerting and log rotation.
# Alerting: create a FREE check at https://healthchecks.io (ping period: 3h, grace 30m),
# put its ping URL in data/heartbeat_url.txt — you get an email/Telegram alert whenever
# the refresh stops succeeding. Without the file, the wrapper still logs failures loudly.
cd "$(dirname "$0")/.."
HB=""
[ -f data/heartbeat_url.txt ] && HB="$(cat data/heartbeat_url.txt)"
if .venv/bin/python update.py >> data/update.log 2>&1; then
  [ -n "$HB" ] && curl -fsS -m 10 --retry 3 "$HB" >/dev/null 2>&1
else
  echo "$(date '+%F %T') UPDATE FAILED — see above" >> data/update.log
  [ -n "$HB" ] && curl -fsS -m 10 --retry 3 "$HB/fail" >/dev/null 2>&1
fi
tail -n 4000 data/update.log > data/update.log.tmp && mv data/update.log.tmp data/update.log
