#!/bin/zsh
# Wrapper for launchd: refresh predictions + redeploy site. Logs to data/update.log.
cd "$(dirname "$0")"
mkdir -p data
{
  echo "===== run $(date '+%Y-%m-%d %H:%M:%S %Z') ====="
  .venv/bin/python update.py
  echo ""
} >> data/update.log 2>&1
# keep the log from growing unbounded
tail -n 2000 data/update.log > data/update.log.tmp && mv data/update.log.tmp data/update.log
