"""Cloudflare Pages deploy helper, shared by update.py / players_build.py / sentiment_publish.py."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

from . import data

ACCOUNT_ID = "6c51b7a0e7b4980a0d5897c365ddc36e"
PROJECT = "wc2026"


def deploy_dashboard(repo_root: Path | None = None, timeout: int = 300) -> bool:
    import fcntl

    root = repo_root or Path(__file__).resolve().parents[2]
    lock_path = data.DATA_DIR / ".deploy.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = open(lock_path, "w")
    try:
        # serialize concurrent deployers (audit: 3 unsynchronized writers raced wrangler)
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _deploy_locked(root, timeout)
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


def _deploy_locked(root: Path, timeout: int) -> bool:
    env = dict(os.environ)
    token_file = data.DATA_DIR / "cf_token.txt"
    if token_file.exists():
        env["CLOUDFLARE_API_TOKEN"] = token_file.read_text().strip()
        env["CLOUDFLARE_ACCOUNT_ID"] = ACCOUNT_ID
    import shutil
    wrangler = ([shutil.which("wrangler")] if shutil.which("wrangler")
                else ["npx", "--yes", "wrangler"])
    try:
        res = subprocess.run(
            [*wrangler, "pages", "deploy", str(root / "dashboard"),
             "--project-name", PROJECT, "--branch", "main", "--commit-dirty=true"],
            timeout=timeout, cwd=root, env=env, capture_output=True, text=True,
        )
        if res.returncode != 0:
            print(f"WARNING: deploy failed:\n{(res.stderr or res.stdout)[-500:]}")
            return False
        return True
    except Exception as exc:  # noqa: BLE001 — deploys must never crash callers
        print(f"WARNING: deploy skipped ({exc})")
        return False
