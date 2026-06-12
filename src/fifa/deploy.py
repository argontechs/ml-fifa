"""Cloudflare Pages deploy helper, shared by update.py / players_build.py / sentiment_publish.py."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

from . import data

ACCOUNT_ID = "6c51b7a0e7b4980a0d5897c365ddc36e"
PROJECT = "wc2026"


def deploy_dashboard(repo_root: Path | None = None, timeout: int = 300) -> bool:
    root = repo_root or Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    token_file = data.DATA_DIR / "cf_token.txt"
    if token_file.exists():
        env["CLOUDFLARE_API_TOKEN"] = token_file.read_text().strip()
        env["CLOUDFLARE_ACCOUNT_ID"] = ACCOUNT_ID
    try:
        res = subprocess.run(
            ["npx", "--yes", "wrangler", "pages", "deploy", str(root / "dashboard"),
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
