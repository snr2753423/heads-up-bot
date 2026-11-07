"""Find public repositories that might be tournament bots.

A fork of the starter is a candidate, not proof that the team submitted it.
HEAD is recorded as a snapshot. It is not treated as the submitted commit
unless a SHA is supplied by hand.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import urllib.request

STARTER = "tamu-datathon-org/heads-up-bundle"
SELF = "snr2753423/heads-up-bot"
USER_AGENT = "brown-ballers-scout"


def _gh(*args: str) -> str:
    if shutil.which("gh") is None:
        raise RuntimeError("gh is not installed; pass repository URLs by hand")
    proc = subprocess.run(
        ["gh", *args],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or f"gh failed: {' '.join(args)}")
    return proc.stdout


def list_starter_forks() -> list[dict]:
    """Public forks of the official starter. Paginated through gh."""
    raw = _gh(
        "api",
        "--paginate",
        f"repos/{STARTER}/forks?per_page=100",
    )
    # gh --paginate concatenates JSON arrays. Split on the boundary.
    chunks = []
    decoder = json.JSONDecoder()
    text = raw.strip()
    index = 0
    while index < len(text):
        while index < len(text) and text[index].isspace():
            index += 1
        if index >= len(text):
            break
        value, index = decoder.raw_decode(text, index)
        if isinstance(value, list):
            chunks.extend(value)
        else:
            chunks.append(value)
    found = []
    for repo in chunks:
        full = repo.get("full_name") or ""
        if full in (STARTER, SELF):
            continue
        found.append({
            "full_name": full,
            "html_url": repo.get("html_url"),
            "pushed_at": repo.get("pushed_at"),
            "description": repo.get("description"),
            "fork_of_starter": True,
            "role": "candidate",
        })
    return found


def head_commit(full_name: str) -> dict:
    payload = json.loads(_gh("api", f"repos/{full_name}/commits/HEAD"))
    commit = payload.get("commit") or {}
    return {
        "sha": payload.get("sha"),
        "date": (commit.get("committer") or {}).get("date"),
        "message": (commit.get("message") or "").splitlines()[0][:160],
        "commit_known_submission": False,
    }


def fetch_text(full_name: str, sha: str, path: str) -> str | None:
    url = f"https://raw.githubusercontent.com/{full_name}/{sha}/{path}"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.read().decode("utf-8", "replace")
    except Exception:
        return None


def parse_manual(spec: str) -> dict:
    """Accept owner/repo or owner/repo@sha."""
    sha = None
    name = spec.strip().rstrip("/")
    if name.startswith("https://github.com/"):
        name = name[len("https://github.com/"):]
    name = name.split("?")[0].strip("/")
    if "@" in name:
        name, sha = name.split("@", 1)
    return {"full_name": name, "sha": sha or None, "fork_of_starter": False, "role": "manual"}
