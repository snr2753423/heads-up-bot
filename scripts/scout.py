"""Refresh data/opponent_profiles.json from public GitHub and local facts.

Usage:
    python scripts/scout.py
    python scripts/scout.py owner/repo
    python scripts/scout.py owner/repo@fullsha

This does not run opponent code. It only reads public source.
"""

from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scouting.analyze import analyze_agent  # noqa: E402
from scouting.discover import fetch_text, head_commit, list_starter_forks, parse_manual  # noqa: E402
from scouting.library import COUNTERS  # noqa: E402

OUT = os.path.join(ROOT, "data", "opponent_profiles.json")
SELF = "snr2753423/heads-up-bot"


def _candidate_rows(extra: list[str]) -> list[dict]:
    rows = []
    try:
        rows.extend(list_starter_forks())
    except Exception as exc:
        print(f"fork search failed: {exc}", file=sys.stderr)
    for spec in extra:
        rows.append(parse_manual(spec))
    # SSD is not a fork of the starter. Keep it as a named public candidate.
    if not any(row.get("full_name") == "Dhanush006/ssd-heads-up-poker" for row in rows):
        rows.append({
            "full_name": "Dhanush006/ssd-heads-up-poker",
            "html_url": "https://github.com/Dhanush006/ssd-heads-up-poker",
            "fork_of_starter": False,
            "role": "public description says TAMU Datathon 2026",
        })
    return [row for row in rows if row.get("full_name") and row["full_name"] != SELF]


def _inspect(row: dict) -> dict:
    name = row["full_name"]
    commit = {"sha": row.get("sha"), "commit_known_submission": bool(row.get("sha"))}
    if not commit["sha"]:
        try:
            commit.update(head_commit(name))
            commit["commit_known_submission"] = False
        except Exception as exc:
            return {"full_name": name, "error": str(exc), "role": row.get("role")}
    sha = commit["sha"]
    agent = fetch_text(name, sha, "agent.py")
    extra = {}
    if agent and "import strategy" in agent:
        body = fetch_text(name, sha, "strategy.py")
        if body:
            extra["strategy.py"] = body
    analysis = analyze_agent(agent or "", extra) if agent else {
        "archetype": "missing",
        "facts": ["agent.py was not readable at this SHA"],
        "inferences": [],
        "params": {},
    }
    return {
        "full_name": name,
        "html_url": row.get("html_url") or f"https://github.com/{name}",
        "sha": sha,
        "commit_date": commit.get("date"),
        "commit_message": commit.get("message"),
        "commit_known_submission": False,
        "snapshot_warning": "This SHA is the public HEAD (or a hand-supplied SHA), not a confirmed tournament submission.",
        "archetype": analysis["archetype"],
        "facts": analysis["facts"],
        "inferences": analysis["inferences"],
    }


def _profiles(inspections: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for row in inspections:
        if row.get("error"):
            continue
        grouped.setdefault(row["archetype"], []).append(row)
    built = []
    labels = {
        "starter_checkfold": "Public repos still on the starter check/fold action",
        "equity_threshold": "Public snapshot with an equity-threshold strategy module",
    }
    for archetype, rows in grouped.items():
        counter = COUNTERS.get(archetype, {})
        built.append({
            "id": archetype,
            "usable_for_match": bool(counter.get("usable_for_match")),
            "signature": counter.get("signature") or {},
            "counter": counter.get("counter") or {},
            "why": counter.get("why") or "",
            "label": labels.get(archetype, archetype),
            "sources": rows,
        })
    for archetype, counter in COUNTERS.items():
        if archetype in grouped or not counter.get("usable_for_match"):
            continue
        built.append({
            "id": archetype,
            "usable_for_match": True,
            "signature": counter["signature"],
            "counter": counter["counter"],
            "why": counter["why"],
            "label": f"Official practice bot pattern: {archetype}",
            "sources": [{
                "full_name": f"local baseline-bots matching {archetype}",
                "commit_known_submission": False,
                "facts": ["Behavior is taken from the practice bots shipped in this repository."],
                "inferences": [],
            }],
        })
    return built


def main(argv: list[str]) -> None:
    inspections = [_inspect(row) for row in _candidate_rows(argv)]
    payload = {
        "note": "Profiles are public snapshots plus the official practice bots. HEAD is not a submitted SHA.",
        "profiles": _profiles(inspections),
        "inspections": inspections,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    usable = [row["id"] for row in payload["profiles"] if row["usable_for_match"]]
    print(f"wrote {OUT}")
    print(f"usable profiles: {', '.join(usable) or 'none'}")
    for row in inspections:
        if row.get("error"):
            print(f"  {row['full_name']}: ERROR {row['error']}")
        else:
            print(f"  {row['full_name']} {row['sha'][:12]} {row['archetype']}")


if __name__ == "__main__":
    main(sys.argv[1:])
