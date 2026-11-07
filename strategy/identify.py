"""Match live frequencies to bundled profiles. No network, no search."""

from __future__ import annotations

import json
import os

_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "opponent_profiles.json")
_PROFILES: list[dict] | None = None

_MIN_OBS = 40
_MAX_DISTANCE = 0.22
_GAP = 0.08
_MAX_WEIGHT = 0.65

_EMPTY = {
    "weight": 0.0,
    "bet_delta": 0.0,
    "call_delta": 0.0,
    "value_delta": 0.0,
    "station": False,
    "profile_id": None,
}


def _load() -> list[dict]:
    global _PROFILES
    if _PROFILES is None:
        try:
            with open(_PATH, encoding="utf-8") as handle:
                payload = json.load(handle)
            _PROFILES = [row for row in payload.get("profiles", []) if row.get("usable_for_match")]
        except (OSError, json.JSONDecodeError):
            _PROFILES = []
    return _PROFILES


def _bucket(memory: dict) -> dict | None:
    opp = memory.get("opp") or {}
    post = opp.get("post") or {}
    pre = opp.get("pre") or {}
    if post.get("fold", 0) + post.get("continue", 0) >= _MIN_OBS:
        return post
    if pre.get("fold", 0) + pre.get("continue", 0) >= _MIN_OBS:
        return pre
    return None


def live_features(memory: dict) -> dict | None:
    stats = _bucket(memory)
    if not stats:
        return None
    folds = stats.get("fold", 0)
    continues = stats.get("continue", 0)
    raises = stats.get("raise", 0)
    non_raises = stats.get("non_raise", 0)
    decisions = folds + continues
    actions = raises + non_raises
    if decisions < _MIN_OBS or actions <= 0:
        return None
    return {
        "fold_rate": folds / decisions,
        "raise_rate": raises / actions,
        "pot_share": (stats.get("pot", 0) / raises) if raises else 0.0,
        "raises": raises,
    }


def _distance(features: dict, signature: dict) -> float:
    gap = abs(features["fold_rate"] - signature["fold_rate"])
    gap += abs(features["raise_rate"] - signature["raise_rate"])
    if features["raises"] >= 8 and "pot_share" in signature:
        gap += 0.5 * abs(features["pot_share"] - signature["pot_share"])
    return gap


def library_adjustments(memory: dict) -> dict:
    """A small extra shift when one verified profile is clearly closest."""
    try:
        features = live_features(memory)
        profiles = _load()
    except Exception:
        return dict(_EMPTY)
    if features is None or not profiles:
        return dict(_EMPTY)
    scored = []
    for row in profiles:
        signature = row.get("signature") or {}
        if "fold_rate" not in signature or "raise_rate" not in signature:
            continue
        scored.append((_distance(features, signature), row))
    if not scored:
        return dict(_EMPTY)
    scored.sort(key=lambda item: item[0])
    best_distance, best = scored[0]
    second = scored[1][0] if len(scored) > 1 else best_distance + _GAP + 1
    if best_distance > _MAX_DISTANCE or second < best_distance + _GAP:
        memory["profile_id"] = None
        return dict(_EMPTY)
    weight = _MAX_WEIGHT * max(0.0, (_MAX_DISTANCE - best_distance) / _MAX_DISTANCE)
    counter = best.get("counter") or {}
    memory["profile_id"] = best.get("id")
    memory["profile_weight"] = weight
    return {
        "weight": weight,
        "bet_delta": float(counter.get("bet_delta") or 0.0),
        "call_delta": float(counter.get("call_delta") or 0.0),
        "value_delta": float(counter.get("value_delta") or 0.0),
        "station": bool(counter.get("station")),
        "profile_id": best.get("id"),
    }
