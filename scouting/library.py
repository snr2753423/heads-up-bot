"""Counters for profiles whose behavior is known. The bot loads the JSON, not this module."""

from __future__ import annotations

# The playing bot does not stack these on a live shift that already points the same way.
# Negative bet_delta means bet wider. Positive call_delta means call tighter.
COUNTERS = {
    "starter_checkfold": {
        "usable_for_match": True,
        "signature": {"fold_rate": 0.90, "raise_rate": 0.02, "pot_share": 0.0},
        "counter": {"bet_delta": -0.10, "call_delta": 0.0, "value_delta": 0.0, "station": False},
        "why": "They check when it is free and fold when it costs chips, so betting into them wins the pot often.",
    },
    "calling_station": {
        "usable_for_match": True,
        "signature": {"fold_rate": 0.02, "raise_rate": 0.02, "pot_share": 0.0},
        "counter": {"bet_delta": 0.06, "call_delta": -0.02, "value_delta": -0.05, "station": True},
        "why": "They never fold and never raise, so bluffs lose and thin value bets get paid.",
    },
    "always_half": {
        "usable_for_match": True,
        "signature": {"fold_rate": 0.02, "raise_rate": 0.85, "pot_share": 0.05},
        "counter": {"bet_delta": 0.0, "call_delta": 0.0, "value_delta": 0.0, "station": False},
        "why": "They raise half the pot with any cards. Showdown quality, not the raise itself, decides whether a call tightens or loosens.",
    },
    "always_pot": {
        "usable_for_match": True,
        "signature": {"fold_rate": 0.02, "raise_rate": 0.85, "pot_share": 0.95},
        "counter": {"bet_delta": 0.0, "call_delta": 0.0, "value_delta": 0.0, "station": False},
        "why": "They pot it with any cards. The price is high, and the live showdown read decides the call. A stored tighten would fold the hands that beat their air.",
    },
    "equity_threshold": {
        "usable_for_match": False,
        "signature": {"fold_rate": 0.42, "raise_rate": 0.30, "pot_share": 0.55},
        "counter": {"bet_delta": -0.03, "call_delta": 0.03, "value_delta": 0.0, "station": False},
        "why": "Published parameters value-bet and fold to a price. The signature is an inference from one public snapshot, so it is stored and not auto-matched.",
    },
}
