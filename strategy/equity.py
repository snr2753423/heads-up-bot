"""Preflop lookup and a small postflop Monte Carlo.

The builder and this loader both index hole cards with
itertools.combinations(range(36), 5). Do not change that order.
"""

from __future__ import annotations

import os
import random
from itertools import combinations

from evaluator import omaha_score

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLE_PATH = os.path.join(_ROOT, "data", "preflop_equity.npy")

_INDEX: dict[tuple[int, ...], int] | None = None
_EQUITY = None
_RNG = random.Random()

# Sample counts stay small so a half stays well inside the 8 second bank.
_FLOP_OPPONENTS = 8
_FLOP_RUNOUTS = 6
_TURN_OPPONENTS = 8
_TURN_RUNOUTS = 4
_RIVER_OPPONENTS = 36


def _load() -> None:
    global _INDEX, _EQUITY
    if _EQUITY is not None:
        return
    _INDEX = {combo: i for i, combo in enumerate(combinations(range(36), 5))}
    if os.path.isfile(TABLE_PATH):
        import numpy as np

        _EQUITY = np.load(TABLE_PATH)
    else:
        _EQUITY = None


def preflop_equity(hole: tuple[int, ...]) -> float:
    """Equity of these 5 hole cards versus a random hand. 0.50 if no table."""
    _load()
    key = tuple(sorted(hole))
    if _EQUITY is None:
        return 0.5
    row = _INDEX.get(key) if _INDEX is not None else None
    if row is None or row >= len(_EQUITY):
        return 0.5
    return float(_EQUITY[row])


def postflop_equity(hole: tuple[int, ...], board: tuple[int, ...]) -> float:
    """Win equity plus half the ties, sampled from the cards still left."""
    missing = 5 - len(board)
    if missing < 0 or missing > 2:
        return 0.5
    dead = set(hole) | set(board)
    deck = [card for card in range(36) if card not in dead]
    if missing == 0:
        opponents, runouts = _RIVER_OPPONENTS, 1
    elif missing == 1:
        opponents, runouts = _TURN_OPPONENTS, _TURN_RUNOUTS
    else:
        opponents, runouts = _FLOP_OPPONENTS, _FLOP_RUNOUTS
    take = 5 + missing
    if len(deck) < take:
        return 0.5
    wins = ties = scored = 0
    hole_list = list(hole)
    board_list = list(board)
    for _ in range(opponents * runouts):
        drawn = _RNG.sample(deck, take)
        full_board = board_list + drawn[5:take]
        hero = omaha_score(hole_list, full_board)
        villain = omaha_score(drawn[:5], full_board)
        scored += 1
        if hero > villain:
            wins += 1
        elif hero == villain:
            ties += 1
    if scored == 0:
        return 0.5
    return (wins + 0.5 * ties) / scored
