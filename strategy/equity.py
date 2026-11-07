"""Preflop lookup and postflop Monte Carlo.

The preflop table is in itertools.combinations(range(36), 5) order.
Its index is the same lexicographic rank as strategy.score.lex_index,
so the 376,992-entry dict is never built during a hand.

Sample counts shrink when the half's time bank runs down. A decision
that would empty the bank is worse than a coarser equity.
"""

from __future__ import annotations

import os

import numpy as np

from strategy.score import batch_made, batch_omaha, lex_index_py

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLE_PATH = os.path.join(_ROOT, "data", "preflop_equity.npy")

_EQUITY: np.ndarray | None = None
_RNG = np.random.default_rng()

# Full-budget sample counts. Missing board cards -> rows.
# 0 river, 1 turn, 2 flop. A few hundred rows is about ±2 to ±3 points
# of equity, which is inside the gaps between the betting thresholds.
_FULL = {0: 480, 1: 320, 2: 240}


def _load() -> None:
    global _EQUITY
    if _EQUITY is not None:
        return
    if os.path.isfile(TABLE_PATH):
        _EQUITY = np.load(TABLE_PATH)
    else:
        _EQUITY = None


def preflop_equity(hole: tuple[int, ...]) -> float:
    """Equity of these 5 hole cards versus a random hand. 0.50 if no table."""
    if _EQUITY is None:
        return 0.5
    cards = tuple(sorted(int(c) for c in hole))
    row = lex_index_py(cards)
    if row < 0 or row >= len(_EQUITY):
        return 0.5
    return float(_EQUITY[row])


def samples_for(memory: dict | None, missing: int) -> int:
    """How many Monte Carlo rows to draw for this street.

    `missing` is 5 minus the number of board cards (0 river, 1 turn, 2 flop).
    A fresh half uses the full count. The count steps down as the bank the
    bot has measured for itself runs out, so the judge's 8 second bank does not.
    """
    full = _FULL.get(missing, 400)
    if not memory:
        return full
    spent = float(memory.get("spent") or 0.0)
    bank = float(memory.get("bank") or 8.0)
    left = bank - spent
    if left <= 2.0:
        return 16
    if left <= 3.5:
        return max(40, full // 5)
    if left <= 5.5:
        return max(80, full // 2)
    return full


def _draw(deck: np.ndarray, rows: int, take: int) -> np.ndarray:
    noise = _RNG.random((rows, deck.size))
    pick = np.argpartition(noise, take - 1, axis=1)[:, :take]
    return deck[pick]


def postflop_equity(
    hole: tuple[int, ...],
    board: tuple[int, ...],
    memory: dict | None = None,
    keep: float = 1.0,
) -> float:
    """Win equity plus half the ties.

    `keep` is the fraction of sampled hands, the strongest on the current
    board, that count. A bet is a stronger range than a random hand, so
    keep is below 1 when the opponent has put chips in.
    """
    board_n = len(board)
    missing = 5 - board_n
    if missing < 0 or missing > 2 or len(hole) != 5:
        return 0.5
    hole_a = np.asarray(hole, dtype=np.int32)
    board_a = np.asarray(board, dtype=np.int32)
    dead = np.concatenate((hole_a, board_a))
    deck = np.setdiff1d(np.arange(36, dtype=np.int32), dead, assume_unique=False)
    take = 5 + missing
    if deck.size < take:
        return 0.5
    rows = samples_for(memory, missing)
    drawn = _draw(deck, rows, take)
    boards = np.empty((rows, 5), dtype=np.int32)
    boards[:, :board_n] = board_a
    if missing:
        boards[:, board_n:] = drawn[:, 5:take]
    opp = drawn[:, :5]
    heroes = np.broadcast_to(hole_a, (rows, 5))
    hero_s = batch_omaha(heroes, boards)
    vill_s = batch_omaha(opp, boards)
    if keep < 0.999 and rows > 1:
        strength = batch_made(opp, board_a)
        kept = max(1, int(round(rows * keep)))
        if kept < rows:
            chosen = np.argpartition(strength, rows - kept)[rows - kept:]
            hero_s = hero_s[chosen]
            vill_s = vill_s[chosen]
            rows = kept
    wins = int(np.count_nonzero(hero_s > vill_s))
    ties = int(np.count_nonzero(hero_s == vill_s))
    if rows == 0:
        return 0.5
    return (wins + 0.5 * ties) / rows


_load()
