"""Vectorised Short Deck Omaha scores.

Same ranking as evaluator.omaha_score. Every pair of hole cards is scored
with every disjoint three-card board once, at import. A decision is then
a table gather instead of a hundred Python dict lookups.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

_N = 36
_FIVE_N = 376992  # C(36, 5)

# C(n, k) for n <= 36 and k <= 5. Lexicographic combination rank.
_BINOM = np.zeros((37, 6), dtype=np.int32)
_BINOM[:, 0] = 1
for _n in range(1, 37):
    for _k in range(1, 6):
        _BINOM[_n, _k] = _BINOM[_n - 1, _k] + _BINOM[_n - 1, _k - 1]

# Which of the five cards form each hole pair and each board triple.
_PAIRS = np.array(list(combinations(range(5), 2)), dtype=np.intp)
_TRIPS = np.array(list(combinations(range(5), 3)), dtype=np.intp)


def lex_index(cards: np.ndarray) -> np.ndarray:
    """Lexicographic rank of sorted 5-card rows. Shape cards is (..., 5)."""
    return _lex(cards, 5)


def lex_index_py(cards: tuple[int, ...]) -> int:
    """Same rank as lex_index, for one sorted 5-card hand."""
    idx = 0
    prev = -1
    for i, card in enumerate(cards):
        r1 = 5 - i
        idx += int(_BINOM[_N - prev - 1, r1] - _BINOM[_N - card, r1])
        prev = card
    return idx


def _lex(cards: np.ndarray, k: int) -> np.ndarray:
    idx = np.zeros(cards.shape[:-1], dtype=np.int32)
    prev = np.full(cards.shape[:-1], -1, dtype=np.int32)
    for i in range(k):
        card = cards[..., i]
        r1 = k - i
        idx += _BINOM[_N - prev - 1, r1] - _BINOM[_N - card, r1]
        prev = card
    return idx


def _build_five() -> np.ndarray:
    from evaluator import get_table

    table = get_table()
    out = np.empty(_FIVE_N, dtype=np.int32)
    for i, comb in enumerate(combinations(range(_N), 5)):
        mask = (1 << comb[0]) | (1 << comb[1]) | (1 << comb[2]) | (1 << comb[3]) | (1 << comb[4])
        out[i] = table[mask]
    return out


def _build_pair_trip(five: np.ndarray) -> np.ndarray:
    """pair index x 3-card index -> five-card score. Overlaps stay 0."""
    pairs = np.array(list(combinations(range(_N), 2)), dtype=np.int32)
    trips = np.array(list(combinations(range(_N), 3)), dtype=np.int32)
    table = np.zeros((len(pairs), len(trips)), dtype=np.int32)
    step = 20
    for start in range(0, len(pairs), step):
        chosen = pairs[start:start + step]
        rows = len(chosen)
        left = np.broadcast_to(chosen[:, None, :], (rows, len(trips), 2)).copy()
        right = np.broadcast_to(trips[None, :, :], (rows, len(trips), 3)).copy()
        cards = np.concatenate((left, right), axis=-1)
        cards.sort(axis=-1)
        overlap = (cards[..., 1:] == cards[..., :-1]).any(axis=-1)
        scores = np.zeros((rows, len(trips)), dtype=np.int32)
        ok = ~overlap
        scores[ok] = five[lex_index(cards[ok])]
        table[start:start + step] = scores
    return table


_PAIR_TRIP = _build_pair_trip(_build_five())


def batch_omaha(holes: np.ndarray, boards: np.ndarray) -> np.ndarray:
    """Best exact-2-hole + exact-3-board score for each row.

    holes and boards are (N, 5) card ints, or a single hand of 5.
    Higher is better.
    """
    holes = np.asarray(holes, dtype=np.int32)
    boards = np.asarray(boards, dtype=np.int32)
    single = holes.ndim == 1
    if single:
        holes = holes.reshape(1, 5)
        boards = boards.reshape(1, 5)
    pair_cards = np.sort(holes[:, _PAIRS], axis=-1)
    trip_cards = np.sort(boards[:, _TRIPS], axis=-1)
    pair_i = _lex(pair_cards, 2)
    trip_i = _lex(trip_cards, 3)
    scores = _PAIR_TRIP[pair_i[:, :, None], trip_i[:, None, :]].max(axis=(1, 2))
    if single:
        return scores[0]
    return scores


def batch_made(holes: np.ndarray, board: np.ndarray) -> np.ndarray:
    """Best hand using the board cards dealt so far. holes is (N, 5) or (5,)."""
    holes = np.asarray(holes, dtype=np.int32)
    board = np.asarray(board, dtype=np.int32)
    single = holes.ndim == 1
    if single:
        holes = holes.reshape(1, 5)
    pair_i = _lex(np.sort(holes[:, _PAIRS], axis=-1), 2)
    if board.size == 5:
        trip_cards = np.sort(board[_TRIPS], axis=-1)
    elif board.size == 4:
        picks = np.array(list(combinations(range(4), 3)), dtype=np.intp)
        trip_cards = np.sort(board[picks], axis=-1)
    elif board.size == 3:
        trip_cards = np.sort(board)[None, :]
    else:
        raise ValueError("board must have 3, 4 or 5 cards")
    trip_i = _lex(trip_cards, 3)
    scores = _PAIR_TRIP[pair_i[:, :, None], trip_i[None, None, :]].max(axis=(1, 2))
    if single:
        return scores[0]
    return scores
