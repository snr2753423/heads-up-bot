"""Build data/preflop_equity.npy. Offline only. Do not import this from the bot.

Row i is the equity of the i-th hand in itertools.combinations(range(36), 5).
Equity is wins plus half the ties versus a random opposing hand and board.
"""

from __future__ import annotations

import os
import random
import sys
import time
from itertools import combinations
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from evaluator import omaha_score, warmup  # noqa: E402

SAMPLES = 40
OUT = os.path.join(ROOT, "data", "preflop_equity.npy")


def estimate(hole, rng, samples=SAMPLES):
    deck = [card for card in range(36) if card not in hole]
    wins = ties = 0
    hole = list(hole)
    for _ in range(samples):
        drawn = rng.sample(deck, 10)
        board = drawn[5:]
        hero = omaha_score(hole, board)
        villain = omaha_score(drawn[:5], board)
        if hero > villain:
            wins += 1
        elif hero == villain:
            ties += 1
    return (wins + 0.5 * ties) / samples


def _init():
    warmup()


def _chunk(payload):
    start, hands, seed = payload
    rng = random.Random(seed)
    return start, [estimate(hand, rng) for hand in hands]


def main():
    import numpy as np

    hands = list(combinations(range(36), 5))
    print(f"hands {len(hands)} samples {SAMPLES}", flush=True)
    warmup()
    t0 = time.perf_counter()
    probe = estimate(hands[0], random.Random(0), samples=5)
    per = (time.perf_counter() - t0) / 5
    print(f"probe equity {probe:.3f}  {per * 1000:.2f} ms/sample", flush=True)

    workers = min(8, os.cpu_count() or 1)
    chunk = 4096
    jobs = []
    for start in range(0, len(hands), chunk):
        jobs.append((start, hands[start:start + chunk], 10_000 + start))
    equity = np.empty(len(hands), dtype=np.float32)
    done = 0
    started = time.perf_counter()
    with Pool(workers, initializer=_init) as pool:
        for start, values in pool.imap_unordered(_chunk, jobs, chunksize=1):
            equity[start:start + len(values)] = values
            done += len(values)
            if done % (chunk * 4) < chunk or done == len(hands):
                elapsed = time.perf_counter() - started
                print(f"{done}/{len(hands)}  {elapsed:.0f}s", flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.save(OUT, equity)
    print(f"wrote {OUT} mean {float(equity.mean()):.4f} min {float(equity.min()):.3f} max {float(equity.max()):.3f}", flush=True)


if __name__ == "__main__":
    main()
