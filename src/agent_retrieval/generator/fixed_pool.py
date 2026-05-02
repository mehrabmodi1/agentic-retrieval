from __future__ import annotations

import hashlib
import random
from typing import Any


def sample_fixed_pool(
    pool: list[dict[str, Any]],
    n: int,
    parametrisation_id: str,
    *,
    balance_key: str | None = None,
) -> list[dict[str, Any]]:
    """Deterministically sample n items from a fixed pool.

    Seeded by a hashlib digest of parametrisation_id so the same id always
    yields the same sample, *across Python processes*. CPython's built-in
    hash() of strings is randomized per-process unless PYTHONHASHSEED is set,
    so we cannot rely on it.

    Parameters
    ----------
    pool:
        The full pool of items to sample from.
    n:
        Number of items to sample.
    parametrisation_id:
        Stable identifier used to seed the RNG.
    balance_key:
        Optional item key for stratified sampling. When given, and n >= 2, and
        the pool contains at least 2 distinct values for that key, exactly 1
        item is picked from each of the first 2 groups (sorted by key value for
        reproducibility), then the remaining n-2 items are sampled uniformly
        from the rest of the pool. If n < 2 or the pool has only 1 distinct
        value for the key, falls back to plain random sampling. The result is
        shuffled with the same seeded RNG, so output is process-deterministic.
    """
    if n > len(pool):
        raise ValueError(f"n={n} exceeds pool size {len(pool)}")
    digest = hashlib.md5(parametrisation_id.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "big")
    rng = random.Random(seed)

    if balance_key is not None and n >= 2:
        groups: dict[str, list[dict[str, Any]]] = {}
        for item in pool:
            k = item[balance_key]
            groups.setdefault(k, []).append(item)
        sorted_keys = sorted(groups.keys())
        if len(sorted_keys) >= 2:
            # Pick 1 from each of the first 2 groups.
            chosen = [
                rng.choice(groups[sorted_keys[0]]),
                rng.choice(groups[sorted_keys[1]]),
            ]
            # Remaining items: everything except the two already chosen.
            remainder = [item for item in pool if item not in chosen]
            if n - 2 > 0:
                chosen += rng.sample(remainder, n - 2)
            rng.shuffle(chosen)
            return chosen

    return rng.sample(pool, n)


def sample_fixed_pool_l3(
    pool: list[dict[str, Any]],
    n: int,
    parametrisation_id: str,
) -> list[dict[str, Any]]:
    """Quadrant-balanced sampler for L3 (live × bound_direction).

    Pool items must each have boolean 'live' and string 'bound_direction'.
    For N>=4: 1 item per quadrant, then N-4 random across all 4 quadrants.
    For N==3: 1 (live, lower) + 1 (live, upper) + 1 random live item.
    For N==2: 1 (live, lower) + 1 (live, upper).
    Result is shuffled with the same seeded RNG (process-deterministic).
    """
    if n < 2:
        raise ValueError(f"sample_fixed_pool_l3 requires n>=2, got {n}")
    if n > len(pool):
        raise ValueError(f"n={n} exceeds pool size {len(pool)}")

    digest = hashlib.md5(parametrisation_id.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "big")
    rng = random.Random(seed)

    quadrants = {
        (True, "lower"): [],
        (True, "upper"): [],
        (False, "lower"): [],
        (False, "upper"): [],
    }
    for item in pool:
        quadrants[(item["live"], item["bound_direction"])].append(item)

    if n == 2:
        chosen = [
            rng.choice(quadrants[(True, "lower")]),
            rng.choice(quadrants[(True, "upper")]),
        ]
        rng.shuffle(chosen)
        return chosen

    if n == 3:
        chosen = [
            rng.choice(quadrants[(True, "lower")]),
            rng.choice(quadrants[(True, "upper")]),
        ]
        live_remainder = [
            it for it in pool if it["live"] and it not in chosen
        ]
        chosen.append(rng.choice(live_remainder))
        rng.shuffle(chosen)
        return chosen

    # n >= 4
    chosen = [
        rng.choice(quadrants[(True, "lower")]),
        rng.choice(quadrants[(True, "upper")]),
        rng.choice(quadrants[(False, "lower")]),
        rng.choice(quadrants[(False, "upper")]),
    ]
    remainder = [it for it in pool if it not in chosen]
    if n - 4 > 0:
        chosen += rng.sample(remainder, n - 4)
    rng.shuffle(chosen)
    return chosen
