from __future__ import annotations

import random
from typing import Any


def sample_fixed_pool(
    pool: list[dict[str, Any]],
    n: int,
    parametrisation_id: str,
) -> list[dict[str, Any]]:
    """Deterministically sample n items from a fixed pool.

    Seeds an RNG from the parametrisation_id so the same id always yields
    the same sample. Items are returned in shuffled order.
    """
    if n > len(pool):
        raise ValueError(f"n={n} exceeds pool size {len(pool)}")
    rng = random.Random(hash(parametrisation_id) ^ 0xCAFE)
    return rng.sample(pool, n)
