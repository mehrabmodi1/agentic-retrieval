from __future__ import annotations

import hashlib
import random
from typing import Any


def sample_fixed_pool(
    pool: list[dict[str, Any]],
    n: int,
    parametrisation_id: str,
) -> list[dict[str, Any]]:
    """Deterministically sample n items from a fixed pool.

    Seeded by a hashlib digest of parametrisation_id so the same id always
    yields the same sample, *across Python processes*. CPython's built-in
    hash() of strings is randomized per-process unless PYTHONHASHSEED is set,
    so we cannot rely on it.
    """
    if n > len(pool):
        raise ValueError(f"n={n} exceeds pool size {len(pool)}")
    digest = hashlib.md5(parametrisation_id.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "big")
    rng = random.Random(seed)
    return rng.sample(pool, n)
