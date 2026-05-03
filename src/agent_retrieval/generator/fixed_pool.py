from __future__ import annotations

import hashlib
import random
from typing import Any


def _seeded_rng(parametrisation_id: str) -> random.Random:
    digest = hashlib.md5(parametrisation_id.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "big")
    return random.Random(seed)


def _split_binding(pool: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Partition pool into (binding, non_binding) by the optional `binding` flag."""
    binding = [it for it in pool if it.get("binding")]
    non_binding = [it for it in pool if not it.get("binding")]
    return binding, non_binding


def sample_fixed_pool(
    pool: list[dict[str, Any]],
    n: int,
    parametrisation_id: str,
    *,
    balance_key: str | None = None,
) -> list[dict[str, Any]]:
    """Deterministically sample n items from a fixed pool.

    Seeded by a hashlib digest of parametrisation_id so the same id always
    yields the same sample, *across Python processes*.

    If any pool items carry `binding: true`, those are always included in the
    output; the remaining n - k slots are sampled from non-binding items using
    `balance_key` if provided.
    """
    if n > len(pool):
        raise ValueError(f"n={n} exceeds pool size {len(pool)}")
    rng = _seeded_rng(parametrisation_id)

    binding, non_binding = _split_binding(pool)
    if len(binding) > n:
        raise ValueError(f"n={n} smaller than binding-item count {len(binding)}")

    chosen = list(binding)
    remaining = n - len(chosen)

    if remaining > 0:
        if balance_key is not None and remaining >= 2:
            # Group non-binding by balance_key, pick 1 from each of the first 2 groups,
            # then sample the rest randomly. Skip groups already covered by binding items
            # if any.
            covered_keys = {it[balance_key] for it in chosen}
            groups: dict[str, list[dict[str, Any]]] = {}
            for item in non_binding:
                k = item[balance_key]
                groups.setdefault(k, []).append(item)
            sorted_keys = sorted(groups.keys())
            uncovered_keys = [k for k in sorted_keys if k not in covered_keys]

            if len(uncovered_keys) >= 2 and remaining >= 2:
                picked = [
                    rng.choice(groups[uncovered_keys[0]]),
                    rng.choice(groups[uncovered_keys[1]]),
                ]
                chosen += picked
                remainder_pool = [it for it in non_binding if it not in picked]
                if remaining - 2 > 0:
                    chosen += rng.sample(remainder_pool, remaining - 2)
            else:
                chosen += rng.sample(non_binding, remaining)
        else:
            chosen += rng.sample(non_binding, remaining)

    rng.shuffle(chosen)
    return chosen


def sample_fixed_pool_l3(
    pool: list[dict[str, Any]],
    n: int,
    parametrisation_id: str,
) -> list[dict[str, Any]]:
    """Sampler for L3 (live × bound_direction).

    Pool items must each have boolean 'live' and string 'bound_direction'.

    If any pool items are marked `binding: true` (must be live), they are always
    included; remaining slots are filled with non-binding items, covering both
    dead quadrants first and then random fill.

    With binding pair (1 live-lower + 1 live-upper) marked:
    - N=2: binding pair only.
    - N=3: binding pair + 1 random non-binding live item.
    - N>=4: binding pair + 1 (dead, lower) + 1 (dead, upper) + N-4 random non-binding.

    With no binding items marked (legacy behaviour, used by tests):
    - N=2: 1 (live, lower) + 1 (live, upper).
    - N=3: 1 (live, lower) + 1 (live, upper) + 1 random live item.
    - N>=4: 1 from each of 4 quadrants + N-4 random.
    """
    if n < 2:
        raise ValueError(f"sample_fixed_pool_l3 requires n>=2, got {n}")
    if n > len(pool):
        raise ValueError(f"n={n} exceeds pool size {len(pool)}")

    rng = _seeded_rng(parametrisation_id)

    binding, non_binding = _split_binding(pool)
    if len(binding) > n:
        raise ValueError(f"n={n} smaller than binding-item count {len(binding)}")
    for it in binding:
        if not it.get("live"):
            raise ValueError("binding items in L3 pool must be live")

    quadrants = {
        (True, "lower"): [],
        (True, "upper"): [],
        (False, "lower"): [],
        (False, "upper"): [],
    }
    for it in pool:
        quadrants[(it["live"], it["bound_direction"])].append(it)

    if not binding:
        # Legacy quadrant-balanced behaviour.
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
            live_remainder = [it for it in pool if it["live"] and it not in chosen]
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

    # Binding-pair behaviour.
    chosen = list(binding)
    remaining = n - len(chosen)

    if remaining == 0:
        rng.shuffle(chosen)
        return chosen

    nb_quadrants = {key: [it for it in items if it not in chosen] for key, items in quadrants.items()}

    if remaining == 1:
        live_pool = nb_quadrants[(True, "lower")] + nb_quadrants[(True, "upper")]
        chosen.append(rng.choice(live_pool))
        rng.shuffle(chosen)
        return chosen

    # remaining >= 2: cover both dead quadrants first, then random fill.
    chosen.append(rng.choice(nb_quadrants[(False, "lower")]))
    chosen.append(rng.choice(nb_quadrants[(False, "upper")]))
    if remaining - 2 > 0:
        leftover = [it for it in pool if it not in chosen]
        chosen += rng.sample(leftover, remaining - 2)

    rng.shuffle(chosen)
    return chosen
