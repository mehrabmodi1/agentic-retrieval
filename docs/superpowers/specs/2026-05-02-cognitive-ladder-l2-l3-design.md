# Cognitive Ladder for `pure_reasoning`: L2 (Mixed Units) and L3 (Conditional Gating)

**Goal:** Extend the existing `pure_reasoning` task with two cognitively layered variants — L2 adds per-item unit conversion, L3 adds conditional gating *on top of* L2 — to localise where the multi-reasoning n-items cliff originates.

**Scope:** This spec covers L2 and L3 only. L4 (transitive ordering) and L5 (iterative refinement) are deferred until L2/L3 results are in.

**Approach:** **Cumulative layering.** Each rung adds one cognitive primitive on top of all prior rungs.
- L2 inherits L1's 16-item pool; rewrites each item's surface form into one of 3 unit/representation variants.
- L3 inherits L2's pool **exactly** (same items, same variants); adds a precondition gate to each item plus a world-state block at the prompt top.

## L1 recap (existing — not modified)

- 16-item pool per content profile (`python_repo`, `noir_fiction`); 8 lower + 8 upper bounds.
- Pool consistency: `max(lower values) < min(upper values)`.
- N items sampled per cell with stratification on `bound_direction` (first 2 balanced, rest random).
- Question: classify each fact as upper/lower bound, output narrowest window.

## L2 — Mixed Units

### Variant assignment

Each L1 item is rewritten into one of 3 unit/representation variants, deterministic by item-index modulo 3 so the variant distribution stays balanced across N for any sampled subset.

**Python repo variants:**
1. **Canonical** — `<NAME>_S = <value>  # seconds` (matches L1 form)
2. **Alt-unit** — `<NAME>_MIN = <value/60>` or `<NAME>_MS = <value*1000>`
3. **Offset-from-reference** — `<NAME>_OFFSET_S = <delta>  # seconds before <REFERENCE>`. The reference's value is given inline in the same item's prose (e.g. *"Reference: `HOURLY_KICK_S = 3600`. Then `BACKUP_DEADLINE_OFFSET_S = 600  # before HOURLY_KICK`"*).

**Noir fiction variants:**
1. **Canonical** — clock-time prose ("the diary entry read 'half-past ten'")
2. **Alt-unit** — 24-hour clock or "twenty minutes before eleven"
3. **Offset-from-anchor** — "Forty-five minutes after the last streetcar ran". The anchor's clock time is stated inline within the same item ("(the streetcar ran at 9:30 PM)").

Anchors are kept **self-contained per item**, not pulled into a shared references block. Trades off realism for generation/sampling simplicity.

### Pool item shape

Adds two fields beyond L1:
- `variant_id: int` — 0/1/2
- `variant_text: str` — fully self-contained surface form

`value` and `bound_direction` stay canonical (seconds for python; minutes-since-midnight for noir).

### Question prompt

Append to L1 question:
> *"Some constraints are stated in mixed units or relative to a named reference. Normalise to a common scale (seconds / clock time) before reasoning."*

### Answer key

Same shape as L1 plus per-item `variant_id` and `variant_text`. Judge scores against canonical `value` and `bound_direction`.

### Sampling

Stratify on `bound_direction` (same as L1). Variant balance is preserved automatically by the deterministic per-index assignment.

## L3 — Conditional Gating

### Pool inherits L2 exactly

L3 items are L2 items (same `variant_text`) with a precondition gate prepended.

### Quadrant stratification

The 16 items are partitioned into a 4-quadrant grid:
- 4 × (live, lower) — gate true, item enforces lower bound
- 4 × (live, upper)
- 4 × (dead, lower) — gate false, item should be ignored
- 4 × (dead, upper)

The 8 lower / 8 upper split from L1 is preserved; liveness is assigned deterministically per item-index.

### Pool item shape

Adds three fields beyond L2:
- `gate_clause: str` — e.g. *"If `current_mode == 'rolling'`,"*
- `gate_world_var: str` — name of the world-state variable referenced
- `live: bool`

The item's rendered prompt text is `<gate_clause> <variant_text>`.

### World state block

Pool authors **4 distinct gate variables** per profile, each referenced by exactly 4 of the 16 items. Because a variable's truth value is global (the world state is shared by all items referencing it), each variable's 4 items must all be live *or* all dead. The fixed assignment is:
- 2 gate variables are TRUE in the world state — together owning 8 live items (4 lower + 4 upper, distributed so each TRUE var owns 2 lower + 2 upper)
- 2 gate variables are FALSE — together owning 8 dead items (same 2L+2U split)

This is a static authoring decision; the world state is not randomised at run time. The full world-state block is always printed in the prompt regardless of which variables the sampled subset references.

Example python world-state block:
```
World state:
- current_mode = "rolling"
- replica_promoted = False
- dual_write_window_open = True
- canary_traffic_capped = True
```

Variables not referenced by any sampled item are still printed (full block always shown), so the agent can't infer relevance from the block size.

### Question prompt

Append to L2 question:
> *"Some constraints are conditional on the world state given above. Ignore constraints whose preconditions are not satisfied. Compute the window from live constraints only."*

### Answer key

Same shape as L2 plus per-item `live: bool`. Window endpoints derived from live items only. Judge rubric should test:
- agent ignored dead items (false-positive penalty if it cited a dead item)
- agent classified live items correctly
- agent reported correct window endpoints from the live subset

### Sampling

`balance_key=(bound_direction, live)`. First 4 items stratified across the 4 quadrants; remaining N-4 items random.

For N<4 we sample only from the **live** quadrants (so the puzzle has a defined answer):
- N=2: 1× (live, lower) + 1× (live, upper)
- N=3: 1× (live, lower) + 1× (live, upper) + 1 random live item
- N≥4: full 4-quadrant stratification on first 4, then random across all 4 quadrants.

### Internal consistency at sample time

Pool consistency for L3 must hold over the **live subset** of any sampled N items: `max(live lower values) < min(live upper values)`. Since the L1 pool already satisfies this globally and L3 just labels a subset live, this is automatic.

## Pool generation pipeline

**Step 1 (one-time, hand-authored):** Author L2 unit variants — one YAML per profile under `workspace/pure_reasoning_l2_pool/{python_repo,noir_fiction}.yaml`. Each file: 16 entries × `(variant_id, variant_text, value, bound_direction)`.

**Step 2 (one-time, hand-authored):** Author L3 gate fields — `workspace/pure_reasoning_l3_pool/{python_repo,noir_fiction}.yaml`. Inherits items from L2; adds `(gate_clause, gate_world_var, live)`. 4 distinct gate variables per profile, each referenced by 4 items balanced across quadrants.

**Step 3 (programmatic, per generation):** Extend `generate_pure_reasoning_cell` to dispatch by level — pick correct pool, sample per the level's `balance_key` spec, render the level-appropriate question, world-state block (L3 only), and AK.

## Identifiers and templates

| Level | Parametrisation ID prefix | Template file |
|---|---|---|
| L1 (existing) | `pure_reasoning__<profile>__n<N>` | `experiments/pure_reasoning.yaml` |
| L2 (new) | `pure_reasoning_l2__<profile>__n<N>` | `experiments/pure_reasoning_l2.yaml` |
| L3 (new) | `pure_reasoning_l3__<profile>__n<N>` | `experiments/pure_reasoning_l3.yaml` |

L1 keeps its existing identifier — no rename, no risk of invalidating the 60 verdicts from yesterday's run.

## Out of scope

- L4 (transitive ordering) and L5 (iterative refinement) — separate spec when L2/L3 results land.
- Any modifications to existing L1 (`pure_reasoning`) pool, prompt, or AK.
- `multi_retrieval` cognitive layering — retrieval-only arm; cognitive ladder applies to reasoning only.
- Schema changes beyond two new optional pool-item fields per level.

## Open implementation choices

These are spelled out here so they're not re-litigated during planning:

1. **Anchor scope (L2 variant 3):** Self-contained anchors inline per item, not a shared block.
2. **L3 world variable count:** 4 per profile — keeps the world-state block compact while enforcing diversity.
3. **L3 world block always shown in full** — even if some variables are unreferenced by the sampled subset, to prevent size-based inference.
4. **L3 gate truth at sample time:** truth values fixed at generation so sampled live items resolve true and sampled dead items resolve false. (No randomisation of world state per run.)
