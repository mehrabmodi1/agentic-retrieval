# Multi-Retrieval and Pure-Reasoning Experiments — Design

## Motivation

The existing `multi_reasoning` experiment couples two distinct cognitive demands on the agent:

1. **Retrieval+retention**: locating N independent pieces of information scattered across a large corpus and holding all N simultaneously through the search.
2. **Cross-fact reasoning**: combining the N retrieved facts into a single coherent answer (rank, classify, identify cascading constraints, derive intervals, etc.).

When `multi_reasoning` scores drop at higher N or larger corpora, the failure mode is ambiguous — was it a retrieval miss, a retention slip, or a reasoning failure? This spec introduces two new experiment types that *dissect* these two demands so they can be measured independently:

| Experiment | Tests | Strips |
|---|---|---|
| `multi_retrieval` | Retrieve and retain N category-coherent items in a large corpus | Cross-fact reasoning (question is pure recall) |
| `pure_reasoning` | Reason across N facts handed in the prompt | Retrieval (no corpus) |

These are complementary halves of the existing `multi_reasoning` task and are designed to be **parameter-lean** — the goal is a clean signal at the hardest cells, not exhaustive grid coverage.

---

## Shared design principles

Both experiments use **fixed pre-authored item pools** committed to the experiment YAML — not LLM-generated pools. This is a deliberate departure from the existing v2 generation pipeline:

- Each content_profile has exactly **16 hand-authored items** committed to the experiment YAML.
- For each `n_items` cell (2, 4, 8, 12, 16), the generator selects N items from the 16 by **seeded random sampling** (deterministic from `parametrisation_id`).
- The same item index always yields the same item across cells, so item identity is preserved across the n_items sweep.

Rationale: tight control. The 16 items determine the experiment's signal — they need to satisfy specific authoring constraints (category-coherence for retrieval, structural coherence for reasoning) that LLM-generated pools cannot guarantee. The cost is one-time authoring effort; the benefit is reproducible, interpretable results.

---

## Experiment 1: `multi_retrieval`

### Goal

Isolate the agent's ability to **retrieve and retain N category-coherent items simultaneously** in a large, distractor-rich corpus.

### Grid

```yaml
grid:
  content_profile: [python_repo, noir_fiction]
  corpus_token_count: [800000]
  discriminability: [hard]
  reference_clarity: [contextual]
  n_items: [2, 4, 8, 12, 16]
```

= **10 cells** (5 n_items × 2 profiles).

The corpus, discriminability, and reference_clarity axes are pinned to their hardest values from the existing `multi_reasoning` grid. The only swept axis is `n_items` — the load knob for retrieval+retention.

### Item categories (fixed across n_items per profile)

- **python_repo**: 16 numeric tuning parameters governing a **canary deployment / progressive rollout** subsystem (traffic split percentages, evaluation windows, regression-detection thresholds, auto-rollback latency multipliers, blast-radius caps, synthetic-traffic ratios, exposure ramp steps, holdback group sizes, abort cooldowns, drift tolerances, etc.). The category is unified by *function*, but item names are diverse — no single grep token unifies all 16. Canary-deployment is absent from the stock `python_repo` profile, so off-target hits are unlikely.

- **noir_fiction**: 16 prose passages that fix a specific time on the **night before the murder** (clocks chiming, train timetables, witness watch readings, light-vs-dark observations, radio program references, restaurant closing times, etc.). Wording varies — no single grep token unifies them. The "night before" framing makes off-target hits unlikely (native noir prose may contain time references, but rarely scoped to that specific night).

### Question template

The question reveals N and asks for verbatim text + value per item.

**python_repo (`hard_contextual`):**

> I have added N=`{n}` numeric tuning parameters to this codebase that govern a canary deployment / progressive rollout system. Find each one and report it as `<verbatim line of code> — value: <numeric value>`. Report all `{n}`.

**noir_fiction (`hard_contextual`):**

> I have added N=`{n}` passages to this story that fix a specific time on the night before the murder. Find each one and report it as `<verbatim passage text> — time referenced: <e.g., 9:45 PM>`. Report all `{n}`.

### Output format

Each item in the agent's answer must include:
- **Verbatim text**: the exact inserted string (line of code or passage).
- **Value**: the numeric value (python_repo) or extracted time (noir_fiction).

### Rubric

```yaml
rubric_criteria:
  - criterion: "recall"
    weight: 1.0
    # Fraction of the N inserted items correctly identified by both
    # verbatim text match AND value match.
  - criterion: "precision"
    weight: 0.3
    # Penalises false-positive items in the agent's answer that are not
    # in the answer-key set.
```

### Authoring constraints for the 16-item pool

For each profile, the 16 items must:
1. **Share the category** — every item plausibly belongs to the canary-deployment subsystem (python_repo) or fixes a time on the night before the murder (noir_fiction).
2. **Avoid a single unifying grep token** — items should not all literally contain "canary" or "the night before"; the unifying signal is functional/semantic.
3. **Be plausibly insertable** into the existing python_repo / noir_fiction file structure (canary parameters fit alongside existing config; timing passages fit into existing prose chapters).
4. **Have unambiguous values** — each item's "value" must be extractable into a normal form for scoring (numeric for python_repo, time-of-day for noir_fiction).

### Generation flow

1. Read the 16 items from the experiment YAML for the profile.
2. Seed RNG from `parametrisation_id`.
3. Sample N items without replacement.
4. Insert each into the corpus using the existing v2 insertion pipeline.
5. Write the answer key with the N selected items, plus the parametrised question (`{n}` substituted).

---

## Experiment 2: `pure_reasoning`

### Goal

Isolate the agent's ability to **reason across N facts** when all N are handed to it directly in the prompt — no corpus, no search, no retrieval.

### Grid

```yaml
grid:
  content_profile: [python_repo, noir_fiction]
  n_items: [2, 4, 8, 12, 16]
```

= **10 cells**. The corpus-related axes (`corpus_token_count`, `discriminability`, `reference_clarity`) are dropped entirely — there is no corpus.

### Question structure

A single coherent reasoning chain that produces a **structured interval-shaped answer** derived from all N facts. This mirrors the structural shape of `multi_reasoning`'s `hard_contextual` examples — single chain, not a battery of unrelated sub-questions.

The reasoning chain has three steps:
1. **Per-item classification**: for each of the N facts, infer whether it implies a *lower bound* or *upper bound* on a single derived quantity (a time window).
2. **Aggregate**: take `max` of lower bounds for one endpoint, `min` of upper bounds for the other endpoint.
3. **Output**: the narrowest defensible interval, with citations to the facts that establish each endpoint.

### Question template — python_repo

> You are running a database migration. It must execute during a window of system quiescence. Below are `{n}` numeric configuration parameters from across the service. Each implies a constraint on *when* the migration can safely begin and end relative to a reference event (the most recent backup, the next traffic peak, the next certificate rotation, the latest replication lag spike, etc.).
>
> Some parameters establish a *lower bound* on the migration start time (must wait at least X seconds after the reference event).
>
> Others establish an *upper bound* on the migration end time (must complete before X seconds after the reference event).
>
> [Numbered list of N facts]
>
> Derive the narrowest safe-migration window `[earliest_start, latest_end]` consistent with all `{n}` constraints. For each endpoint, cite the parameter that establishes it. Justify your classification of each parameter's bound direction.

### Question template — noir_fiction

> You are given `{n}` pieces of evidence from a homicide investigation, each pertaining to events on the **night of the murder**. Some pieces establish that the suspect was at large (free, unaccounted-for) at or after a specific time; others establish that the suspect was off-the-streets (accounted-for, with an alibi) at or before a specific time.
>
> [Numbered list of N facts]
>
> Derive the narrowest defensible time window during which the suspect was at large on the night of the murder. Cite the specific pieces of evidence that establish your window's lower and upper bounds. Justify your classification of each piece of evidence's bound direction.

### Note on bound direction in noir prose

The bound direction of a noir time-reference depends on language, not just the time itself: "the doorman watched him leave the bar at 9 PM" implies a lower bound on the suspect being at-large (he was free as of 9 PM); "his landlady saw him return home at 10 PM" implies an upper bound (he was off-the-streets by 10 PM). Per-item classification is a real reasoning step — the agent must interpret each piece of evidence's *implication*, not just extract a number.

Note this is distinct from `multi_retrieval`'s noir_fiction category, which concerns time references on the night *before* the murder. The two experiments use independent item pools.

### Prompt format

Light prose framing followed by a plain numbered list of facts:

```
[Question setup paragraph]

Below are the N facts:

1. <fact text>
2. <fact text>
...
N. <fact text>

[Question paragraph(s)]
```

No mock file blocks, no fake context — the agent's full burden is reasoning. No corpus is generated; the runner skips the corpus-loading step for this experiment type.

### Rubric

```yaml
rubric_criteria:
  - criterion: "endpoint_correctness"
    weight: 1.0
    # Both window endpoints (start and end) are correct, with correct
    # citation of which fact establishes each endpoint.
  - criterion: "classification_accuracy"
    weight: 0.5
    # The agent's classification of each of the N facts as a lower or
    # upper bound is correct (or correctly justified in the reasoning).
```

`classification_accuracy` is included to catch agents that guess the right endpoints by always picking extremes — it forces them to demonstrate per-item understanding.

### Authoring constraints for the 16-item pool

For each profile, the 16 items must:
1. **All bound the same single quantity** — the migration window (python_repo) or the suspect's at-large window (noir_fiction).
2. **Mix bound directions** — roughly half lower-bound facts and half upper-bound facts, so any reasonable subset has both endpoints defined.
3. **Be free of contradictions in the full pool** — `max(all 16 lowers) < min(all 16 uppers)`. This guarantees no subset can ever produce a contradictory interval (subsetting can only widen the safe window). See [Why no contradictions](#why-no-contradictions) below.
4. **Have ambiguous-but-resolvable bound directions** — the direction should be inferable only by reading the parameter's role / the evidence's language, not by a naming pattern (e.g., a `MAX_*` parameter that's actually a *lower* bound on the migration window because of some indirect relationship). This makes per-item classification non-trivial.
5. **Have a well-defined max-lower and min-upper among the 16** — so there's a deterministic answer key for n=16 and well-defined answers for any subset.

#### Why no contradictions

If the full pool's max-lower ≥ min-upper, then n=16 is always contradictory and smaller N is contradictory iff the offending pair is sampled — confounding the n_items axis with sampling luck. If the full pool is contradiction-free, no subset can ever produce a contradiction (proof: subset_max_lower ≤ pool_max_lower < pool_min_upper ≤ subset_min_upper).

### Generation flow

1. Read the 16 facts from the experiment YAML for the profile.
2. Seed RNG from `parametrisation_id`.
3. Sample N facts without replacement.
4. Skip corpus generation entirely.
5. Write the answer key with the N selected facts, the parametrised question (`{n}` substituted), and the expected endpoints derived from the selected subset.

The runner, when handed a `pure_reasoning` parametrisation, sends the question + facts directly to the agent without loading any corpus.

---

## Implementation outline

### Generator changes

- New experiment type `multi_retrieval`: existing v2 insertion pipeline reused, but the pool generation step is bypassed (items are read from the experiment YAML directly).
- New experiment type `pure_reasoning`: corpus and insertion stages bypassed entirely; only the answer key is generated.
- Common: a new `fixed_pool` block in the experiment YAML structure, keyed by `content_profile`, containing the 16 hand-authored items.

### Runner changes

- For `pure_reasoning` parametrisations, the runner skips the corpus-loading step and constructs the prompt directly from the answer key's facts + question template.
- For `multi_retrieval`, the runner behaves identically to other corpus-based experiments.

### Judge changes

- New rubric criteria: `recall`, `precision` (multi_retrieval); `endpoint_correctness`, `classification_accuracy` (pure_reasoning).
- The judge prompt for each new criterion describes how to score, with concrete pass/fail examples.

### Authoring deliverables (one-time)

The 16-item pools for both profiles, both experiments — **64 hand-authored items in total**:

| Profile | Experiment | Items |
|---|---|---|
| python_repo | multi_retrieval | 16 canary-deployment numeric parameters |
| noir_fiction | multi_retrieval | 16 night-before-the-murder timing passages |
| python_repo | pure_reasoning | 16 migration-window-bound parameters |
| noir_fiction | pure_reasoning | 16 night-of-the-murder at-large/alibi bounds |

These pools are the experimental signal source. They will be drafted carefully, reviewed, and committed to the respective experiment YAMLs.

---

## What's out of scope

- **Sweeping `corpus_token_count` or `discriminability` for `multi_retrieval`** — only the hardest cells are tested. If retrieval breaks down at the easiest cells too, that's a separate question.
- **Sweeping question shapes for `pure_reasoning`** — one question shape per profile (the migration window / at-large window). Other reasoning shapes (e.g., dependency-graph reasoning, contradiction detection over inconsistent pools) are deliberately out.
- **Comparing `multi_retrieval` × `pure_reasoning` items to existing `multi_reasoning` items** — items are independent. The dissection is structural (which cognitive demand is being tested), not item-level paired.
- **LLM-generated pools** — pools are hand-authored. No pool LLM agent is invoked.
