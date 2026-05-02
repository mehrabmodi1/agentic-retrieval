# Cognitive Ladder L2 + L3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two new experiment types — `pure_reasoning_l2` (mixed units) and `pure_reasoning_l3` (conditional gating layered on L2) — that extend the existing `pure_reasoning` task by cumulatively layering cognitive primitives, so we can localise where the multi-reasoning n-items cliff originates.

**Architecture:** L2 inherits L1's 16-item fixed pool per profile and rewrites each item's surface form into one of three unit/representation variants (deterministic by item-index). L3 inherits L2's items exactly and adds a precondition gate per item plus a world-state block printed at the top of the prompt; sampling enforces a 4-quadrant balance (live × dead × lower × upper) with a live-only fallback for N<4. Both new types reuse `generate_pure_reasoning_cell`'s pipeline (no LLM, no corpus assembly — just deterministic AK writing). New experiment templates live alongside the L1 one in `experiments/`, and a small dispatcher tweak in the parallel generator routes them to the same code path.

**Tech Stack:** Python 3.12+, Poetry, pytest, pytest-asyncio, Pydantic v2, PyYAML.

**Spec reference:** [docs/superpowers/specs/2026-05-02-cognitive-ladder-l2-l3-design.md](docs/superpowers/specs/2026-05-02-cognitive-ladder-l2-l3-design.md)

---

## File Map

**Create:**
- `experiments/pure_reasoning_l2.yaml` — L2 template (question, grid, fixed_pool with 16 unit-variant items × 2 profiles)
- `experiments/pure_reasoning_l3.yaml` — L3 template (question with `{world_state_block}` placeholder, grid, fixed_pool with 16 gate-augmented items × 2 profiles)
- `tests/test_fixed_pool_l3.py` — tests for `sample_fixed_pool_l3`
- `tests/test_pure_reasoning_l2_l3.py` — tests for L2/L3 dispatch in `generate_pure_reasoning_cell`
- `batches/cognitive-ladder_opus-4-7_effort-max.yaml`
- `batches/cognitive-ladder_opus-4-6_effort-max.yaml`
- `batches/cognitive-ladder_sonnet-4-6_effort-max.yaml`

**Modify:**
- `src/agent_retrieval/schema/template.py` — extend `experiment_type` Literal with two new values; extend `validate_grid_n_items`'s `multi_types` set
- `src/agent_retrieval/generator/fixed_pool.py` — add `sample_fixed_pool_l3` function
- `src/agent_retrieval/generator/pure_reasoning_gen.py` — dispatch L2 / L3 in `generate_pure_reasoning_cell`
- `scripts/generate_parallel.py` — broaden the `pure_reasoning` dispatch check to match all three experiment types

---

## Task 1: Schema literal extension

**Files:**
- Modify: `src/agent_retrieval/schema/template.py:35-41` and `:50`
- Test: `tests/test_template_schema.py` (add)

- [ ] **Step 1: Write the failing test**

Add to `tests/test_template_schema.py`:

```python
def test_pure_reasoning_l2_is_valid_experiment_type():
    template = ExperimentTemplate.model_validate({
        "experiment_type": "pure_reasoning_l2",
        "payload": {"item_type": "fact"},
        "question_examples": {"python_repo": {"hard_contextual": {"question": "q", "answer": "a"}}},
        "rubric_criteria": [{"criterion": "endpoint_correctness", "weight": 1.0}],
        "grid": {"content_profile": ["python_repo"], "n_items": [2]},
        "fixed_pool": {"python_repo": []},
    })
    assert template.experiment_type == "pure_reasoning_l2"


def test_pure_reasoning_l3_is_valid_experiment_type():
    template = ExperimentTemplate.model_validate({
        "experiment_type": "pure_reasoning_l3",
        "payload": {"item_type": "fact"},
        "question_examples": {"python_repo": {"hard_contextual": {"question": "q", "answer": "a"}}},
        "rubric_criteria": [{"criterion": "endpoint_correctness", "weight": 1.0}],
        "grid": {"content_profile": ["python_repo"], "n_items": [2]},
        "fixed_pool": {"python_repo": []},
    })
    assert template.experiment_type == "pure_reasoning_l3"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `poetry run pytest tests/test_template_schema.py::test_pure_reasoning_l2_is_valid_experiment_type tests/test_template_schema.py::test_pure_reasoning_l3_is_valid_experiment_type -v`
Expected: FAIL with Pydantic validation error (literal mismatch).

- [ ] **Step 3: Extend the Literal and `multi_types`**

Edit `src/agent_retrieval/schema/template.py`:

```python
    experiment_type: Literal[
        "single_needle",
        "multi_chain",
        "multi_reasoning",
        "multi_retrieval",
        "pure_reasoning",
        "pure_reasoning_l2",
        "pure_reasoning_l3",
    ]
```

And in `validate_grid_n_items`:

```python
        multi_types = {
            "multi_chain", "multi_reasoning", "multi_retrieval",
            "pure_reasoning", "pure_reasoning_l2", "pure_reasoning_l3",
        }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `poetry run pytest tests/test_template_schema.py -v`
Expected: PASS for the two new tests; existing tests still pass.

- [ ] **Step 5: Commit**

```bash
git add src/agent_retrieval/schema/template.py tests/test_template_schema.py
git commit -m "schema: accept pure_reasoning_l2 and pure_reasoning_l3 experiment types"
```

---

## Task 2: `sample_fixed_pool_l3` quadrant-balanced sampler

**Files:**
- Modify: `src/agent_retrieval/generator/fixed_pool.py` (add new function)
- Test: `tests/test_fixed_pool_l3.py` (create)

L3 sampling needs:
- For N ≥ 4: 1 item per quadrant `(live, lower)`, `(live, upper)`, `(dead, lower)`, `(dead, upper)`, then `N-4` random across all 4 quadrants.
- For N == 3: 1× (live, lower) + 1× (live, upper) + 1 random live item.
- For N == 2: 1× (live, lower) + 1× (live, upper).
- N < 2 not supported (raise).
- Process-deterministic: seeded by `hashlib.md5(parametrisation_id)`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_fixed_pool_l3.py`:

```python
from agent_retrieval.generator.fixed_pool import sample_fixed_pool_l3


def _make_pool():
    """16-item pool: 4 each of (live, lower), (live, upper), (dead, lower), (dead, upper)."""
    pool = []
    for live in (True, False):
        for direction in ("lower", "upper"):
            for i in range(4):
                pool.append({
                    "text": f"{'L' if live else 'D'}_{direction}_{i}",
                    "bound_direction": direction,
                    "live": live,
                    "bound_value": str(i * 10 + (0 if direction == "lower" else 1000)),
                })
    return pool


class TestSampleFixedPoolL3:
    def test_n2_returns_one_live_lower_and_one_live_upper(self):
        pool = _make_pool()
        sample = sample_fixed_pool_l3(pool, n=2, parametrisation_id="pid_a")
        assert len(sample) == 2
        live = [it for it in sample if it["live"]]
        assert len(live) == 2
        directions = sorted(it["bound_direction"] for it in live)
        assert directions == ["lower", "upper"]

    def test_n3_returns_balanced_live_pair_plus_one_more_live(self):
        pool = _make_pool()
        sample = sample_fixed_pool_l3(pool, n=3, parametrisation_id="pid_b")
        assert len(sample) == 3
        live = [it for it in sample if it["live"]]
        assert len(live) == 3
        directions = [it["bound_direction"] for it in live]
        assert "lower" in directions and "upper" in directions

    def test_n4_covers_all_four_quadrants(self):
        pool = _make_pool()
        sample = sample_fixed_pool_l3(pool, n=4, parametrisation_id="pid_c")
        assert len(sample) == 4
        quadrants = {(it["live"], it["bound_direction"]) for it in sample}
        assert quadrants == {(True, "lower"), (True, "upper"), (False, "lower"), (False, "upper")}

    def test_n8_has_balanced_first4_and_4_random(self):
        pool = _make_pool()
        sample = sample_fixed_pool_l3(pool, n=8, parametrisation_id="pid_d")
        assert len(sample) == 8
        # All 4 quadrants must appear at least once across the 8 items.
        quadrants = {(it["live"], it["bound_direction"]) for it in sample}
        assert quadrants == {(True, "lower"), (True, "upper"), (False, "lower"), (False, "upper")}

    def test_deterministic_across_calls(self):
        pool = _make_pool()
        a = sample_fixed_pool_l3(pool, n=4, parametrisation_id="same_pid")
        b = sample_fixed_pool_l3(pool, n=4, parametrisation_id="same_pid")
        assert [it["text"] for it in a] == [it["text"] for it in b]

    def test_different_pids_produce_different_samples(self):
        pool = _make_pool()
        a = sample_fixed_pool_l3(pool, n=8, parametrisation_id="pid_x")
        b = sample_fixed_pool_l3(pool, n=8, parametrisation_id="pid_y")
        # Highly unlikely to be identical; if they are something is wrong.
        assert [it["text"] for it in a] != [it["text"] for it in b]

    def test_n_too_large_raises(self):
        pool = _make_pool()
        import pytest
        with pytest.raises(ValueError):
            sample_fixed_pool_l3(pool, n=20, parametrisation_id="pid")

    def test_n_below_two_raises(self):
        pool = _make_pool()
        import pytest
        with pytest.raises(ValueError):
            sample_fixed_pool_l3(pool, n=1, parametrisation_id="pid")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `poetry run pytest tests/test_fixed_pool_l3.py -v`
Expected: FAIL with `ImportError: cannot import name 'sample_fixed_pool_l3'`.

- [ ] **Step 3: Implement `sample_fixed_pool_l3`**

Append to `src/agent_retrieval/generator/fixed_pool.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `poetry run pytest tests/test_fixed_pool_l3.py -v`
Expected: all 8 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/agent_retrieval/generator/fixed_pool.py tests/test_fixed_pool_l3.py
git commit -m "generator: quadrant-balanced sampler for pure_reasoning_l3"
```

---

## Task 3: Author L2 `python_repo` pool + template

**Files:**
- Create: `experiments/pure_reasoning_l2.yaml`

This is content-authoring. For each of the 16 L1 items, pick variant_id by `i % 3` (0=canonical, 1=alt-unit, 2=offset-from-reference) and rewrite the surface form. Each item carries: `text` (the rendered surface form, which becomes `inserted_text` in the AK), `bound_direction`, `bound_value` (canonical, in seconds), `context_summary`, plus two new fields: `variant_id`, `variant_text`.

For minimum churn, set `text == variant_text` (the rendered form is the only form ever shown to the agent).

Reference: L1 pool is in `experiments/pure_reasoning.yaml:46-112` (python_repo).

- [ ] **Step 1: Create the L2 template skeleton**

Create `experiments/pure_reasoning_l2.yaml`:

```yaml
experiment_type: "pure_reasoning_l2"

payload:
  item_type: "fact"

question_examples:
  python_repo:
    hard_contextual:
      question: |
        You are running a database migration. It must execute during a window of system quiescence. Below are {n} numeric configuration parameters from across the service. Each implies a constraint on *when* the migration can safely begin and end relative to a reference event (the most recent backup, the next traffic peak, the next certificate rotation, the latest replication lag spike, etc.).

        Some parameters establish a *lower bound* on the migration start time (must wait at least X seconds after the reference event).

        Others establish an *upper bound* on the migration end time (must complete before X seconds after the reference event).

        Some constraints are stated in mixed units or relative to a named reference. Normalise to a common scale (seconds) before reasoning.

        Below are the {n} facts:

        {facts_block}

        Derive the narrowest safe-migration window [earliest_start, latest_end] consistent with all {n} constraints. For each endpoint, cite the parameter that establishes it. Justify your classification of each parameter's bound direction.
      answer: "Window endpoints with citations"
  noir_fiction:
    hard_contextual:
      question: |
        You are given {n} pieces of evidence from a homicide investigation, each pertaining to events on the night of the murder. Some pieces establish that the suspect was at large (free, unaccounted-for) at or after a specific time; others establish that the suspect was off-the-streets (accounted-for, with an alibi) at or before a specific time.

        Some pieces of evidence are stated as clock times, others as offsets from a named anchor event, and some on a 24-hour clock or in colloquial form. Normalise to a single clock-time scale before reasoning.

        Below are the {n} pieces of evidence:

        {facts_block}

        Derive the narrowest defensible time window during which the suspect was at large on the night of the murder. Cite the specific pieces of evidence that establish your window's lower and upper bounds. Justify your classification of each piece of evidence's bound direction.
      answer: "Window endpoints with citations"

rubric_criteria:
  - criterion: "endpoint_correctness"
    weight: 1.0
  - criterion: "classification_accuracy"
    weight: 0.5

grid:
  content_profile: [python_repo, noir_fiction]
  n_items: [2, 4, 8, 12, 16]

fixed_pool:
  python_repo: []
  noir_fiction: []
```

- [ ] **Step 2: Author the 16 python_repo unit-variant items**

The variant assignment is deterministic by index (0=canonical, 1=alt-unit, 2=offset-from-reference); cycle through 0,1,2,0,1,2,... across all 16 items so each variant is hit ~5-6 times. Maintain the same `bound_value` and `bound_direction` as the L1 pool (8 lower + 8 upper, values in seconds).

Edit `experiments/pure_reasoning_l2.yaml` and replace `python_repo: []` under `fixed_pool` with the 16-item list. Use this template per item:

```yaml
    - text: <variant_text — full surface form rendered for the agent>
      bound_direction: <"lower" | "upper">
      bound_value: <canonical seconds, as string>
      context_summary: <brief summary, may differ slightly from L1 to reflect variant>
      variant_id: <0 | 1 | 2>
      variant_text: <same as text — kept for downstream tooling clarity>
```

Concrete 16 items (use these exactly):

```yaml
fixed_pool:
  python_repo:
    # ---- 8 lower-bound items ----
    - text: "REPLICATION_LAG_RECOVERY_S = 300"
      bound_direction: "lower"
      bound_value: "300"
      context_summary: "Time the read replicas need to catch up after the latest traffic-peak lag spike."
      variant_id: 0
      variant_text: "REPLICATION_LAG_RECOVERY_S = 300"
    - text: "CERT_ROTATION_GRACE_MIN = 30  # minutes"
      bound_direction: "lower"
      bound_value: "1800"
      context_summary: "Quiet period after the most recent certificate rotation; expressed in minutes."
      variant_id: 1
      variant_text: "CERT_ROTATION_GRACE_MIN = 30  # minutes"
    - text: "Reference: HOURLY_KICK_S = 3600. Then DDL_LOCK_COOLDOWN_OFFSET_S = 3000  # seconds before HOURLY_KICK"
      bound_direction: "lower"
      bound_value: "600"
      context_summary: "Cooldown after DDL lock release, expressed as offset before the hourly kick (3600 - 3000 = 600s)."
      variant_id: 2
      variant_text: "Reference: HOURLY_KICK_S = 3600. Then DDL_LOCK_COOLDOWN_OFFSET_S = 3000  # seconds before HOURLY_KICK"
    - text: "LEADER_ELECTION_QUIESCE_S = 900"
      bound_direction: "lower"
      bound_value: "900"
      context_summary: "Stabilization window after the most recent leader election."
      variant_id: 0
      variant_text: "LEADER_ELECTION_QUIESCE_S = 900"
    - text: "AUTOVACUUM_TAIL_DRAIN_MS = 1200000  # milliseconds"
      bound_direction: "lower"
      bound_value: "1200"
      context_summary: "Auto-vacuum tail drain delay, expressed in milliseconds."
      variant_id: 1
      variant_text: "AUTOVACUUM_TAIL_DRAIN_MS = 1200000  # milliseconds"
    - text: "Reference: BACKUP_FINISH_S = 600. Then INFLIGHT_RPC_DRAIN_OFFSET_S = 420  # seconds before BACKUP_FINISH"
      bound_direction: "lower"
      bound_value: "180"
      context_summary: "RPC drain wait, expressed as offset before backup finish (600 - 420 = 180s)."
      variant_id: 2
      variant_text: "Reference: BACKUP_FINISH_S = 600. Then INFLIGHT_RPC_DRAIN_OFFSET_S = 420  # seconds before BACKUP_FINISH"
    - text: "TRACE_FLUSH_BUFFER_S = 120"
      bound_direction: "lower"
      bound_value: "120"
      context_summary: "Tracing collector buffer flush delay."
      variant_id: 0
      variant_text: "TRACE_FLUSH_BUFFER_S = 120"
    - text: "ON_CALL_HANDOFF_QUIET_MIN = 25  # minutes"
      bound_direction: "lower"
      bound_value: "1500"
      context_summary: "Quiet interval after on-call handoff, expressed in minutes."
      variant_id: 1
      variant_text: "ON_CALL_HANDOFF_QUIET_MIN = 25  # minutes"
    # ---- 8 upper-bound items ----
    - text: "Reference: WEEKEND_ROLLOVER_S = 86400. Then BACKUP_WINDOW_OPEN_OFFSET_S = 79200  # seconds before WEEKEND_ROLLOVER"
      bound_direction: "upper"
      bound_value: "7200"
      context_summary: "Backup window open, expressed as offset before weekend rollover (86400 - 79200 = 7200s)."
      variant_id: 2
      variant_text: "Reference: WEEKEND_ROLLOVER_S = 86400. Then BACKUP_WINDOW_OPEN_OFFSET_S = 79200  # seconds before WEEKEND_ROLLOVER"
    - text: "PEAK_TRAFFIC_RAMP_BEGIN_S = 5400"
      bound_direction: "upper"
      bound_value: "5400"
      context_summary: "Daily traffic ramp begins."
      variant_id: 0
      variant_text: "PEAK_TRAFFIC_RAMP_BEGIN_S = 5400"
    - text: "AUTOSCALER_SCALEDOWN_DEADLINE_MIN = 180  # minutes"
      bound_direction: "upper"
      bound_value: "10800"
      context_summary: "Autoscaler scaledown deadline, expressed in minutes."
      variant_id: 1
      variant_text: "AUTOSCALER_SCALEDOWN_DEADLINE_MIN = 180  # minutes"
    - text: "Reference: NIGHTLY_LOG_TRIM_S = 28800. Then WEEKLY_SNAPSHOT_KICKOFF_OFFSET_S = 3600  # seconds before NIGHTLY_LOG_TRIM (negative = after)"
      bound_direction: "upper"
      bound_value: "25200"
      context_summary: "Weekly snapshot kickoff, expressed as offset before nightly log trim (28800 - 3600 = 25200s)."
      variant_id: 2
      variant_text: "Reference: NIGHTLY_LOG_TRIM_S = 28800. Then WEEKLY_SNAPSHOT_KICKOFF_OFFSET_S = 3600  # seconds before NIGHTLY_LOG_TRIM (negative = after)"
    - text: "DEPLOY_FREEZE_BOUNDARY_S = 14400"
      bound_direction: "upper"
      bound_value: "14400"
      context_summary: "Change-management deploy-freeze takes effect."
      variant_id: 0
      variant_text: "DEPLOY_FREEZE_BOUNDARY_S = 14400"
    - text: "ALERTING_SILENCE_EXPIRES_MIN = 75  # minutes"
      bound_direction: "upper"
      bound_value: "4500"
      context_summary: "Alerting silence window expiry, expressed in minutes."
      variant_id: 1
      variant_text: "ALERTING_SILENCE_EXPIRES_MIN = 75  # minutes"
    - text: "Reference: SHIFT_END_S = 21600. Then PLANNED_MAINTENANCE_CLOSE_OFFSET_S = 3600  # seconds before SHIFT_END (negative = after)"
      bound_direction: "upper"
      bound_value: "18000"
      context_summary: "Planned-maintenance window close, expressed as offset before shift end (21600 - 3600 = 18000s)."
      variant_id: 2
      variant_text: "Reference: SHIFT_END_S = 21600. Then PLANNED_MAINTENANCE_CLOSE_OFFSET_S = 3600  # seconds before SHIFT_END (negative = after)"
    - text: "METRICS_ROLLUP_TRIGGER_S = 9000"
      bound_direction: "upper"
      bound_value: "9000"
      context_summary: "Hourly metrics rollup begins."
      variant_id: 0
      variant_text: "METRICS_ROLLUP_TRIGGER_S = 9000"
```

- [ ] **Step 3: Verify the template loads via Pydantic**

Run:

```bash
poetry run python -c "
from pathlib import Path
from agent_retrieval.schema.template import ExperimentTemplate
t = ExperimentTemplate.from_yaml(Path('experiments/pure_reasoning_l2.yaml'))
items = t.fixed_pool['python_repo']
assert len(items) == 16
lowers = [it for it in items if it['bound_direction'] == 'lower']
uppers = [it for it in items if it['bound_direction'] == 'upper']
assert len(lowers) == 8 and len(uppers) == 8
max_lower = max(int(it['bound_value']) for it in lowers)
min_upper = min(int(it['bound_value']) for it in uppers)
assert max_lower < min_upper, f'pool inconsistent: max_lower={max_lower}, min_upper={min_upper}'
variants = sorted(it['variant_id'] for it in items)
print(f'OK: 16 items, 8L/8U, max_lower={max_lower} < min_upper={min_upper}, variants={variants}')
"
```

Expected output:
```
OK: 16 items, 8L/8U, max_lower=1800 < min_upper=4500, variants=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
```

- [ ] **Step 4: Commit**

```bash
git add experiments/pure_reasoning_l2.yaml
git commit -m "experiments: pure_reasoning_l2 template + python_repo unit-variant pool"
```

---

## Task 4: Author L2 `noir_fiction` pool

**Files:**
- Modify: `experiments/pure_reasoning_l2.yaml` (replace `noir_fiction: []` with 16 items)

Same approach: 16 items, 8 lower (suspect at large at/after T, spread 6:00 PM – 8:50 PM), 8 upper (suspect off the streets at/before T, spread 9:05 PM – 11:55 PM). Variant assignment cycles 0,1,2 across the 16 items.

- [ ] **Step 1: Author the 16 noir_fiction items**

Edit `experiments/pure_reasoning_l2.yaml` — replace `noir_fiction: []` with:

```yaml
  noir_fiction:
    # ---- 8 lower-bound items (suspect at large at/after time T) ----
    - text: "Old Mrs. Halloran from across the hall swore on her late husband's grave that Costa came up the stoop and slammed his door at six on the dot — she had just put the kettle on and the radio was running the news."
      bound_direction: "lower"
      bound_value: "6:00 PM"
      context_summary: "Neighbor narrator places Costa at the apartment building at 6:00 PM."
      variant_id: 0
      variant_text: "Old Mrs. Halloran from across the hall swore on her late husband's grave that Costa came up the stoop and slammed his door at six on the dot — she had just put the kettle on and the radio was running the news."
    - text: "Eddie at the newsstand on Forty-Third remembered selling Costa an evening Standard at 18:15 — 'He paid in nickels, like always,' Eddie said, 'and walked off east toward the river.'"
      bound_direction: "lower"
      bound_value: "6:15 PM"
      context_summary: "Newsstand vendor's account at 18:15 (24-hour clock variant)."
      variant_id: 1
      variant_text: "Eddie at the newsstand on Forty-Third remembered selling Costa an evening Standard at 18:15 — 'He paid in nickels, like always,' Eddie said, 'and walked off east toward the river.'"
    - text: "Anchor: the rosary bell at St. Catherine's tolled at 6:00 PM sharp that evening. The Greek who runs the diner on Lex told me Costa took the corner booth fifty minutes after the rosary bell and ordered nothing — just sat. He left after maybe twenty minutes."
      bound_direction: "lower"
      bound_value: "6:50 PM"
      context_summary: "Diner owner places Costa at 50 minutes after the 6:00 PM rosary bell anchor."
      variant_id: 2
      variant_text: "Anchor: the rosary bell at St. Catherine's tolled at 6:00 PM sharp that evening. The Greek who runs the diner on Lex told me Costa took the corner booth fifty minutes after the rosary bell and ordered nothing — just sat. He left after maybe twenty minutes."
    - text: "A streetcar conductor punched Costa's transfer at seven-oh-five, the night the Adler woman died. He remembered the face — said the man had a bruise on his cheek that hadn't been there the day before."
      bound_direction: "lower"
      bound_value: "7:05 PM"
      context_summary: "Streetcar conductor punches transfer at 7:05 PM."
      variant_id: 0
      variant_text: "A streetcar conductor punched Costa's transfer at seven-oh-five, the night the Adler woman died. He remembered the face — said the man had a bruise on his cheek that hadn't been there the day before."
    - text: "Mickey was last seen alive — and very much alone — by the doorman at the Astor at 19:30, the night of the killing."
      bound_direction: "lower"
      bound_value: "7:30 PM"
      context_summary: "Astor doorman's logbook shows the suspect at 19:30 (24-hour variant)."
      variant_id: 1
      variant_text: "Mickey was last seen alive — and very much alone — by the doorman at the Astor at 19:30, the night of the killing."
    - text: "Anchor: the express to Chicago departed Penn at 7:00 PM that evening. I caught up with the cigarette girl at the Indigo. She remembered him fine. 'Forty-five minutes after the Chicago express pulled out,' she said, 'he bought a pack of Luckies and didn't tip.'"
      bound_direction: "lower"
      bound_value: "7:45 PM"
      context_summary: "Cigarette girl pegs Costa at 45 minutes after the 7:00 PM Chicago express anchor."
      variant_id: 2
      variant_text: "Anchor: the express to Chicago departed Penn at 7:00 PM that evening. I caught up with the cigarette girl at the Indigo. She remembered him fine. 'Forty-five minutes after the Chicago express pulled out,' she said, 'he bought a pack of Luckies and didn't tip.'"
    - text: "The bartender at the Indigo served Costa a third gin at twenty before nine; Costa lingered another ten minutes, the bartender said, then left at half past."
      bound_direction: "lower"
      bound_value: "8:50 PM"
      context_summary: "Bartender's account stretches Costa's documented at-large time to 8:50 PM (colloquial variant)."
      variant_id: 0
      variant_text: "The bartender at the Indigo served Costa a third gin at twenty before nine; Costa lingered another ten minutes, the bartender said, then left at half past."
    - text: "Two patrolmen on the Grand-Central beat said they tipped their caps to Costa as he came up out of the IRT — they fixed it at 20:30, give or take, because they had just turned the corner past the cigar shop."
      bound_direction: "lower"
      bound_value: "8:30 PM"
      context_summary: "Beat cops fix Costa emerging from the subway at 20:30 (24-hour variant)."
      variant_id: 1
      variant_text: "Two patrolmen on the Grand-Central beat said they tipped their caps to Costa as he came up out of the IRT — they fixed it at 20:30, give or take, because they had just turned the corner past the cigar shop."
    # ---- 8 upper-bound items (suspect off the streets at/before time T) ----
    - text: "Anchor: the late edition went to press at 10:00 PM that evening. The desk sergeant logged Costa's pickup at the precinct fifty-five minutes before the late edition went to press — the suspect, demonstrably, was no longer at large."
      bound_direction: "upper"
      bound_value: "9:05 PM"
      context_summary: "Precinct intake at 55 minutes before the 10:00 PM late edition anchor."
      variant_id: 2
      variant_text: "Anchor: the late edition went to press at 10:00 PM that evening. The desk sergeant logged Costa's pickup at the precinct fifty-five minutes before the late edition went to press — the suspect, demonstrably, was no longer at large."
    - text: "The jail clerk pushed the booking slip across his desk and tapped the time stamp: nine-twenty, sharp. 'He's been ours since,' the clerk said, 'and he ain't going anywhere.'"
      bound_direction: "upper"
      bound_value: "9:20 PM"
      context_summary: "Jail clerk's booking slip at 9:20 PM."
      variant_id: 0
      variant_text: "The jail clerk pushed the booking slip across his desk and tapped the time stamp: nine-twenty, sharp. 'He's been ours since,' the clerk said, 'and he ain't going anywhere.'"
    - text: "Costa's lawyer arrived at the holding cell at 21:45; Costa had been there for the better part of an hour by then."
      bound_direction: "upper"
      bound_value: "9:45 PM"
      context_summary: "Lawyer's arrival at 21:45 (24-hour variant)."
      variant_id: 1
      variant_text: "Costa's lawyer arrived at the holding cell at 21:45; Costa had been there for the better part of an hour by then."
    - text: "Anchor: the second feature at the Roxy began at 9:30 PM that evening. The night nurse at Bellevue had Costa sedated and under restraint from thirty minutes after the second feature at the Roxy began — there was no question of him leaving that ward."
      bound_direction: "upper"
      bound_value: "10:00 PM"
      context_summary: "Night nurse's record at 30 min after the 9:30 PM Roxy anchor."
      variant_id: 2
      variant_text: "Anchor: the second feature at the Roxy began at 9:30 PM that evening. The night nurse at Bellevue had Costa sedated and under restraint from thirty minutes after the second feature at the Roxy began — there was no question of him leaving that ward."
    - text: "Mrs. Pavone, his landlady, watched him climb the stairs to his room at twenty past ten and bolted the front door behind him; she said the lock had not been turned again before morning."
      bound_direction: "upper"
      bound_value: "10:20 PM"
      context_summary: "Landlady's bolt-the-door account at 10:20 PM (canonical clock-time variant)."
      variant_id: 0
      variant_text: "Mrs. Pavone, his landlady, watched him climb the stairs to his room at twenty past ten and bolted the front door behind him; she said the lock had not been turned again before morning."
    - text: "The hotel switchboard operator had Costa on a long-distance call to Newark from 23:00 clear through midnight; she stayed on the line, as she did with every long-distance, and could swear to it."
      bound_direction: "upper"
      bound_value: "11:00 PM"
      context_summary: "Switchboard operator's continuous call from 23:00 (24-hour variant)."
      variant_id: 1
      variant_text: "The hotel switchboard operator had Costa on a long-distance call to Newark from 23:00 clear through midnight; she stayed on the line, as she did with every long-distance, and could swear to it."
    - text: "Anchor: the last ferry to Hoboken sailed at 11:00 PM that evening. The night porter on the eastbound Twentieth Century swung Costa's bag aboard forty-five minutes after the last Hoboken ferry sailed; he saw the man tucked into the lower berth and drew the curtain himself."
      bound_direction: "upper"
      bound_value: "11:45 PM"
      context_summary: "Train porter's berth-record at 45 min after the 11:00 PM ferry anchor."
      variant_id: 2
      variant_text: "Anchor: the last ferry to Hoboken sailed at 11:00 PM that evening. The night porter on the eastbound Twentieth Century swung Costa's bag aboard forty-five minutes after the last Hoboken ferry sailed; he saw the man tucked into the lower berth and drew the curtain himself."
    - text: "By five of midnight, the theatre usher was certain — he'd walked Costa, dead drunk, into the manager's office to sleep it off, and the manager had locked the door from the outside."
      bound_direction: "upper"
      bound_value: "11:55 PM"
      context_summary: "Theatre usher places Costa locked in manager's office by 11:55 PM (canonical variant)."
      variant_id: 0
      variant_text: "By five of midnight, the theatre usher was certain — he'd walked Costa, dead drunk, into the manager's office to sleep it off, and the manager had locked the door from the outside."
```

- [ ] **Step 2: Verify the noir pool loads and is consistent**

Run:

```bash
poetry run python -c "
from pathlib import Path
from agent_retrieval.schema.template import ExperimentTemplate
from agent_retrieval.generator.pure_reasoning_gen import _to_comparable
t = ExperimentTemplate.from_yaml(Path('experiments/pure_reasoning_l2.yaml'))
items = t.fixed_pool['noir_fiction']
assert len(items) == 16
lowers = [it for it in items if it['bound_direction'] == 'lower']
uppers = [it for it in items if it['bound_direction'] == 'upper']
assert len(lowers) == 8 and len(uppers) == 8
max_lower = max(_to_comparable(it['bound_value']) for it in lowers)
min_upper = min(_to_comparable(it['bound_value']) for it in uppers)
assert max_lower < min_upper, f'inconsistent: max_lower={max_lower}, min_upper={min_upper}'
variants = sorted(it['variant_id'] for it in items)
print(f'OK: 16 items, 8L/8U, max_lower={max_lower} < min_upper={min_upper}, variants={variants}')
"
```

Expected: prints "OK: 16 items, 8L/8U, max_lower=530.0 < min_upper=545.0, variants=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]" or similar (variant counts differ slightly).

- [ ] **Step 3: Commit**

```bash
git add experiments/pure_reasoning_l2.yaml
git commit -m "experiments: pure_reasoning_l2 noir_fiction unit-variant pool"
```

---

## Task 5: Author L3 `python_repo` pool + template

**Files:**
- Create: `experiments/pure_reasoning_l3.yaml`

Each item inherits its `text`, `bound_direction`, `bound_value`, `variant_id`, `variant_text`, and `context_summary` from the L2 python_repo pool. New fields:
- `gate_clause` — string prepended to `variant_text` to form the rendered prompt text
- `gate_world_var` — name of the world-state variable referenced
- `live` — bool

4 distinct gate variables per profile, each owning 4 items balanced 2 lower + 2 upper. 2 vars TRUE → 8 live items; 2 vars FALSE → 8 dead items.

Python world variables (4 total):
- `current_phase = "rolling"` (TRUE — 4 items use `if current_phase == "rolling"`)
- `replica_promoted = True` (TRUE — 4 items use `if replica_promoted`)
- `dual_write_enabled = False` (FALSE — 4 items use `if dual_write_enabled`, all dead)
- `read_traffic_paused = False` (FALSE — 4 items use `if read_traffic_paused`, all dead)

The L3 item's `text` field is the FULL rendered string (gate_clause + variant_text), so it can be inlined directly into the question.

- [ ] **Step 1: Create L3 template skeleton**

Create `experiments/pure_reasoning_l3.yaml`:

```yaml
experiment_type: "pure_reasoning_l3"

payload:
  item_type: "fact"

question_examples:
  python_repo:
    hard_contextual:
      question: |
        You are running a database migration. It must execute during a window of system quiescence. Below are {n} numeric configuration parameters from across the service. Each implies a constraint on *when* the migration can safely begin and end relative to a reference event.

        Some parameters establish a *lower bound* on the migration start time (must wait at least X seconds after the reference event); others establish an *upper bound* on the migration end time (must complete before X seconds after the reference event).

        Some constraints are stated in mixed units or relative to a named reference. Normalise to a common scale (seconds) before reasoning.

        The constraints are conditional on the current system state. Some apply only if the corresponding precondition holds. Use only constraints whose preconditions are satisfied. Ignore the rest.

        World state:
        {world_state_block}

        Below are the {n} facts:

        {facts_block}

        Derive the narrowest safe-migration window [earliest_start, latest_end] consistent with all *applicable* constraints. For each endpoint, cite the parameter that establishes it. Justify your classification of each parameter's bound direction, and call out which parameters were ignored because their precondition was not satisfied.
      answer: "Window endpoints with citations"
  noir_fiction:
    hard_contextual:
      question: |
        You are given {n} pieces of evidence from a homicide investigation, each pertaining to events on the night of the murder. Some pieces establish that the suspect was at large at or after a specific time; others establish that the suspect was off-the-streets at or before a specific time.

        Some pieces are stated as clock times, others as offsets from a named anchor event, and some on a 24-hour clock or in colloquial form. Normalise to a single clock-time scale before reasoning.

        Each piece of evidence is conditional on a separately-attested fact about that night. Use only pieces whose precondition is corroborated; ignore those that are not.

        Corroborated facts:
        {world_state_block}

        Below are the {n} pieces of evidence:

        {facts_block}

        Derive the narrowest defensible time window during which the suspect was at large on the night of the murder. Cite the specific pieces of evidence that establish your window's lower and upper bounds. Justify your classification of each piece of evidence's bound direction, and call out which pieces were ignored because their precondition was not corroborated.
      answer: "Window endpoints with citations"

rubric_criteria:
  - criterion: "endpoint_correctness"
    weight: 1.0
  - criterion: "classification_accuracy"
    weight: 0.5

grid:
  content_profile: [python_repo, noir_fiction]
  n_items: [2, 4, 8, 12, 16]

world_state:
  python_repo:
    current_phase: "rolling"
    replica_promoted: true
    dual_write_enabled: false
    read_traffic_paused: false
  noir_fiction:
    vincent_returned_from_brooklyn: true
    dispatcher_logbook_reliable: true
    lawyer_was_truthful: false
    conductor_punch_records_accurate: false

fixed_pool:
  python_repo: []
  noir_fiction: []
```

(Note: `world_state` is a top-level field; the schema's `model_config = ConfigDict(extra="ignore")` lets it pass through Pydantic without becoming a typed field. The generator reads it directly from the raw YAML in Task 7.)

- [ ] **Step 2: Author the 16 python_repo gated items**

Edit `experiments/pure_reasoning_l3.yaml`, replace `python_repo: []` under `fixed_pool` with this list. Each item's `text` is `<gate_clause> <variant_text>`. Quadrant assignment: 8 (live) split 4 lower + 4 upper, 8 (dead) split 4 lower + 4 upper.

```yaml
fixed_pool:
  python_repo:
    # ---- (live, lower) — gate var TRUE, 4 items ----
    - text: "If current_phase == 'rolling', then REPLICATION_LAG_RECOVERY_S = 300"
      bound_direction: "lower"
      bound_value: "300"
      context_summary: "Replication lag recovery; gated on rolling phase."
      variant_id: 0
      variant_text: "REPLICATION_LAG_RECOVERY_S = 300"
      gate_clause: "If current_phase == 'rolling', then"
      gate_world_var: "current_phase"
      live: true
    - text: "If current_phase == 'rolling', then CERT_ROTATION_GRACE_MIN = 30  # minutes"
      bound_direction: "lower"
      bound_value: "1800"
      context_summary: "Cert rotation grace; minutes; gated on rolling phase."
      variant_id: 1
      variant_text: "CERT_ROTATION_GRACE_MIN = 30  # minutes"
      gate_clause: "If current_phase == 'rolling', then"
      gate_world_var: "current_phase"
      live: true
    - text: "If replica_promoted, then Reference: HOURLY_KICK_S = 3600. Then DDL_LOCK_COOLDOWN_OFFSET_S = 3000  # seconds before HOURLY_KICK"
      bound_direction: "lower"
      bound_value: "600"
      context_summary: "DDL lock cooldown; offset variant; gated on replica promotion."
      variant_id: 2
      variant_text: "Reference: HOURLY_KICK_S = 3600. Then DDL_LOCK_COOLDOWN_OFFSET_S = 3000  # seconds before HOURLY_KICK"
      gate_clause: "If replica_promoted, then"
      gate_world_var: "replica_promoted"
      live: true
    - text: "If replica_promoted, then LEADER_ELECTION_QUIESCE_S = 900"
      bound_direction: "lower"
      bound_value: "900"
      context_summary: "Leader election quiesce; gated on replica promotion."
      variant_id: 0
      variant_text: "LEADER_ELECTION_QUIESCE_S = 900"
      gate_clause: "If replica_promoted, then"
      gate_world_var: "replica_promoted"
      live: true
    # ---- (live, upper) — gate var TRUE, 4 items ----
    - text: "If current_phase == 'rolling', then Reference: WEEKEND_ROLLOVER_S = 86400. Then BACKUP_WINDOW_OPEN_OFFSET_S = 79200  # seconds before WEEKEND_ROLLOVER"
      bound_direction: "upper"
      bound_value: "7200"
      context_summary: "Backup window open; offset variant; gated on rolling phase."
      variant_id: 2
      variant_text: "Reference: WEEKEND_ROLLOVER_S = 86400. Then BACKUP_WINDOW_OPEN_OFFSET_S = 79200  # seconds before WEEKEND_ROLLOVER"
      gate_clause: "If current_phase == 'rolling', then"
      gate_world_var: "current_phase"
      live: true
    - text: "If current_phase == 'rolling', then PEAK_TRAFFIC_RAMP_BEGIN_S = 5400"
      bound_direction: "upper"
      bound_value: "5400"
      context_summary: "Peak traffic ramp begin; gated on rolling phase."
      variant_id: 0
      variant_text: "PEAK_TRAFFIC_RAMP_BEGIN_S = 5400"
      gate_clause: "If current_phase == 'rolling', then"
      gate_world_var: "current_phase"
      live: true
    - text: "If replica_promoted, then AUTOSCALER_SCALEDOWN_DEADLINE_MIN = 180  # minutes"
      bound_direction: "upper"
      bound_value: "10800"
      context_summary: "Autoscaler scaledown deadline; minutes; gated on replica promotion."
      variant_id: 1
      variant_text: "AUTOSCALER_SCALEDOWN_DEADLINE_MIN = 180  # minutes"
      gate_clause: "If replica_promoted, then"
      gate_world_var: "replica_promoted"
      live: true
    - text: "If replica_promoted, then Reference: NIGHTLY_LOG_TRIM_S = 28800. Then WEEKLY_SNAPSHOT_KICKOFF_OFFSET_S = 3600  # seconds before NIGHTLY_LOG_TRIM (negative = after)"
      bound_direction: "upper"
      bound_value: "25200"
      context_summary: "Weekly snapshot kickoff; offset variant; gated on replica promotion."
      variant_id: 2
      variant_text: "Reference: NIGHTLY_LOG_TRIM_S = 28800. Then WEEKLY_SNAPSHOT_KICKOFF_OFFSET_S = 3600  # seconds before NIGHTLY_LOG_TRIM (negative = after)"
      gate_clause: "If replica_promoted, then"
      gate_world_var: "replica_promoted"
      live: true
    # ---- (dead, lower) — gate var FALSE, 4 items ----
    - text: "If dual_write_enabled, then AUTOVACUUM_TAIL_DRAIN_MS = 1200000  # milliseconds"
      bound_direction: "lower"
      bound_value: "1200"
      context_summary: "Autovacuum tail drain; ms variant; gated on dual-write (off)."
      variant_id: 1
      variant_text: "AUTOVACUUM_TAIL_DRAIN_MS = 1200000  # milliseconds"
      gate_clause: "If dual_write_enabled, then"
      gate_world_var: "dual_write_enabled"
      live: false
    - text: "If dual_write_enabled, then Reference: BACKUP_FINISH_S = 600. Then INFLIGHT_RPC_DRAIN_OFFSET_S = 420  # seconds before BACKUP_FINISH"
      bound_direction: "lower"
      bound_value: "180"
      context_summary: "RPC drain; offset variant; gated on dual-write (off)."
      variant_id: 2
      variant_text: "Reference: BACKUP_FINISH_S = 600. Then INFLIGHT_RPC_DRAIN_OFFSET_S = 420  # seconds before BACKUP_FINISH"
      gate_clause: "If dual_write_enabled, then"
      gate_world_var: "dual_write_enabled"
      live: false
    - text: "If read_traffic_paused, then TRACE_FLUSH_BUFFER_S = 120"
      bound_direction: "lower"
      bound_value: "120"
      context_summary: "Trace flush buffer; gated on read-traffic-paused (off)."
      variant_id: 0
      variant_text: "TRACE_FLUSH_BUFFER_S = 120"
      gate_clause: "If read_traffic_paused, then"
      gate_world_var: "read_traffic_paused"
      live: false
    - text: "If read_traffic_paused, then ON_CALL_HANDOFF_QUIET_MIN = 25  # minutes"
      bound_direction: "lower"
      bound_value: "1500"
      context_summary: "On-call handoff quiet; minutes; gated on read-traffic-paused (off)."
      variant_id: 1
      variant_text: "ON_CALL_HANDOFF_QUIET_MIN = 25  # minutes"
      gate_clause: "If read_traffic_paused, then"
      gate_world_var: "read_traffic_paused"
      live: false
    # ---- (dead, upper) — gate var FALSE, 4 items ----
    - text: "If dual_write_enabled, then DEPLOY_FREEZE_BOUNDARY_S = 14400"
      bound_direction: "upper"
      bound_value: "14400"
      context_summary: "Deploy freeze boundary; gated on dual-write (off)."
      variant_id: 0
      variant_text: "DEPLOY_FREEZE_BOUNDARY_S = 14400"
      gate_clause: "If dual_write_enabled, then"
      gate_world_var: "dual_write_enabled"
      live: false
    - text: "If dual_write_enabled, then ALERTING_SILENCE_EXPIRES_MIN = 75  # minutes"
      bound_direction: "upper"
      bound_value: "4500"
      context_summary: "Alerting silence expires; minutes; gated on dual-write (off)."
      variant_id: 1
      variant_text: "ALERTING_SILENCE_EXPIRES_MIN = 75  # minutes"
      gate_clause: "If dual_write_enabled, then"
      gate_world_var: "dual_write_enabled"
      live: false
    - text: "If read_traffic_paused, then Reference: SHIFT_END_S = 21600. Then PLANNED_MAINTENANCE_CLOSE_OFFSET_S = 3600  # seconds before SHIFT_END (negative = after)"
      bound_direction: "upper"
      bound_value: "18000"
      context_summary: "Planned maintenance close; offset variant; gated on read-traffic-paused (off)."
      variant_id: 2
      variant_text: "Reference: SHIFT_END_S = 21600. Then PLANNED_MAINTENANCE_CLOSE_OFFSET_S = 3600  # seconds before SHIFT_END (negative = after)"
      gate_clause: "If read_traffic_paused, then"
      gate_world_var: "read_traffic_paused"
      live: false
    - text: "If read_traffic_paused, then METRICS_ROLLUP_TRIGGER_S = 9000"
      bound_direction: "upper"
      bound_value: "9000"
      context_summary: "Metrics rollup trigger; gated on read-traffic-paused (off)."
      variant_id: 0
      variant_text: "METRICS_ROLLUP_TRIGGER_S = 9000"
      gate_clause: "If read_traffic_paused, then"
      gate_world_var: "read_traffic_paused"
      live: false
```

- [ ] **Step 3: Verify the python_repo L3 pool loads and is consistent**

Run:

```bash
poetry run python -c "
from pathlib import Path
import yaml
data = yaml.safe_load(Path('experiments/pure_reasoning_l3.yaml').read_text())
items = data['fixed_pool']['python_repo']
assert len(items) == 16
quads = {}
for it in items:
    key = (it['live'], it['bound_direction'])
    quads[key] = quads.get(key, 0) + 1
assert quads == {(True, 'lower'): 4, (True, 'upper'): 4, (False, 'lower'): 4, (False, 'upper'): 4}, quads
ws = data['world_state']['python_repo']
for it in items:
    var = it['gate_world_var']
    assert var in ws, f'gate var {var} not in world_state'
    assert bool(ws[var]) == it['live'], f'live mismatch for {it[\"text\"][:60]}: ws[{var}]={ws[var]}, live={it[\"live\"]}'
print('OK: 16 items, 4 quadrants of 4, gates consistent with world_state')
"
```

Expected: prints "OK: 16 items, 4 quadrants of 4, gates consistent with world_state".

- [ ] **Step 4: Commit**

```bash
git add experiments/pure_reasoning_l3.yaml
git commit -m "experiments: pure_reasoning_l3 template + python_repo gated pool"
```

---

## Task 6: Author L3 `noir_fiction` pool

**Files:**
- Modify: `experiments/pure_reasoning_l3.yaml` (replace `noir_fiction: []`)

Same shape as Task 5 for noir. 4 narrative gate variables; each owns 4 items balanced 2 lower + 2 upper. 2 vars TRUE → 8 live; 2 FALSE → 8 dead.

Noir gates:
- `vincent_returned_from_brooklyn = True` (TRUE — 4 items)
- `dispatcher_logbook_reliable = True` (TRUE — 4 items)
- `lawyer_was_truthful = False` (FALSE — 4 items, all dead)
- `conductor_punch_records_accurate = False` (FALSE — 4 items, all dead)

- [ ] **Step 1: Author the 16 noir_fiction gated items**

Edit `experiments/pure_reasoning_l3.yaml`, replace `noir_fiction: []` with:

```yaml
  noir_fiction:
    # ---- (live, lower) ----
    - text: "If Vincent did return from Brooklyn that night, then Old Mrs. Halloran from across the hall swore on her late husband's grave that Costa came up the stoop and slammed his door at six on the dot — she had just put the kettle on and the radio was running the news."
      bound_direction: "lower"
      bound_value: "6:00 PM"
      context_summary: "Mrs. Halloran's account at 6:00 PM; gated on Vincent's return."
      variant_id: 0
      variant_text: "Old Mrs. Halloran from across the hall swore on her late husband's grave that Costa came up the stoop and slammed his door at six on the dot — she had just put the kettle on and the radio was running the news."
      gate_clause: "If Vincent did return from Brooklyn that night, then"
      gate_world_var: "vincent_returned_from_brooklyn"
      live: true
    - text: "If Vincent did return from Brooklyn that night, then Eddie at the newsstand on Forty-Third remembered selling Costa an evening Standard at 18:15 — 'He paid in nickels, like always,' Eddie said, 'and walked off east toward the river.'"
      bound_direction: "lower"
      bound_value: "6:15 PM"
      context_summary: "Eddie's newsstand account at 18:15; gated on Vincent's return."
      variant_id: 1
      variant_text: "Eddie at the newsstand on Forty-Third remembered selling Costa an evening Standard at 18:15 — 'He paid in nickels, like always,' Eddie said, 'and walked off east toward the river.'"
      gate_clause: "If Vincent did return from Brooklyn that night, then"
      gate_world_var: "vincent_returned_from_brooklyn"
      live: true
    - text: "If the dispatcher's logbook is reliable, then Anchor: the rosary bell at St. Catherine's tolled at 6:00 PM sharp that evening. The Greek who runs the diner on Lex told me Costa took the corner booth fifty minutes after the rosary bell and ordered nothing — just sat. He left after maybe twenty minutes."
      bound_direction: "lower"
      bound_value: "6:50 PM"
      context_summary: "Diner owner's account at 50 min after rosary bell anchor; gated on dispatcher logbook."
      variant_id: 2
      variant_text: "Anchor: the rosary bell at St. Catherine's tolled at 6:00 PM sharp that evening. The Greek who runs the diner on Lex told me Costa took the corner booth fifty minutes after the rosary bell and ordered nothing — just sat. He left after maybe twenty minutes."
      gate_clause: "If the dispatcher's logbook is reliable, then"
      gate_world_var: "dispatcher_logbook_reliable"
      live: true
    - text: "If the dispatcher's logbook is reliable, then a streetcar conductor punched Costa's transfer at seven-oh-five, the night the Adler woman died. He remembered the face — said the man had a bruise on his cheek that hadn't been there the day before."
      bound_direction: "lower"
      bound_value: "7:05 PM"
      context_summary: "Streetcar conductor's punch at 7:05 PM; gated on dispatcher logbook."
      variant_id: 0
      variant_text: "A streetcar conductor punched Costa's transfer at seven-oh-five, the night the Adler woman died. He remembered the face — said the man had a bruise on his cheek that hadn't been there the day before."
      gate_clause: "If the dispatcher's logbook is reliable, then"
      gate_world_var: "dispatcher_logbook_reliable"
      live: true
    # ---- (live, upper) ----
    - text: "If Vincent did return from Brooklyn that night, then Anchor: the late edition went to press at 10:00 PM that evening. The desk sergeant logged Costa's pickup at the precinct fifty-five minutes before the late edition went to press — the suspect, demonstrably, was no longer at large."
      bound_direction: "upper"
      bound_value: "9:05 PM"
      context_summary: "Precinct intake at 9:05 PM via 10 PM late-edition anchor; gated on Vincent's return."
      variant_id: 2
      variant_text: "Anchor: the late edition went to press at 10:00 PM that evening. The desk sergeant logged Costa's pickup at the precinct fifty-five minutes before the late edition went to press — the suspect, demonstrably, was no longer at large."
      gate_clause: "If Vincent did return from Brooklyn that night, then"
      gate_world_var: "vincent_returned_from_brooklyn"
      live: true
    - text: "If Vincent did return from Brooklyn that night, then the jail clerk pushed the booking slip across his desk and tapped the time stamp: nine-twenty, sharp. 'He's been ours since,' the clerk said, 'and he ain't going anywhere.'"
      bound_direction: "upper"
      bound_value: "9:20 PM"
      context_summary: "Jail clerk's booking slip at 9:20 PM; gated on Vincent's return."
      variant_id: 0
      variant_text: "The jail clerk pushed the booking slip across his desk and tapped the time stamp: nine-twenty, sharp. 'He's been ours since,' the clerk said, 'and he ain't going anywhere.'"
      gate_clause: "If Vincent did return from Brooklyn that night, then"
      gate_world_var: "vincent_returned_from_brooklyn"
      live: true
    - text: "If the dispatcher's logbook is reliable, then Costa's lawyer arrived at the holding cell at 21:45; Costa had been there for the better part of an hour by then."
      bound_direction: "upper"
      bound_value: "9:45 PM"
      context_summary: "Lawyer's arrival at 21:45; gated on dispatcher logbook."
      variant_id: 1
      variant_text: "Costa's lawyer arrived at the holding cell at 21:45; Costa had been there for the better part of an hour by then."
      gate_clause: "If the dispatcher's logbook is reliable, then"
      gate_world_var: "dispatcher_logbook_reliable"
      live: true
    - text: "If the dispatcher's logbook is reliable, then Anchor: the second feature at the Roxy began at 9:30 PM that evening. The night nurse at Bellevue had Costa sedated and under restraint from thirty minutes after the second feature at the Roxy began — there was no question of him leaving that ward."
      bound_direction: "upper"
      bound_value: "10:00 PM"
      context_summary: "Nurse's record at 10 PM via Roxy 9:30 PM anchor; gated on dispatcher logbook."
      variant_id: 2
      variant_text: "Anchor: the second feature at the Roxy began at 9:30 PM that evening. The night nurse at Bellevue had Costa sedated and under restraint from thirty minutes after the second feature at the Roxy began — there was no question of him leaving that ward."
      gate_clause: "If the dispatcher's logbook is reliable, then"
      gate_world_var: "dispatcher_logbook_reliable"
      live: true
    # ---- (dead, lower) ----
    - text: "If the lawyer was telling the truth, then Mickey was last seen alive — and very much alone — by the doorman at the Astor at 19:30, the night of the killing."
      bound_direction: "lower"
      bound_value: "7:30 PM"
      context_summary: "Doorman's logbook at 19:30; gated on lawyer's truthfulness (off)."
      variant_id: 1
      variant_text: "Mickey was last seen alive — and very much alone — by the doorman at the Astor at 19:30, the night of the killing."
      gate_clause: "If the lawyer was telling the truth, then"
      gate_world_var: "lawyer_was_truthful"
      live: false
    - text: "If the lawyer was telling the truth, then Anchor: the express to Chicago departed Penn at 7:00 PM that evening. I caught up with the cigarette girl at the Indigo. She remembered him fine. 'Forty-five minutes after the Chicago express pulled out,' she said, 'he bought a pack of Luckies and didn't tip.'"
      bound_direction: "lower"
      bound_value: "7:45 PM"
      context_summary: "Cigarette girl at 45 min after 7 PM Chicago express anchor; gated on lawyer's truthfulness (off)."
      variant_id: 2
      variant_text: "Anchor: the express to Chicago departed Penn at 7:00 PM that evening. I caught up with the cigarette girl at the Indigo. She remembered him fine. 'Forty-five minutes after the Chicago express pulled out,' she said, 'he bought a pack of Luckies and didn't tip.'"
      gate_clause: "If the lawyer was telling the truth, then"
      gate_world_var: "lawyer_was_truthful"
      live: false
    - text: "If the conductor's punch records were accurate that night, then the bartender at the Indigo served Costa a third gin at twenty before nine; Costa lingered another ten minutes, the bartender said, then left at half past."
      bound_direction: "lower"
      bound_value: "8:50 PM"
      context_summary: "Bartender's account at 8:50 PM; gated on conductor records (off)."
      variant_id: 0
      variant_text: "The bartender at the Indigo served Costa a third gin at twenty before nine; Costa lingered another ten minutes, the bartender said, then left at half past."
      gate_clause: "If the conductor's punch records were accurate that night, then"
      gate_world_var: "conductor_punch_records_accurate"
      live: false
    - text: "If the conductor's punch records were accurate that night, then two patrolmen on the Grand-Central beat said they tipped their caps to Costa as he came up out of the IRT — they fixed it at 20:30, give or take, because they had just turned the corner past the cigar shop."
      bound_direction: "lower"
      bound_value: "8:30 PM"
      context_summary: "Patrolmen's account at 20:30; gated on conductor records (off)."
      variant_id: 1
      variant_text: "Two patrolmen on the Grand-Central beat said they tipped their caps to Costa as he came up out of the IRT — they fixed it at 20:30, give or take, because they had just turned the corner past the cigar shop."
      gate_clause: "If the conductor's punch records were accurate that night, then"
      gate_world_var: "conductor_punch_records_accurate"
      live: false
    # ---- (dead, upper) ----
    - text: "If the lawyer was telling the truth, then Mrs. Pavone, his landlady, watched him climb the stairs to his room at twenty past ten and bolted the front door behind him; she said the lock had not been turned again before morning."
      bound_direction: "upper"
      bound_value: "10:20 PM"
      context_summary: "Landlady's bolt-the-door at 10:20 PM; gated on lawyer's truthfulness (off)."
      variant_id: 0
      variant_text: "Mrs. Pavone, his landlady, watched him climb the stairs to his room at twenty past ten and bolted the front door behind him; she said the lock had not been turned again before morning."
      gate_clause: "If the lawyer was telling the truth, then"
      gate_world_var: "lawyer_was_truthful"
      live: false
    - text: "If the lawyer was telling the truth, then the hotel switchboard operator had Costa on a long-distance call to Newark from 23:00 clear through midnight; she stayed on the line, as she did with every long-distance, and could swear to it."
      bound_direction: "upper"
      bound_value: "11:00 PM"
      context_summary: "Switchboard operator's continuous call from 23:00; gated on lawyer's truthfulness (off)."
      variant_id: 1
      variant_text: "The hotel switchboard operator had Costa on a long-distance call to Newark from 23:00 clear through midnight; she stayed on the line, as she did with every long-distance, and could swear to it."
      gate_clause: "If the lawyer was telling the truth, then"
      gate_world_var: "lawyer_was_truthful"
      live: false
    - text: "If the conductor's punch records were accurate that night, then Anchor: the last ferry to Hoboken sailed at 11:00 PM that evening. The night porter on the eastbound Twentieth Century swung Costa's bag aboard forty-five minutes after the last Hoboken ferry sailed; he saw the man tucked into the lower berth and drew the curtain himself."
      bound_direction: "upper"
      bound_value: "11:45 PM"
      context_summary: "Train porter at 45 min after 11 PM ferry anchor; gated on conductor records (off)."
      variant_id: 2
      variant_text: "Anchor: the last ferry to Hoboken sailed at 11:00 PM that evening. The night porter on the eastbound Twentieth Century swung Costa's bag aboard forty-five minutes after the last Hoboken ferry sailed; he saw the man tucked into the lower berth and drew the curtain himself."
      gate_clause: "If the conductor's punch records were accurate that night, then"
      gate_world_var: "conductor_punch_records_accurate"
      live: false
    - text: "If the conductor's punch records were accurate that night, then by five of midnight, the theatre usher was certain — he'd walked Costa, dead drunk, into the manager's office to sleep it off, and the manager had locked the door from the outside."
      bound_direction: "upper"
      bound_value: "11:55 PM"
      context_summary: "Theatre usher at 11:55 PM; gated on conductor records (off)."
      variant_id: 0
      variant_text: "By five of midnight, the theatre usher was certain — he'd walked Costa, dead drunk, into the manager's office to sleep it off, and the manager had locked the door from the outside."
      gate_clause: "If the conductor's punch records were accurate that night, then"
      gate_world_var: "conductor_punch_records_accurate"
      live: false
```

- [ ] **Step 2: Verify the noir L3 pool loads and is consistent**

Run:

```bash
poetry run python -c "
from pathlib import Path
import yaml
data = yaml.safe_load(Path('experiments/pure_reasoning_l3.yaml').read_text())
items = data['fixed_pool']['noir_fiction']
assert len(items) == 16
quads = {}
for it in items:
    key = (it['live'], it['bound_direction'])
    quads[key] = quads.get(key, 0) + 1
assert quads == {(True, 'lower'): 4, (True, 'upper'): 4, (False, 'lower'): 4, (False, 'upper'): 4}, quads
ws = data['world_state']['noir_fiction']
for it in items:
    var = it['gate_world_var']
    assert var in ws, f'gate var {var} not in world_state'
    assert bool(ws[var]) == it['live'], f'live mismatch for var {var}'
print('OK: 16 noir items, 4 quadrants of 4, gates consistent with world_state')
"
```

Expected: prints success message.

- [ ] **Step 3: Commit**

```bash
git add experiments/pure_reasoning_l3.yaml
git commit -m "experiments: pure_reasoning_l3 noir_fiction gated pool"
```

---

## Task 7: Extend `generate_pure_reasoning_cell` to dispatch L2 / L3

**Files:**
- Modify: `src/agent_retrieval/generator/pure_reasoning_gen.py`
- Test: `tests/test_pure_reasoning_l2_l3.py` (create)

`generate_pure_reasoning_cell` currently:
1. Loads pool, samples N items via `sample_fixed_pool` with `balance_key="bound_direction"`
2. Renders `question` from `template.question_examples[profile]["hard_contextual"].question` with `{n}` and `{facts_block}`
3. Computes max_lower / min_upper for the AK's `correctness` text
4. Writes AK YAML

For L2: same as L1 (sampling, AK shape) — only the question template differs (already authored to include the unit-normalisation clause; no code change needed for question rendering, since it just substitutes `{n}` and `{facts_block}`). The AK items get extra `variant_id` / `variant_text` fields written to YAML.

For L3: needs (a) `sample_fixed_pool_l3` instead of `sample_fixed_pool`; (b) load `world_state[profile]` from raw YAML and render a `{world_state_block}` placeholder; (c) AK items get extra `gate_clause`, `gate_world_var`, `live` fields; (d) `expected_answers.correctness` must restrict the answer to live items only.

Note: `template.fixed_pool` carries the items but doesn't carry `world_state` (since `extra="ignore"`). The generator must read `world_state` directly from the YAML file. We pass it through via a new optional argument on `generate_pure_reasoning_cell` so callers can supply it.

- [ ] **Step 1: Write failing tests**

Create `tests/test_pure_reasoning_l2_l3.py`:

```python
from pathlib import Path

import pytest

from agent_retrieval.generator.pure_reasoning_gen import generate_pure_reasoning_cell
from agent_retrieval.schema.answer_key import AnswerKey
from agent_retrieval.schema.template import ExperimentTemplate, Parametrisation


def _l2_template():
    """L2 template with 8 lower + 8 upper items, mixed variants."""
    items = []
    for i in range(8):
        items.append({
            "text": f"L_item_{i}_v{i % 3}",
            "bound_direction": "lower",
            "bound_value": str(100 + i * 10),
            "context_summary": f"lower {i}",
            "variant_id": i % 3,
            "variant_text": f"L_item_{i}_v{i % 3}",
        })
    for i in range(8):
        items.append({
            "text": f"U_item_{i}_v{i % 3}",
            "bound_direction": "upper",
            "bound_value": str(500 + i * 10),
            "context_summary": f"upper {i}",
            "variant_id": i % 3,
            "variant_text": f"U_item_{i}_v{i % 3}",
        })
    return ExperimentTemplate.model_validate({
        "experiment_type": "pure_reasoning_l2",
        "payload": {"item_type": "fact"},
        "question_examples": {
            "python_repo": {
                "hard_contextual": {
                    "question": (
                        "L2 prompt. Normalise to a common scale (seconds) before reasoning.\n"
                        "{n} facts:\n{facts_block}"
                    ),
                    "answer": "ok",
                },
            },
        },
        "rubric_criteria": [{"criterion": "endpoint_correctness", "weight": 1.0}],
        "grid": {"content_profile": ["python_repo"], "n_items": [4]},
        "fixed_pool": {"python_repo": items},
    })


def _l3_template_and_world_state():
    """L3 template with 4 quadrants of 4 items each."""
    items = []
    for live, direction, gate_var, var_value in [
        (True, "lower", "phase_rolling", True),
        (True, "upper", "phase_rolling", True),
        (False, "lower", "feature_x_off", False),
        (False, "upper", "feature_x_off", False),
    ]:
        for i in range(4):
            items.append({
                "text": f"if {gate_var} then {direction}_{i}",
                "bound_direction": direction,
                "bound_value": str((100 if direction == "lower" else 500) + i * 10),
                "context_summary": f"{direction} {i} live={live}",
                "variant_id": i % 3,
                "variant_text": f"{direction}_{i}",
                "gate_clause": f"if {gate_var} then",
                "gate_world_var": gate_var,
                "live": live,
            })
    template = ExperimentTemplate.model_validate({
        "experiment_type": "pure_reasoning_l3",
        "payload": {"item_type": "fact"},
        "question_examples": {
            "python_repo": {
                "hard_contextual": {
                    "question": (
                        "L3 prompt. World state:\n{world_state_block}\n"
                        "{n} facts:\n{facts_block}"
                    ),
                    "answer": "ok",
                },
            },
        },
        "rubric_criteria": [{"criterion": "endpoint_correctness", "weight": 1.0}],
        "grid": {"content_profile": ["python_repo"], "n_items": [4]},
        "fixed_pool": {"python_repo": items},
    })
    world_state = {
        "python_repo": {"phase_rolling": True, "feature_x_off": False}
    }
    return template, world_state


class TestL2Generation:
    def test_l2_writes_ak_with_unit_normalisation_in_question(self, tmp_path):
        template = _l2_template()
        param = Parametrisation(
            experiment_type="pure_reasoning_l2",
            content_profile="python_repo",
            n_items=4,
        )
        ak_path = tmp_path / "ak.yaml"
        generate_pure_reasoning_cell(
            template=template, parametrisation=param, answer_key_path=ak_path,
        )
        ak = AnswerKey.from_yaml(ak_path)
        assert ak.parametrisation_id == "pure_reasoning_l2__python_repo__n4"
        assert "Normalise to a common scale" in ak.expected_answers.question
        assert len(ak.items) == 4

    def test_l2_ak_items_carry_variant_metadata(self, tmp_path):
        """variant_id/variant_text must round-trip through the written YAML file."""
        import yaml
        template = _l2_template()
        param = Parametrisation(
            experiment_type="pure_reasoning_l2",
            content_profile="python_repo",
            n_items=4,
        )
        ak_path = tmp_path / "ak.yaml"
        generate_pure_reasoning_cell(
            template=template, parametrisation=param, answer_key_path=ak_path,
        )
        raw = yaml.safe_load(ak_path.read_text())
        for it in raw["items"]:
            assert "variant_id" in it
            assert "variant_text" in it


class TestL3Generation:
    def test_l3_n2_picks_only_live_items(self, tmp_path):
        import yaml
        template, world_state = _l3_template_and_world_state()
        param = Parametrisation(
            experiment_type="pure_reasoning_l3",
            content_profile="python_repo",
            n_items=2,
        )
        ak_path = tmp_path / "ak.yaml"
        generate_pure_reasoning_cell(
            template=template, parametrisation=param,
            answer_key_path=ak_path, world_state=world_state,
        )
        raw = yaml.safe_load(ak_path.read_text())
        for it in raw["items"]:
            assert it["live"] is True

    def test_l3_n4_covers_all_quadrants(self, tmp_path):
        import yaml
        template, world_state = _l3_template_and_world_state()
        param = Parametrisation(
            experiment_type="pure_reasoning_l3",
            content_profile="python_repo",
            n_items=4,
        )
        ak_path = tmp_path / "ak.yaml"
        generate_pure_reasoning_cell(
            template=template, parametrisation=param,
            answer_key_path=ak_path, world_state=world_state,
        )
        raw = yaml.safe_load(ak_path.read_text())
        quadrants = {(it["live"], it["bound_direction"]) for it in raw["items"]}
        assert quadrants == {(True, "lower"), (True, "upper"), (False, "lower"), (False, "upper")}

    def test_l3_question_includes_world_state_block(self, tmp_path):
        template, world_state = _l3_template_and_world_state()
        param = Parametrisation(
            experiment_type="pure_reasoning_l3",
            content_profile="python_repo",
            n_items=4,
        )
        ak_path = tmp_path / "ak.yaml"
        generate_pure_reasoning_cell(
            template=template, parametrisation=param,
            answer_key_path=ak_path, world_state=world_state,
        )
        ak = AnswerKey.from_yaml(ak_path)
        # World-state block must appear and contain both world variables
        assert "phase_rolling" in ak.expected_answers.question
        assert "feature_x_off" in ak.expected_answers.question

    def test_l3_correctness_mentions_live_items_only(self, tmp_path):
        template, world_state = _l3_template_and_world_state()
        param = Parametrisation(
            experiment_type="pure_reasoning_l3",
            content_profile="python_repo",
            n_items=4,
        )
        ak_path = tmp_path / "ak.yaml"
        generate_pure_reasoning_cell(
            template=template, parametrisation=param,
            answer_key_path=ak_path, world_state=world_state,
        )
        ak = AnswerKey.from_yaml(ak_path)
        assert "live" in ak.expected_answers.correctness.lower() or \
            "applicable" in ak.expected_answers.correctness.lower() or \
            "precondition" in ak.expected_answers.correctness.lower()


class TestL1StillWorks:
    """Make sure L1 (the existing pure_reasoning) still generates AKs identically."""
    def test_l1_unchanged(self, tmp_path):
        template = ExperimentTemplate.model_validate({
            "experiment_type": "pure_reasoning",
            "payload": {"item_type": "fact"},
            "question_examples": {
                "python_repo": {
                    "hard_contextual": {
                        "question": "L1: {n} facts:\n{facts_block}",
                        "answer": "ok",
                    },
                },
            },
            "rubric_criteria": [{"criterion": "endpoint_correctness", "weight": 1.0}],
            "grid": {"content_profile": ["python_repo"], "n_items": [2]},
            "fixed_pool": {
                "python_repo": [
                    {"text": "lower fact", "bound_direction": "lower", "bound_value": "100", "context_summary": "x"},
                    {"text": "upper fact", "bound_direction": "upper", "bound_value": "500", "context_summary": "y"},
                ],
            },
        })
        param = Parametrisation(
            experiment_type="pure_reasoning",
            content_profile="python_repo",
            n_items=2,
        )
        ak_path = tmp_path / "ak.yaml"
        generate_pure_reasoning_cell(
            template=template, parametrisation=param, answer_key_path=ak_path,
        )
        ak = AnswerKey.from_yaml(ak_path)
        assert ak.parametrisation_id == "pure_reasoning__python_repo__n2"
        assert len(ak.items) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `poetry run pytest tests/test_pure_reasoning_l2_l3.py -v`
Expected: FAIL — L2/L3 dispatch and `world_state` parameter don't exist yet.

- [ ] **Step 3: Refactor `generate_pure_reasoning_cell` to dispatch by experiment_type**

Replace the body of `src/agent_retrieval/generator/pure_reasoning_gen.py` with this updated version (preserving all existing helpers and adding the new dispatch logic):

```python
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from agent_retrieval.generator.fixed_pool import sample_fixed_pool, sample_fixed_pool_l3
from agent_retrieval.schema.template import ExperimentTemplate, Parametrisation


def _format_facts_block(items: list[dict[str, Any]]) -> str:
    """Format selected facts as a numbered list for inclusion in the prompt."""
    return "\n".join(f"{i}. {item['text']}" for i, item in enumerate(items, start=1))


def _format_world_state_block(world_state: dict[str, Any]) -> str:
    """Render a world-state dict as a bulleted block."""
    return "\n".join(f"- {k} = {v}" for k, v in world_state.items())


def _to_comparable(value: str) -> float:
    """Parse a bound_value string into a comparable float.

    Handles plain numeric strings ('300', '1.5') and 12-hour time-of-day
    strings ('7:30 PM', '12:15 AM') — the latter converted to minutes
    since midnight.
    """
    s = value.strip()
    try:
        return float(s)
    except ValueError:
        pass
    s_upper = s.upper()
    is_pm = "PM" in s_upper
    is_am = "AM" in s_upper
    if not (is_pm or is_am):
        raise ValueError(f"unrecognised bound_value: {value!r}")
    s_clean = s_upper.replace("PM", "").replace("AM", "").strip()
    if ":" in s_clean:
        h_str, m_str = s_clean.split(":", 1)
        h = int(h_str)
        m = int(m_str)
    else:
        h = int(s_clean)
        m = 0
    if is_pm and h != 12:
        h += 12
    elif is_am and h == 12:
        h = 0
    return float(h * 60 + m)


def _compute_expected_endpoints(items: list[dict[str, Any]]) -> tuple[str, str, str, str]:
    """Compute (max_lower_value, max_lower_text, min_upper_value, min_upper_text).

    For L3 callers should pre-filter to live items.
    """
    lowers = [it for it in items if it["bound_direction"] == "lower"]
    uppers = [it for it in items if it["bound_direction"] == "upper"]
    if not lowers or not uppers:
        return "", "", "", ""
    max_lower = max(lowers, key=lambda it: _to_comparable(it["bound_value"]))
    min_upper = min(uppers, key=lambda it: _to_comparable(it["bound_value"]))
    return (
        max_lower["bound_value"],
        max_lower["text"],
        min_upper["bound_value"],
        min_upper["text"],
    )


def _build_ak_item(idx: int, it: dict[str, Any]) -> dict[str, Any]:
    """Build a single AK item dict from a pool item, including any extra metadata fields."""
    ak_item = {
        "item_id": f"target_{idx:03d}",
        "inserted_text": it["text"],
        "value": it["bound_value"],
        "bound_direction": it["bound_direction"],
        "context_summary": it.get("context_summary", ""),
    }
    # Carry through L2/L3 metadata if present
    for extra in ("variant_id", "variant_text", "gate_clause", "gate_world_var", "live"):
        if extra in it:
            ak_item[extra] = it[extra]
    return ak_item


def generate_pure_reasoning_cell(
    template: ExperimentTemplate,
    parametrisation: Parametrisation,
    answer_key_path: Path,
    world_state: dict[str, dict[str, Any]] | None = None,
) -> None:
    """Generate one pure_reasoning / pure_reasoning_l2 / pure_reasoning_l3 answer key.

    No corpus, no LLM calls — purely deterministic.

    For L3, `world_state` must be provided as a dict keyed by content_profile.
    """
    if answer_key_path.exists():
        return

    profile = parametrisation.content_profile
    pool = template.fixed_pool.get(profile, [])
    if not pool:
        raise ValueError(f"fixed_pool is empty for profile {profile}")

    n = parametrisation.n_items or 1
    is_l3 = template.experiment_type == "pure_reasoning_l3"

    if is_l3:
        if world_state is None or profile not in world_state:
            raise ValueError(f"world_state required for pure_reasoning_l3, profile {profile}")
        selected = sample_fixed_pool_l3(
            pool, n=n, parametrisation_id=parametrisation.parametrisation_id,
        )
    else:
        selected = sample_fixed_pool(
            pool, n=n, parametrisation_id=parametrisation.parametrisation_id,
            balance_key="bound_direction",
        )

    profile_examples = template.question_examples.get(profile, {})
    example = profile_examples.get("hard_contextual")
    if example is None:
        raise ValueError(f"No 'hard_contextual' example for profile {profile}")

    facts_block = _format_facts_block(selected)
    question = example.question.replace("{n}", str(n)).replace("{facts_block}", facts_block)
    if is_l3:
        ws_block = _format_world_state_block(world_state[profile])
        question = question.replace("{world_state_block}", ws_block)

    items_yaml = [_build_ak_item(i, it) for i, it in enumerate(selected, start=1)]

    # For L3, compute endpoints from live items only.
    endpoint_items = [it for it in selected if it.get("live", True)]
    max_lower_val, max_lower_text, min_upper_val, min_upper_text = _compute_expected_endpoints(endpoint_items)

    rubric_yaml = [{"criterion": c.criterion, "weight": c.weight} for c in template.rubric_criteria]

    if is_l3:
        correctness = (
            f"The answer must derive the narrowest valid window from the *live* "
            f"(applicable) facts only. A fact is live iff its precondition is "
            f"satisfied by the world state given in the prompt. The lower endpoint "
            f"is established by the live fact whose bound_direction is 'lower' and "
            f"bound_value is largest: {max_lower_text!r} (value: {max_lower_val}). "
            f"The upper endpoint is established by the live fact whose "
            f"bound_direction is 'upper' and bound_value is smallest: "
            f"{min_upper_text!r} (value: {min_upper_val}). The agent must cite both "
            f"facts. The agent's cited evidence must come from the {n} facts in the "
            f"question. Citing a dead (precondition-not-satisfied) fact toward the "
            f"endpoints is incorrect. Citing a fact not present in the question is a "
            f"HALLUCINATION."
        )
        completeness = (
            f"The agent must classify each of the {n} facts as live/dead per the "
            f"world state, and (for live facts) as lower or upper bound consistent "
            f"with the answer-key bound_direction field. Misclassifying a live "
            f"fact as dead, or treating a dead fact as binding, reduces "
            f"classification_accuracy. Hallucinated facts also reduce it."
        )
    else:
        correctness = (
            f"The answer must derive the narrowest valid window. The lower endpoint "
            f"is established by the fact whose bound_direction is 'lower' and "
            f"bound_value is largest: {max_lower_text!r} (value: {max_lower_val}). "
            f"The upper endpoint is established by the fact whose bound_direction "
            f"is 'upper' and bound_value is smallest: {min_upper_text!r} "
            f"(value: {min_upper_val}). The agent must cite both facts. The "
            f"agent's cited evidence must come from the {n} facts provided in the "
            f"question. Any cited fact that is not verbatim one of the {n} provided "
            f"facts is a HALLUCINATION and must be penalized under the "
            f"endpoint_correctness criterion."
        )
        completeness = (
            f"The agent must classify each of the {n} facts as a lower or upper "
            f"bound consistent with the answer-key bound_direction field, OR "
            f"justify any deviation in the reasoning. Classifications attached to "
            f"facts not actually present in the question prompt are hallucinations "
            f"and reduce classification_accuracy."
        )

    parameters = {"content_profile": profile, "n_items": n}

    ak_dict = {
        "parametrisation_id": parametrisation.parametrisation_id,
        "experiment_type": template.experiment_type,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "parameters": parameters,
        "items": items_yaml,
        "expected_answers": {
            "question": question,
            "correctness": correctness,
            "completeness": completeness,
        },
        "rubric_criteria": rubric_yaml,
    }

    answer_key_path.parent.mkdir(parents=True, exist_ok=True)
    with open(answer_key_path, "w") as f:
        yaml.dump(ak_dict, f, default_flow_style=False, sort_keys=False)
```

- [ ] **Step 4: Run all pure_reasoning tests to verify they pass**

Run: `poetry run pytest tests/test_pure_reasoning_l2_l3.py tests/test_pure_reasoning_gen.py -v`
Expected: all tests PASS (existing L1 tests unchanged, new L2/L3 tests pass).

- [ ] **Step 5: Commit**

```bash
git add src/agent_retrieval/generator/pure_reasoning_gen.py tests/test_pure_reasoning_l2_l3.py
git commit -m "generator: dispatch pure_reasoning L1/L2/L3 in generate_pure_reasoning_cell"
```

---

## Task 8: Wire `generate_parallel.py` dispatch

**Files:**
- Modify: `scripts/generate_parallel.py:126-133`

The current code dispatches only on exact string match `"pure_reasoning"`. Broaden it to also match L2 and L3 — and pass `world_state` through to L3.

- [ ] **Step 1: Update `process_one` in `scripts/generate_parallel.py`**

Replace the existing `pure_reasoning` block (currently at lines ~126-133):

```python
    # pure_reasoning has no corpus and no LLM call; just write the answer key.
    if template.experiment_type == "pure_reasoning":
        try:
            generate_pure_reasoning_cell(
                template=template, parametrisation=param, answer_key_path=answer_key_path,
            )
            return (pid, True, "done", None)
        except Exception as e:
            return (pid, False, str(e), None)
```

with this expanded version that handles all three pure-reasoning types and reads `world_state` from raw YAML for L3:

```python
    # pure_reasoning (L1/L2/L3) has no corpus and no LLM call; just write the answer key.
    if template.experiment_type in ("pure_reasoning", "pure_reasoning_l2", "pure_reasoning_l3"):
        try:
            world_state = None
            if template.experiment_type == "pure_reasoning_l3":
                # world_state lives at template-file top level, outside the typed schema.
                import yaml
                exp_yaml_path = next(
                    (p for p in (workspace_dir.parent / "experiments").glob("*.yaml")
                     if p.stem == experiment_name),
                    None,
                )
                if exp_yaml_path is None:
                    return (pid, False, f"experiment file missing for {experiment_name}", None)
                raw = yaml.safe_load(exp_yaml_path.read_text())
                world_state = raw.get("world_state", {})
            generate_pure_reasoning_cell(
                template=template, parametrisation=param,
                answer_key_path=answer_key_path, world_state=world_state,
            )
            return (pid, True, "done", None)
        except Exception as e:
            return (pid, False, str(e), None)
```

Note: `workspace_dir.parent / "experiments"` resolves to the project's `experiments/` directory because `workspace_dir` is `Path(args.workspace).resolve()`, which is `<project>/workspace`. This is the same path the dispatcher otherwise gets via the `experiments-dir` arg, but since `process_one` doesn't currently take `experiments_dir`, we infer it from the workspace path. (Alternatively, plumb `args.experiments_dir` through — see step 2.)

- [ ] **Step 2: Plumb `experiments_dir` through to `process_one` (cleaner)**

Replace the inferred path with an explicit argument. Edit `scripts/generate_parallel.py`:

In `process_one`, add `experiments_dir: Path` parameter:

```python
async def process_one(
    experiment_name: str,
    parametrisation_id: str,
    template: ExperimentTemplate,
    workspace_dir: Path,
    worker_id: int,
    experiments_dir: Path,
) -> tuple[str, bool, str, InsertionStats | None]:
```

Then in the L3 branch use `experiments_dir / f"{experiment_name}.yaml"` directly:

```python
            world_state = None
            if template.experiment_type == "pure_reasoning_l3":
                import yaml
                exp_yaml_path = experiments_dir / f"{experiment_name}.yaml"
                if not exp_yaml_path.exists():
                    return (pid, False, f"experiment file missing: {exp_yaml_path}", None)
                raw = yaml.safe_load(exp_yaml_path.read_text())
                world_state = raw.get("world_state", {})
```

In `run_workers`, accept and pass `experiments_dir`:

```python
async def run_workers(
    remaining: list[tuple[str, str, ExperimentTemplate]],
    workspace_dir: Path,
    max_workers: int,
    experiments_dir: Path,
) -> None:
```

```python
            result_pid, success, msg, stats = await process_one(
                exp_name, pid, template, workspace_dir, idx, experiments_dir,
            )
```

In `main`, pass `experiments_dir`:

```python
    asyncio.run(run_workers(remaining, workspace_dir, args.workers, experiments_dir))
```

- [ ] **Step 3: Smoke test — generate one L2 cell and one L3 cell**

Run from `experiments/` filtering to one experiment at a time:

```bash
poetry run python scripts/generate_parallel.py --workers 3 --experiments pure_reasoning_l2 --dry-run
```

Expected: prints "would generate: pure_reasoning_l2__python_repo__n2", etc. (10 cells total).

Then live (write actually):

```bash
poetry run python scripts/generate_parallel.py --workers 3 --experiments pure_reasoning_l2
```

Expected: 10 AKs written under `workspace/judge/answer_keys/pure_reasoning_l2__*.yaml`. No errors.

Then L3:

```bash
poetry run python scripts/generate_parallel.py --workers 3 --experiments pure_reasoning_l3
```

Expected: 10 AKs written under `workspace/judge/answer_keys/pure_reasoning_l3__*.yaml`. No errors.

- [ ] **Step 4: Inspect a generated L3 AK to confirm shape**

Run:

```bash
poetry run python -c "
import yaml
from pathlib import Path
ak = yaml.safe_load(Path('workspace/judge/answer_keys/pure_reasoning_l3__python_repo__n4.yaml').read_text())
print('parametrisation_id:', ak['parametrisation_id'])
print('experiment_type:', ak['experiment_type'])
print('items in AK:', len(ak['items']))
quadrants = {(it.get('live'), it['bound_direction']) for it in ak['items']}
print('quadrants:', quadrants)
print('world-state in question:', 'phase' in ak['expected_answers']['question'].lower() or 'rolling' in ak['expected_answers']['question'].lower())
print('correctness mentions live:', 'live' in ak['expected_answers']['correctness'].lower())
"
```

Expected: prints values consistent with N=4 fully covering all 4 quadrants, world_state present in question, correctness mentions live.

- [ ] **Step 5: Commit**

```bash
git add scripts/generate_parallel.py
git commit -m "generator: dispatch pure_reasoning_l2/l3 from parallel generator"
```

(The newly written AKs are workspace state and not committed.)

---

## Task 9: Create batch YAML files for the cognitive ladder

**Files:**
- Create: `batches/cognitive-ladder_opus-4-7_effort-max.yaml`
- Create: `batches/cognitive-ladder_opus-4-6_effort-max.yaml`
- Create: `batches/cognitive-ladder_sonnet-4-6_effort-max.yaml`

Each batch runs L2 + L3 across the same 5 n_items × 2 profiles grid × 3 repeats = 60 runs. Mirrors the multi-reasoning-dissection batch shape.

- [ ] **Step 1: Create opus-4-7 batch**

Create `batches/cognitive-ladder_opus-4-7_effort-max.yaml`:

```yaml
batch_name: "cognitive-ladder_opus-4-7_effort-max"
max_parallel: 3
retry_failed: true
agent_model: "claude-opus-4-7"
effort_mode: "max"
n_repeats: 3
max_turns: 75
allowed_tools: ["Read", "Glob", "Grep", "Bash"]

# Cognitive ladder layered on pure_reasoning:
# - pure_reasoning_l2: + per-item unit normalisation
# - pure_reasoning_l3: + conditional gating on top of L2
#
# pure_reasoning_l2: 2 profiles x 5 n_items x 3 repeats = 30 runs
# pure_reasoning_l3: 2 profiles x 5 n_items x 3 repeats = 30 runs
# Total per batch: 60 runs
experiments:
  - experiment_type: "pure_reasoning_l2"
    grid:
      content_profile: [python_repo, noir_fiction]
      n_items: [2, 4, 8, 12, 16]
  - experiment_type: "pure_reasoning_l3"
    grid:
      content_profile: [python_repo, noir_fiction]
      n_items: [2, 4, 8, 12, 16]
```

- [ ] **Step 2: Create opus-4-6 and sonnet-4-6 batches**

Create `batches/cognitive-ladder_opus-4-6_effort-max.yaml` — identical to the opus-4-7 batch except:
- `batch_name: "cognitive-ladder_opus-4-6_effort-max"`
- `agent_model: "claude-opus-4-6"`

Create `batches/cognitive-ladder_sonnet-4-6_effort-max.yaml` — identical except:
- `batch_name: "cognitive-ladder_sonnet-4-6_effort-max"`
- `agent_model: "claude-sonnet-4-6"`

- [ ] **Step 3: Verify all 3 batch files load via the runner's BatchConfig**

Run:

```bash
poetry run python -c "
from pathlib import Path
from agent_retrieval.schema.batch import BatchConfig
for name in ['cognitive-ladder_opus-4-7_effort-max', 'cognitive-ladder_opus-4-6_effort-max', 'cognitive-ladder_sonnet-4-6_effort-max']:
    b = BatchConfig.from_yaml(Path(f'batches/{name}.yaml'))
    print(f'{name}: agent={b.agent_model}, n_repeats={b.n_repeats}, experiments={len(b.experiments)}')
"
```

Expected output:
```
cognitive-ladder_opus-4-7_effort-max: agent=claude-opus-4-7, n_repeats=3, experiments=2
cognitive-ladder_opus-4-6_effort-max: agent=claude-opus-4-6, n_repeats=3, experiments=2
cognitive-ladder_sonnet-4-6_effort-max: agent=claude-sonnet-4-6, n_repeats=3, experiments=2
```

- [ ] **Step 4: Commit**

```bash
git add batches/cognitive-ladder_opus-4-7_effort-max.yaml batches/cognitive-ladder_opus-4-6_effort-max.yaml batches/cognitive-ladder_sonnet-4-6_effort-max.yaml
git commit -m "batches: cognitive-ladder L2/L3 at max effort for 3 models"
```

---

## Task 10: Final smoke — run a tiny live batch on one model to confirm wiring

**Files:** None modified

This is a sanity check before kicking off the full 3-batch run. Runs a single L2 cell (smallest n) on the cheapest model to confirm the new generation, runner pipeline, and AK can talk to each other.

- [ ] **Step 1: Confirm AKs are present**

```bash
ls workspace/judge/answer_keys/pure_reasoning_l2__*.yaml | wc -l
ls workspace/judge/answer_keys/pure_reasoning_l3__*.yaml | wc -l
```

Expected: each prints `10`.

- [ ] **Step 2: Manually inspect one AK each**

```bash
cat workspace/judge/answer_keys/pure_reasoning_l2__python_repo__n2.yaml | head -50
echo "==="
cat workspace/judge/answer_keys/pure_reasoning_l3__python_repo__n2.yaml | head -50
```

Expected: question contains the appropriate clauses; items list looks well-formed.

- [ ] **Step 3: Run all tests as a final regression check**

```bash
poetry run pytest -v
```

Expected: all tests PASS.

---

## Self-Review Checklist (for plan author — me)

**Spec coverage (against `docs/superpowers/specs/2026-05-02-cognitive-ladder-l2-l3-design.md`):**
- [x] Cumulative layering (L2 inherits L1 items; L3 inherits L2 surface forms exactly): Tasks 3 and 5 build L3 atop L2's `variant_text`s.
- [x] L2 three-variant per-item-index assignment: Tasks 3 and 4 fix `variant_id` per item.
- [x] L2 self-contained anchors: variants 2 (offset-from-reference) include the reference inline in the same item — see python item 3 ("Reference: HOURLY_KICK_S = 3600").
- [x] L3 four-quadrant pool stratification: Tasks 5 and 6 produce 4-4-4-4 pools, verified in step 3 of each.
- [x] L3 four gate variables; 2 TRUE + 2 FALSE: Tasks 5 and 6 enforce this.
- [x] L3 sampling for N=2 from live quadrants only: Task 2 + tests in Task 7.
- [x] L3 sampling for N=3 with extra random live: Task 2 + tests.
- [x] L3 sampling for N≥4 across all quadrants: Task 2 + tests.
- [x] World-state block always printed in full: rendered from `_format_world_state_block` in Task 7.
- [x] L1 unchanged: Task 7's `TestL1StillWorks` regression test.
- [x] Identifiers `pure_reasoning_l2__…`, `pure_reasoning_l3__…`: Task 1 schema literal + new template files in Tasks 3 and 5.
- [x] Schema additive only (no AnswerKeyItem changes): metadata fields written via raw YAML dict in Task 7's `_build_ak_item`; pydantic schema drops them on parse but they remain in the file.

**Placeholder scan:** No "TBD"/"TODO" markers; no "implement later"/"add validation" placeholders; every code step shows the exact code; every command shows expected output.

**Type consistency:** `sample_fixed_pool_l3` signature `(pool, n, parametrisation_id) -> list[dict]` matches across Tasks 2 and 7. `generate_pure_reasoning_cell` signature gains `world_state` kwarg in Task 7 and is invoked with that kwarg in Task 8. Pool item field names (`variant_id`, `variant_text`, `gate_clause`, `gate_world_var`, `live`) used identically in Tasks 3-7. `world_state` keyed by content_profile with snake_case variable names matches between L3 templates (Tasks 5/6) and the generator (Task 7).
