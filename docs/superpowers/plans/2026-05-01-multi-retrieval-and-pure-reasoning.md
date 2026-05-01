# Multi-Retrieval and Pure-Reasoning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two new experiment types — `multi_retrieval` and `pure_reasoning` — that dissect the existing `multi_reasoning` task into its retrieval+retention and cross-fact-reasoning halves. Both experiments use hand-authored 16-item pools committed to YAML, with N items sampled per cell.

**Architecture:** Extend the existing v2 generator pipeline with two new experiment types:
- `multi_retrieval` reuses the corpus → assembly → insertion pipeline but bypasses LLM pool generation in favor of a YAML-committed fixed pool. The insertion agent inserts pre-authored needles instead of inventing them.
- `pure_reasoning` skips corpus generation entirely. The generator selects N facts from the YAML pool, formats the prompt, and writes the answer key directly. The runner detects the experiment type and skips corpus loading.

The schema gains: two new `experiment_type` literals; optional corpus-related axes in `GridSpec`/`Parametrisation` (so `pure_reasoning` can omit them); a `fixed_pool` field on `ExperimentTemplate`; optional `value`, `bound_direction`, `file_path`, and `line_range` fields on `AnswerKeyItem`.

**Tech Stack:** Python 3.12+, Poetry, Pydantic v2, pytest, `claude_agent_sdk`, Claude Code CLI.

**Spec reference:** `docs/superpowers/specs/2026-05-01-multi-retrieval-and-pure-reasoning-design.md`

---

## Pre-flight

- [ ] **Verify clean working tree.**

Run:
```bash
git status
```

Expected: only the spec doc and untracked workspace artifacts. If any unrelated source files are dirty, stash or commit them first.

- [ ] **Confirm test suite is green before changes.**

Run:
```bash
poetry run pytest -v
```

Expected: all tests pass. Record the count as a baseline.

---

## Task 1: Schema — extend `experiment_type` to include the two new types

**Why:** `ExperimentTemplate.experiment_type` is currently a Literal of three values. Adding `multi_retrieval` and `pure_reasoning` is the foundation for everything else.

**Files:**
- Modify: `src/agent_retrieval/schema/template.py`
- Test: `tests/test_template_schema.py`

- [ ] **Step 1: Write failing test for the two new experiment types validating.**

Append to `tests/test_template_schema.py` after the existing `TestExperimentTemplate` class:

```python
class TestNewExperimentTypes:
    def _base_dict(self, experiment_type: str) -> dict:
        return {
            "experiment_type": experiment_type,
            "payload": {"item_type": "fact"},
            "question_examples": {
                "python_repo": {
                    "hard_contextual": {
                        "question": "q",
                        "answer": "a",
                    },
                },
            },
            "rubric_criteria": [{"criterion": "recall", "weight": 1.0}],
            "grid": {
                "content_profile": ["python_repo"],
                "corpus_token_count": [800000],
                "discriminability": ["hard"],
                "reference_clarity": ["contextual"],
                "n_items": [2, 4],
            },
        }

    def test_multi_retrieval_validates(self):
        tmpl = ExperimentTemplate.model_validate(self._base_dict("multi_retrieval"))
        assert tmpl.experiment_type == "multi_retrieval"

    def test_pure_reasoning_validates(self):
        d = self._base_dict("pure_reasoning")
        tmpl = ExperimentTemplate.model_validate(d)
        assert tmpl.experiment_type == "pure_reasoning"
```

- [ ] **Step 2: Run test to verify it fails.**

```bash
poetry run pytest tests/test_template_schema.py::TestNewExperimentTypes -v
```

Expected: FAIL — pydantic rejects the new literal values.

- [ ] **Step 3: Extend the Literal in `ExperimentTemplate`.**

In `src/agent_retrieval/schema/template.py`, change:

```python
experiment_type: Literal["single_needle", "multi_chain", "multi_reasoning"]
```

to:

```python
experiment_type: Literal[
    "single_needle",
    "multi_chain",
    "multi_reasoning",
    "multi_retrieval",
    "pure_reasoning",
]
```

Also update the `validate_grid_n_items` method body so the multi-types check covers the new types:

```python
@model_validator(mode="after")
def validate_grid_n_items(self) -> ExperimentTemplate:
    multi_types = {"multi_chain", "multi_reasoning", "multi_retrieval", "pure_reasoning"}
    is_multi = self.experiment_type in multi_types
    has_n_items = self.grid.n_items is not None
    if is_multi and not has_n_items:
        raise ValueError(
            f"Experiment type '{self.experiment_type}' requires 'n_items' in grid"
        )
    if not is_multi and has_n_items:
        raise ValueError(
            f"Experiment type '{self.experiment_type}' must not have 'n_items' in grid"
        )
    return self
```

- [ ] **Step 4: Run test to verify it passes.**

```bash
poetry run pytest tests/test_template_schema.py::TestNewExperimentTypes -v
```

Expected: PASS.

- [ ] **Step 5: Run full schema test suite to verify no regressions.**

```bash
poetry run pytest tests/test_template_schema.py tests/test_schema.py -v
```

Expected: all PASS.

- [ ] **Step 6: Commit.**

```bash
git add src/agent_retrieval/schema/template.py tests/test_template_schema.py
git commit -m "$(cat <<'EOF'
schema: register multi_retrieval and pure_reasoning experiment types

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Schema — make corpus-related grid axes optional

**Why:** `pure_reasoning` doesn't generate a corpus, so `corpus_token_count`, `discriminability`, and `reference_clarity` don't apply. Make them optional in `GridSpec` and `Parametrisation`. The `parametrisation_id` should omit them when absent.

**Files:**
- Modify: `src/agent_retrieval/schema/template.py`
- Test: `tests/test_template_schema.py`

- [ ] **Step 1: Write failing test for optional axes.**

Append to `tests/test_template_schema.py` inside the existing `TestNewExperimentTypes` class:

```python
def test_pure_reasoning_grid_omits_corpus_axes(self):
    """pure_reasoning grids may omit corpus-related axes."""
    d = self._base_dict("pure_reasoning")
    del d["grid"]["corpus_token_count"]
    del d["grid"]["discriminability"]
    del d["grid"]["reference_clarity"]
    tmpl = ExperimentTemplate.model_validate(d)
    assert tmpl.grid.corpus_token_count is None
    assert tmpl.grid.discriminability is None
    assert tmpl.grid.reference_clarity is None

def test_pure_reasoning_parametrisation_id_omits_corpus_segments(self):
    """When corpus axes are None, the parametrisation_id omits them."""
    p = Parametrisation(
        experiment_type="pure_reasoning",
        content_profile="python_repo",
        n_items=8,
    )
    assert p.parametrisation_id == "pure_reasoning__python_repo__n8"
```

- [ ] **Step 2: Run test to verify it fails.**

```bash
poetry run pytest tests/test_template_schema.py::TestNewExperimentTypes::test_pure_reasoning_grid_omits_corpus_axes tests/test_template_schema.py::TestNewExperimentTypes::test_pure_reasoning_parametrisation_id_omits_corpus_segments -v
```

Expected: FAIL — fields are required.

- [ ] **Step 3: Make fields optional in `GridSpec` and `Parametrisation`.**

In `src/agent_retrieval/schema/template.py`:

Replace `GridSpec` with:

```python
class GridSpec(BaseModel):
    content_profile: list[str]
    corpus_token_count: list[int] | None = None
    discriminability: list[Literal["easy", "hard"]] | None = None
    reference_clarity: list[Literal["exact", "synonym", "contextual"]] | None = None
    n_items: list[int] | None = None
```

Replace `Parametrisation` with:

```python
class Parametrisation(BaseModel):
    experiment_type: str
    content_profile: str
    corpus_token_count: int | None = None
    discriminability: str | None = None
    reference_clarity: str | None = None
    n_items: int | None = None

    @property
    def parametrisation_id(self) -> str:
        parts: list[str] = [self.experiment_type, self.content_profile]
        if self.corpus_token_count is not None:
            parts.append(_format_token_count(self.corpus_token_count))
        if self.discriminability is not None:
            parts.append(self.discriminability)
        if self.reference_clarity is not None:
            parts.append(self.reference_clarity)
        if self.n_items is not None:
            parts.append(f"n{self.n_items}")
        return "__".join(parts)
```

- [ ] **Step 4: Update `expand_gridspec` to skip None axes.**

In `src/agent_retrieval/generator/grid.py`, replace `expand_gridspec` with:

```python
def expand_gridspec(grid: GridSpec, experiment_type: str) -> list[Parametrisation]:
    dimensions: list[tuple[str, list[Any]]] = [("content_profile", grid.content_profile)]
    if grid.corpus_token_count is not None:
        dimensions.append(("corpus_token_count", grid.corpus_token_count))
    if grid.discriminability is not None:
        dimensions.append(("discriminability", grid.discriminability))
    if grid.reference_clarity is not None:
        dimensions.append(("reference_clarity", grid.reference_clarity))
    if grid.n_items is not None:
        dimensions.append(("n_items", grid.n_items))

    keys = [k for k, _ in dimensions]
    values = [v for _, v in dimensions]

    parametrisations: list[Parametrisation] = []
    for combo in itertools.product(*values):
        params = dict(zip(keys, combo))
        params["experiment_type"] = experiment_type
        parametrisations.append(Parametrisation(**params))

    return parametrisations
```

- [ ] **Step 5: Run new tests to verify they pass.**

```bash
poetry run pytest tests/test_template_schema.py::TestNewExperimentTypes -v
```

Expected: all PASS.

- [ ] **Step 6: Run full test suite to verify no regressions.**

```bash
poetry run pytest -v
```

Expected: all PASS. (Existing parametrisation_id tests still work because they always supply all fields.)

- [ ] **Step 7: Commit.**

```bash
git add src/agent_retrieval/schema/template.py src/agent_retrieval/generator/grid.py tests/test_template_schema.py
git commit -m "$(cat <<'EOF'
schema: make corpus-related grid axes optional for pure_reasoning

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Schema — add `fixed_pool` field to `ExperimentTemplate`

**Why:** Both new experiments hold their 16-item pools in YAML. `ExperimentTemplate` needs a permissive `fixed_pool` field that maps content_profile → list of item dicts (shape varies by experiment type; validated downstream by the generator).

**Files:**
- Modify: `src/agent_retrieval/schema/template.py`
- Test: `tests/test_template_schema.py`

- [ ] **Step 1: Write failing test.**

Append to `tests/test_template_schema.py` inside `TestNewExperimentTypes`:

```python
def test_fixed_pool_field_accepts_per_profile_items(self):
    d = self._base_dict("multi_retrieval")
    d["fixed_pool"] = {
        "python_repo": [
            {"inserted_text": "canary_traffic_split = 5", "value": "5",
             "content_hint": "in deploy/canary.py"},
            {"inserted_text": "evaluation_window_hours = 6", "value": "6",
             "content_hint": "in deploy/canary.py"},
        ],
    }
    tmpl = ExperimentTemplate.model_validate(d)
    assert "python_repo" in tmpl.fixed_pool
    assert len(tmpl.fixed_pool["python_repo"]) == 2
    assert tmpl.fixed_pool["python_repo"][0]["value"] == "5"

def test_fixed_pool_field_optional(self):
    """Existing experiment types without fixed_pool still validate."""
    d = self._base_dict("multi_retrieval")
    tmpl = ExperimentTemplate.model_validate(d)
    assert tmpl.fixed_pool == {}
```

- [ ] **Step 2: Run test to verify it fails.**

```bash
poetry run pytest tests/test_template_schema.py::TestNewExperimentTypes::test_fixed_pool_field_accepts_per_profile_items tests/test_template_schema.py::TestNewExperimentTypes::test_fixed_pool_field_optional -v
```

Expected: FAIL — `fixed_pool` is not a field.

- [ ] **Step 3: Add `fixed_pool` field to `ExperimentTemplate`.**

In `src/agent_retrieval/schema/template.py`, add a field to the `ExperimentTemplate` class. Place it after `question_examples`:

```python
fixed_pool: dict[str, list[dict[str, Any]]] = {}
```

You'll also need `from typing import Any` at the top of the file if not already imported.

- [ ] **Step 4: Run tests to verify they pass.**

```bash
poetry run pytest tests/test_template_schema.py::TestNewExperimentTypes -v
```

Expected: PASS.

- [ ] **Step 5: Commit.**

```bash
git add src/agent_retrieval/schema/template.py tests/test_template_schema.py
git commit -m "$(cat <<'EOF'
schema: add fixed_pool field to ExperimentTemplate

Permissive dict[content_profile, list[item_dict]] field that holds
hand-authored 16-item pools for multi_retrieval and pure_reasoning.
Per-experiment-type validation happens in the generator.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Schema — extend `AnswerKeyItem` with optional fields

**Why:** Multi-retrieval items need a `value` field (the parsed numeric value or time). Pure-reasoning items need `bound_direction` (lower/upper) and don't have `file_path`/`line_range` (no corpus). Make these fields optional with defaults, preserving backward compatibility for existing answer keys.

**Files:**
- Modify: `src/agent_retrieval/schema/answer_key.py`
- Test: `tests/test_schema.py`

- [ ] **Step 1: Write failing tests for the new fields.**

Append to `tests/test_schema.py` (or create a new file `tests/test_answer_key_schema.py` if `test_schema.py` is for `ExperimentSpec` only — check existing structure first; if creating new, follow the same import style).

```python
from agent_retrieval.schema.answer_key import AnswerKey, AnswerKeyItem


class TestAnswerKeyItemNewFields:
    def test_existing_item_validates_without_new_fields(self):
        """Backward-compat: legacy answer keys still parse."""
        item = AnswerKeyItem(
            item_id="target_001",
            inserted_text="x = 1",
            file_path="src/foo.py",
            line_range=[10, 10],
            context_summary="inserted at top of foo.py",
        )
        assert item.value is None
        assert item.bound_direction is None

    def test_multi_retrieval_item_with_value(self):
        item = AnswerKeyItem(
            item_id="target_001",
            inserted_text="canary_traffic_split = 5",
            file_path="deploy/canary.py",
            line_range=[10, 10],
            context_summary="canary traffic split percentage",
            value="5",
        )
        assert item.value == "5"

    def test_pure_reasoning_item_omits_corpus_fields(self):
        """pure_reasoning items have no file_path or line_range."""
        item = AnswerKeyItem(
            item_id="target_001",
            inserted_text="REPLICATION_LAG_RECOVERY_S = 300",
            context_summary="lower bound on migration start time",
            value="300",
            bound_direction="lower",
        )
        assert item.file_path is None
        assert item.line_range is None
        assert item.bound_direction == "lower"
```

- [ ] **Step 2: Run tests to verify they fail.**

```bash
poetry run pytest tests/test_schema.py::TestAnswerKeyItemNewFields -v
```

(Or `tests/test_answer_key_schema.py` if you created a new file.)

Expected: FAIL on the missing fields and on the missing-required-field assertion for `pure_reasoning`.

- [ ] **Step 3: Make fields optional and add new ones.**

In `src/agent_retrieval/schema/answer_key.py`, replace `AnswerKeyItem`:

```python
from typing import Literal


class AnswerKeyItem(BaseModel):
    item_id: str
    inserted_text: str
    file_path: str | None = None
    line_range: list[int] | None = None
    context_summary: str
    value: str | None = None
    bound_direction: Literal["lower", "upper"] | None = None
```

- [ ] **Step 4: Run tests to verify they pass.**

```bash
poetry run pytest tests/test_schema.py::TestAnswerKeyItemNewFields -v
```

Expected: PASS.

- [ ] **Step 5: Run full test suite to verify no regressions.**

```bash
poetry run pytest -v
```

Expected: all PASS.

- [ ] **Step 6: Commit.**

```bash
git add src/agent_retrieval/schema/answer_key.py tests/test_schema.py
git commit -m "$(cat <<'EOF'
schema: extend AnswerKeyItem with optional value, bound_direction fields

file_path and line_range become optional so pure_reasoning items
(which have no corpus) can be represented. value carries the parsed
numeric/time. bound_direction marks lower/upper bound for reasoning.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Generator — fixed-pool sampling helper

**Why:** Both new experiments need the same "given a parametrisation_id, sample N items from the 16-item pool deterministically" behavior. Extract this into a single function.

**Files:**
- Create: `src/agent_retrieval/generator/fixed_pool.py`
- Test: `tests/test_fixed_pool.py`

- [ ] **Step 1: Write failing tests.**

Create `tests/test_fixed_pool.py`:

```python
from agent_retrieval.generator.fixed_pool import sample_fixed_pool


class TestSampleFixedPool:
    def test_samples_n_items(self):
        pool = [{"inserted_text": f"item_{i}"} for i in range(16)]
        sampled = sample_fixed_pool(pool, n=4, parametrisation_id="test__a__n4")
        assert len(sampled) == 4

    def test_deterministic_for_same_parametrisation_id(self):
        pool = [{"inserted_text": f"item_{i}"} for i in range(16)]
        s1 = sample_fixed_pool(pool, n=8, parametrisation_id="test__a__n8")
        s2 = sample_fixed_pool(pool, n=8, parametrisation_id="test__a__n8")
        assert s1 == s2

    def test_different_pid_gives_different_sample(self):
        pool = [{"inserted_text": f"item_{i}"} for i in range(16)]
        s1 = sample_fixed_pool(pool, n=8, parametrisation_id="test__a__n8")
        s2 = sample_fixed_pool(pool, n=8, parametrisation_id="test__b__n8")
        # Different IDs should very likely give different orderings.
        assert s1 != s2

    def test_n_equal_to_pool_returns_full_pool(self):
        pool = [{"inserted_text": f"item_{i}"} for i in range(16)]
        sampled = sample_fixed_pool(pool, n=16, parametrisation_id="test__a__n16")
        # Same items, possibly reordered.
        assert sorted(s["inserted_text"] for s in sampled) == sorted(p["inserted_text"] for p in pool)

    def test_n_greater_than_pool_raises(self):
        pool = [{"inserted_text": f"item_{i}"} for i in range(16)]
        import pytest
        with pytest.raises(ValueError, match="n=20 exceeds pool size 16"):
            sample_fixed_pool(pool, n=20, parametrisation_id="test__a__n20")
```

- [ ] **Step 2: Run tests to verify they fail.**

```bash
poetry run pytest tests/test_fixed_pool.py -v
```

Expected: FAIL — module doesn't exist.

- [ ] **Step 3: Implement the helper.**

Create `src/agent_retrieval/generator/fixed_pool.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass.**

```bash
poetry run pytest tests/test_fixed_pool.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit.**

```bash
git add src/agent_retrieval/generator/fixed_pool.py tests/test_fixed_pool.py
git commit -m "$(cat <<'EOF'
generator: add fixed-pool sampling helper

Deterministically samples n items from a hand-authored pool, seeded by
parametrisation_id. Used by multi_retrieval and pure_reasoning to pick
a subset of the 16-item pool per cell.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Generator — `multi_retrieval` insertion path

**Why:** For `multi_retrieval`, the insertion agent must use pre-authored needles from the fixed pool, not invent its own. Build a separate insertion entry point that hands the agent the specific needles to insert + writes the answer key with the pre-known `value` field.

**Files:**
- Create: `src/agent_retrieval/generator/insertion_fixed.py`
- Test: `tests/test_insertion_fixed.py`

- [ ] **Step 1: Write failing test for the prompt-building helper.**

Create `tests/test_insertion_fixed.py`:

```python
from pathlib import Path

import pytest

from agent_retrieval.generator.insertion_fixed import (
    build_fixed_insertion_prompt,
    write_fixed_pool_answer_key,
)
from agent_retrieval.schema.answer_key import AnswerKey
from agent_retrieval.schema.template import ExperimentTemplate, Parametrisation


@pytest.fixture
def sample_template() -> ExperimentTemplate:
    return ExperimentTemplate.model_validate({
        "experiment_type": "multi_retrieval",
        "payload": {"item_type": "fact"},
        "question_examples": {
            "python_repo": {
                "hard_contextual": {
                    "question": "Find the {n} canary deployment parameters.",
                    "answer": "List of {n} parameters",
                },
            },
        },
        "rubric_criteria": [
            {"criterion": "recall", "weight": 1.0},
            {"criterion": "precision", "weight": 0.3},
        ],
        "grid": {
            "content_profile": ["python_repo"],
            "corpus_token_count": [800000],
            "discriminability": ["hard"],
            "reference_clarity": ["contextual"],
            "n_items": [2],
        },
        "fixed_pool": {
            "python_repo": [
                {"inserted_text": "canary_traffic_split = 5", "value": "5",
                 "content_hint": "canary traffic split percentage"},
                {"inserted_text": "evaluation_window_hours = 6", "value": "6",
                 "content_hint": "canary evaluation window"},
            ],
        },
    })


class TestBuildFixedInsertionPrompt:
    def test_prompt_lists_each_selected_needle(self, sample_template):
        param = Parametrisation(
            experiment_type="multi_retrieval",
            content_profile="python_repo",
            corpus_token_count=800000,
            discriminability="hard",
            reference_clarity="contextual",
            n_items=2,
        )
        selected = sample_template.fixed_pool["python_repo"]  # both items
        prompt = build_fixed_insertion_prompt(
            template=sample_template,
            parametrisation=param,
            selected_items=selected,
            target_files_content="### File: foo.py\n```\npass\n```",
            answer_key_path=Path("/tmp/ak.yaml"),
        )
        assert "canary_traffic_split = 5" in prompt
        assert "evaluation_window_hours = 6" in prompt

    def test_prompt_forbids_inventing_new_needles(self, sample_template):
        param = Parametrisation(
            experiment_type="multi_retrieval",
            content_profile="python_repo",
            corpus_token_count=800000,
            discriminability="hard",
            reference_clarity="contextual",
            n_items=2,
        )
        selected = sample_template.fixed_pool["python_repo"]
        prompt = build_fixed_insertion_prompt(
            template=sample_template,
            parametrisation=param,
            selected_items=selected,
            target_files_content="...",
            answer_key_path=Path("/tmp/ak.yaml"),
        )
        # Must explicitly forbid invention.
        assert "do not invent" in prompt.lower() or "do not modify" in prompt.lower()


class TestWriteFixedPoolAnswerKey:
    def test_writes_valid_answer_key_with_values(self, sample_template, tmp_path):
        param = Parametrisation(
            experiment_type="multi_retrieval",
            content_profile="python_repo",
            corpus_token_count=800000,
            discriminability="hard",
            reference_clarity="contextual",
            n_items=2,
        )
        selected = sample_template.fixed_pool["python_repo"]
        # Simulate the insertion agent having written file_path/line_range.
        items_with_locations = [
            {**selected[0], "file_path": "deploy/canary.py", "line_range": [10, 10]},
            {**selected[1], "file_path": "deploy/canary.py", "line_range": [20, 20]},
        ]
        ak_path = tmp_path / "ak.yaml"
        write_fixed_pool_answer_key(
            template=sample_template,
            parametrisation=param,
            items_with_locations=items_with_locations,
            answer_key_path=ak_path,
        )
        ak = AnswerKey.from_yaml(ak_path)
        assert ak.parametrisation_id == "multi_retrieval__python_repo__800k__hard__contextual__n2"
        assert len(ak.items) == 2
        assert ak.items[0].value == "5"
        assert "{n}" not in ak.expected_answers.question  # n was substituted
        assert "2" in ak.expected_answers.question
```

- [ ] **Step 2: Run tests to verify they fail.**

```bash
poetry run pytest tests/test_insertion_fixed.py -v
```

Expected: FAIL — module doesn't exist.

- [ ] **Step 3: Implement `insertion_fixed.py`.**

Create `src/agent_retrieval/generator/insertion_fixed.py`:

```python
from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    ToolUseBlock,
    query,
)

from agent_retrieval.generator.fixed_pool import sample_fixed_pool
from agent_retrieval.generator.insertion import (
    InsertionStats,
    _read_target_fragments,
    _select_target_files,
)
from agent_retrieval.schema.experiment import PAYLOAD_INSERTION_MODEL_MULTI
from agent_retrieval.schema.template import ExperimentTemplate, Parametrisation


def build_fixed_insertion_prompt(
    template: ExperimentTemplate,
    parametrisation: Parametrisation,
    selected_items: list[dict[str, Any]],
    target_files_content: str,
    answer_key_path: Path,
) -> str:
    """Build a system prompt for inserting pre-authored needles.

    Differs from build_insertion_prompt in that the needles are fixed —
    the agent's only job is to find good insertion sites and report
    file_path + line_range per item.
    """
    n_items = parametrisation.n_items or len(selected_items)
    needles_block = "\n".join(
        f"{i + 1}. inserted_text: {item['inserted_text']!r}\n"
        f"   value: {item['value']!r}\n"
        f"   content_hint: {item.get('content_hint', '')!r}"
        for i, item in enumerate(selected_items)
    )

    return (
        f"You are inserting {n_items} pre-authored needles into a corpus for a "
        f"retrieval experiment. The needles are FIXED — do not invent new ones, "
        f"do not modify their text, do not paraphrase. Insert each verbatim.\n\n"
        f"Experiment type: multi_retrieval\n"
        f"Number of items: {n_items}\n\n"
        f"## Needles to insert (verbatim, no modifications)\n{needles_block}\n\n"
        f"## Target fragments (pre-selected — use these)\n{target_files_content}\n\n"
        f"## Instructions\n"
        f"Each fragment above is a 30-line window from a corpus file, with its "
        f"start_line offset. Do NOT read or browse any files. Work only with the "
        f"fragments provided above.\n\n"
        f"1. For each of the {n_items} needles, choose a target fragment and an "
        f"insertion line within that fragment where the needle reads naturally.\n"
        f"2. Use the Edit tool to insert each needle VERBATIM (same text, same "
        f"capitalisation, same operators, same spacing).\n"
        f"3. Write the answer key YAML to {answer_key_path.resolve()} using the "
        f"schema below. For each item, fill in file_path, line_range, and "
        f"context_summary; copy inserted_text and value verbatim from the needles "
        f"above.\n\n"
        f"IMPORTANT: Batch ALL Edit and Write tool calls into a single response. "
        f"Do not use multiple turns.\n\n"
        f"## Answer key schema\n"
        f"```yaml\n"
        f"parametrisation_id: \"{parametrisation.parametrisation_id}\"\n"
        f"experiment_type: \"multi_retrieval\"\n"
        f"items:\n"
        f"  - item_id: \"target_001\"\n"
        f"    inserted_text: \"<verbatim from needle 1>\"\n"
        f"    value: \"<verbatim from needle 1>\"\n"
        f"    file_path: \"<relative path>\"\n"
        f"    line_range: [<start>, <end>]\n"
        f"    context_summary: \"<one sentence>\"\n"
        f"  # ... item_002 through item_{n_items:03d}\n"
        f"```\n"
    )


def write_fixed_pool_answer_key(
    template: ExperimentTemplate,
    parametrisation: Parametrisation,
    items_with_locations: list[dict[str, Any]],
    answer_key_path: Path,
) -> None:
    """Write a complete answer key for a multi_retrieval cell.

    items_with_locations includes the fixed-pool fields (inserted_text, value)
    plus the insertion-site fields (file_path, line_range, context_summary).
    """
    n = len(items_with_locations)
    profile = parametrisation.content_profile
    profile_examples = template.question_examples.get(profile, {})
    example = profile_examples.get("hard_contextual")
    if example is None:
        raise ValueError(f"No 'hard_contextual' example for profile {profile}")
    question = example.question.replace("{n}", str(n))

    items_yaml = []
    for i, it in enumerate(items_with_locations, start=1):
        items_yaml.append({
            "item_id": f"target_{i:03d}",
            "inserted_text": it["inserted_text"],
            "value": it["value"],
            "file_path": it.get("file_path"),
            "line_range": it.get("line_range"),
            "context_summary": it.get("context_summary", ""),
        })

    parameters: dict[str, Any] = {"content_profile": profile}
    if parametrisation.corpus_token_count is not None:
        parameters["corpus_token_count"] = parametrisation.corpus_token_count
    if parametrisation.discriminability is not None:
        parameters["discriminability"] = parametrisation.discriminability
    if parametrisation.reference_clarity is not None:
        parameters["reference_clarity"] = parametrisation.reference_clarity
    if parametrisation.n_items is not None:
        parameters["n_items"] = parametrisation.n_items

    rubric_yaml = [
        {"criterion": c.criterion, "weight": c.weight}
        for c in template.rubric_criteria
    ]

    ak_dict = {
        "parametrisation_id": parametrisation.parametrisation_id,
        "experiment_type": parametrisation.experiment_type,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "parameters": parameters,
        "items": items_yaml,
        "expected_answers": {
            "question": question,
            "correctness": (
                f"Each of the {n} items must be reported with both the verbatim "
                f"inserted_text and the corresponding value, matching the "
                f"answer-key items."
            ),
            "completeness": (
                f"All {n} items must be reported. Recall is fraction of items "
                f"correctly reported."
            ),
        },
        "rubric_criteria": rubric_yaml,
    }

    answer_key_path.parent.mkdir(parents=True, exist_ok=True)
    with open(answer_key_path, "w") as f:
        yaml.dump(ak_dict, f, default_flow_style=False, sort_keys=False)


async def insert_fixed_payloads(
    template: ExperimentTemplate,
    parametrisation: Parametrisation,
    corpus_dir: Path,
    answer_key_path: Path,
) -> InsertionStats | None:
    """Insertion driver for multi_retrieval: use fixed pool, not LLM-invented needles."""
    if answer_key_path.exists():
        return None

    answer_key_path.parent.mkdir(parents=True, exist_ok=True)

    pool = template.fixed_pool.get(parametrisation.content_profile, [])
    if not pool:
        raise ValueError(
            f"fixed_pool is empty for profile {parametrisation.content_profile}"
        )
    n_items = parametrisation.n_items or 1
    selected = sample_fixed_pool(pool, n=n_items, parametrisation_id=parametrisation.parametrisation_id)

    n_target_files = max(n_items * 2, 4)
    target_files = _select_target_files(corpus_dir, parametrisation, n_target_files)
    base_seed = hash(parametrisation.parametrisation_id) ^ 0xDEAD
    target_files_content = _read_target_fragments(target_files, corpus_dir, base_seed)

    system_prompt = build_fixed_insertion_prompt(
        template=template,
        parametrisation=parametrisation,
        selected_items=selected,
        target_files_content=target_files_content,
        answer_key_path=answer_key_path,
    )

    options = ClaudeAgentOptions(
        model=PAYLOAD_INSERTION_MODEL_MULTI,
        system_prompt=system_prompt,
        cwd=str(corpus_dir.resolve()),
        allowed_tools=["Edit", "Write"],
        permission_mode="acceptEdits",
        max_turns=max(n_items * 3, 10),
    )

    prompt = (
        "Insert the pre-authored needles into the provided fragments and write "
        "the answer key. Do not invent new needles; do not modify the provided "
        "needle text. Batch all Edit and Write calls into a single response."
    )

    stats = InsertionStats(model=PAYLOAD_INSERTION_MODEL_MULTI)

    async for message in query(prompt=prompt, options=options):
        if isinstance(message, AssistantMessage):
            if message.usage:
                stats.input_tokens += message.usage.get("input_tokens", 0)
                stats.output_tokens += message.usage.get("output_tokens", 0)
            for block in message.content:
                if isinstance(block, ToolUseBlock):
                    stats.tool_calls.append(block.name)
        elif isinstance(message, ResultMessage):
            stats.num_turns = message.num_turns
            stats.duration_ms = message.duration_ms
            stats.total_cost_usd = message.total_cost_usd or 0.0
            stats.is_error = message.is_error
            stats.errors = message.errors or []
            break

    stats.answer_key_written = answer_key_path.exists()
    if not stats.answer_key_written and stats.tool_calls:
        shutil.rmtree(corpus_dir, ignore_errors=True)
        stats.errors.append("rolled back corpus: edits applied but no answer key")

    return stats
```

- [ ] **Step 4: Run tests to verify they pass.**

```bash
poetry run pytest tests/test_insertion_fixed.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit.**

```bash
git add src/agent_retrieval/generator/insertion_fixed.py tests/test_insertion_fixed.py
git commit -m "$(cat <<'EOF'
generator: insertion path for multi_retrieval (fixed-pool needles)

Builds a system prompt that hands the insertion agent specific
pre-authored needles to insert verbatim, plus a writer that emits a
complete answer key with the pre-known value field.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Generator — `pure_reasoning` answer-key writer

**Why:** `pure_reasoning` skips corpus generation entirely. Generation is reduced to: sample N facts → format the question with the facts inlined → write the answer key. No LLM calls at generation time.

**Files:**
- Create: `src/agent_retrieval/generator/pure_reasoning_gen.py`
- Test: `tests/test_pure_reasoning_gen.py`

- [ ] **Step 1: Write failing test.**

Create `tests/test_pure_reasoning_gen.py`:

```python
from pathlib import Path

import pytest

from agent_retrieval.generator.pure_reasoning_gen import generate_pure_reasoning_cell
from agent_retrieval.schema.answer_key import AnswerKey
from agent_retrieval.schema.template import ExperimentTemplate, Parametrisation


@pytest.fixture
def sample_template() -> ExperimentTemplate:
    return ExperimentTemplate.model_validate({
        "experiment_type": "pure_reasoning",
        "payload": {"item_type": "fact"},
        "question_examples": {
            "python_repo": {
                "hard_contextual": {
                    "question": (
                        "You have {n} parameters. Below are the facts:\n\n"
                        "{facts_block}\n\n"
                        "Derive the narrowest safe-migration window."
                    ),
                    "answer": "Window endpoints + citations.",
                },
            },
        },
        "rubric_criteria": [
            {"criterion": "endpoint_correctness", "weight": 1.0},
            {"criterion": "classification_accuracy", "weight": 0.5},
        ],
        "grid": {
            "content_profile": ["python_repo"],
            "n_items": [2],
        },
        "fixed_pool": {
            "python_repo": [
                {"text": "REPLICATION_LAG_RECOVERY_S = 300",
                 "bound_direction": "lower", "bound_value": "300",
                 "context_summary": "must wait 300s after replication peak"},
                {"text": "BACKUP_WINDOW_OPEN_S = 7200",
                 "bound_direction": "upper", "bound_value": "7200",
                 "context_summary": "must complete before next backup at 7200s"},
            ],
        },
    })


class TestGeneratePureReasoningCell:
    def test_writes_answer_key_with_facts_inlined_in_question(self, sample_template, tmp_path):
        param = Parametrisation(
            experiment_type="pure_reasoning",
            content_profile="python_repo",
            n_items=2,
        )
        ak_path = tmp_path / "ak.yaml"
        generate_pure_reasoning_cell(
            template=sample_template,
            parametrisation=param,
            answer_key_path=ak_path,
        )
        ak = AnswerKey.from_yaml(ak_path)
        assert ak.parametrisation_id == "pure_reasoning__python_repo__n2"
        assert len(ak.items) == 2
        # Facts must be inlined into the question prompt.
        assert "REPLICATION_LAG_RECOVERY_S = 300" in ak.expected_answers.question
        assert "BACKUP_WINDOW_OPEN_S = 7200" in ak.expected_answers.question
        # Each item carries its bound metadata.
        directions = sorted(it.bound_direction for it in ak.items)
        assert directions == ["lower", "upper"]

    def test_idempotent_on_existing_answer_key(self, sample_template, tmp_path):
        """Running twice with skip_existing should not overwrite."""
        param = Parametrisation(
            experiment_type="pure_reasoning",
            content_profile="python_repo",
            n_items=2,
        )
        ak_path = tmp_path / "ak.yaml"
        generate_pure_reasoning_cell(
            template=sample_template,
            parametrisation=param,
            answer_key_path=ak_path,
        )
        first = ak_path.read_text()
        generate_pure_reasoning_cell(
            template=sample_template,
            parametrisation=param,
            answer_key_path=ak_path,
        )
        assert ak_path.read_text() == first  # unchanged
```

- [ ] **Step 2: Run tests to verify they fail.**

```bash
poetry run pytest tests/test_pure_reasoning_gen.py -v
```

Expected: FAIL — module doesn't exist.

- [ ] **Step 3: Implement `pure_reasoning_gen.py`.**

Create `src/agent_retrieval/generator/pure_reasoning_gen.py`:

```python
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from agent_retrieval.generator.fixed_pool import sample_fixed_pool
from agent_retrieval.schema.template import ExperimentTemplate, Parametrisation


def _format_facts_block(items: list[dict[str, Any]]) -> str:
    """Format selected facts as a numbered list for inclusion in the prompt."""
    return "\n".join(f"{i}. {item['text']}" for i, item in enumerate(items, start=1))


def _compute_expected_endpoints(items: list[dict[str, Any]]) -> tuple[str, str, str, str]:
    """Compute (max_lower_value, max_lower_text, min_upper_value, min_upper_text).

    bound_value is treated as a string here. Numeric comparison is left to the
    judge; the answer key records the chosen items so the judge can verify.
    """
    lowers = [it for it in items if it["bound_direction"] == "lower"]
    uppers = [it for it in items if it["bound_direction"] == "upper"]
    if not lowers or not uppers:
        return "", "", "", ""
    max_lower = max(lowers, key=lambda it: float(it["bound_value"]))
    min_upper = min(uppers, key=lambda it: float(it["bound_value"]))
    return (
        max_lower["bound_value"],
        max_lower["text"],
        min_upper["bound_value"],
        min_upper["text"],
    )


def generate_pure_reasoning_cell(
    template: ExperimentTemplate,
    parametrisation: Parametrisation,
    answer_key_path: Path,
) -> None:
    """Generate one pure_reasoning answer key. No corpus, no LLM calls."""
    if answer_key_path.exists():
        return

    profile = parametrisation.content_profile
    pool = template.fixed_pool.get(profile, [])
    if not pool:
        raise ValueError(f"fixed_pool is empty for profile {profile}")

    n = parametrisation.n_items or 1
    selected = sample_fixed_pool(pool, n=n, parametrisation_id=parametrisation.parametrisation_id)

    profile_examples = template.question_examples.get(profile, {})
    example = profile_examples.get("hard_contextual")
    if example is None:
        raise ValueError(f"No 'hard_contextual' example for profile {profile}")

    facts_block = _format_facts_block(selected)
    question = example.question.replace("{n}", str(n)).replace("{facts_block}", facts_block)

    items_yaml = []
    for i, it in enumerate(selected, start=1):
        items_yaml.append({
            "item_id": f"target_{i:03d}",
            "inserted_text": it["text"],
            "value": it["bound_value"],
            "bound_direction": it["bound_direction"],
            "context_summary": it.get("context_summary", ""),
        })

    max_lower_val, max_lower_text, min_upper_val, min_upper_text = _compute_expected_endpoints(selected)

    rubric_yaml = [
        {"criterion": c.criterion, "weight": c.weight}
        for c in template.rubric_criteria
    ]

    ak_dict = {
        "parametrisation_id": parametrisation.parametrisation_id,
        "experiment_type": "pure_reasoning",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "parameters": {"content_profile": profile, "n_items": n},
        "items": items_yaml,
        "expected_answers": {
            "question": question,
            "correctness": (
                f"The answer must derive the narrowest valid window. The lower "
                f"endpoint is established by the fact whose bound_direction is "
                f"'lower' and bound_value is largest: {max_lower_text!r} "
                f"(value: {max_lower_val}). The upper endpoint is established "
                f"by the fact whose bound_direction is 'upper' and bound_value "
                f"is smallest: {min_upper_text!r} (value: {min_upper_val}). "
                f"The agent must cite both facts."
            ),
            "completeness": (
                f"The agent must classify each of the {n} facts as a lower or "
                f"upper bound consistent with the answer-key bound_direction "
                f"field, OR justify any deviation in the reasoning."
            ),
        },
        "rubric_criteria": rubric_yaml,
    }

    answer_key_path.parent.mkdir(parents=True, exist_ok=True)
    with open(answer_key_path, "w") as f:
        yaml.dump(ak_dict, f, default_flow_style=False, sort_keys=False)
```

- [ ] **Step 4: Run tests to verify they pass.**

```bash
poetry run pytest tests/test_pure_reasoning_gen.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit.**

```bash
git add src/agent_retrieval/generator/pure_reasoning_gen.py tests/test_pure_reasoning_gen.py
git commit -m "$(cat <<'EOF'
generator: pure_reasoning cell generator (no corpus, no LLM)

Samples N facts from the fixed pool, inlines them into the question
template, and writes the answer key with bound metadata. Generation
is fully deterministic — no LLM calls.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Generator — dispatch in `generate_experiment_v2`

**Why:** The top-level generator currently runs the same pipeline (pool → corpus → insertion) for all types. Branch by experiment type so:
- `multi_retrieval` uses `insert_fixed_payloads` (Task 6) instead of the LLM-invented insertion path
- `pure_reasoning` skips corpus generation entirely and calls `generate_pure_reasoning_cell` (Task 7)

**Files:**
- Modify: `src/agent_retrieval/generator/generate.py`
- Test: `tests/test_generate_dispatch.py` (new)

- [ ] **Step 1: Write failing test using mocked SDK calls.**

Create `tests/test_generate_dispatch.py`:

```python
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from agent_retrieval.generator.generate import generate_experiment_v2
from agent_retrieval.schema.template import ExperimentTemplate


@pytest.fixture
def pure_reasoning_template_yaml(tmp_path: Path) -> Path:
    template = {
        "experiment_type": "pure_reasoning",
        "payload": {"item_type": "fact"},
        "question_examples": {
            "python_repo": {
                "hard_contextual": {
                    "question": "You have {n} facts: {facts_block}. Derive window.",
                    "answer": "answer",
                },
            },
        },
        "rubric_criteria": [
            {"criterion": "endpoint_correctness", "weight": 1.0},
            {"criterion": "classification_accuracy", "weight": 0.5},
        ],
        "grid": {
            "content_profile": ["python_repo"],
            "n_items": [2],
        },
        "fixed_pool": {
            "python_repo": [
                {"text": "A", "bound_direction": "lower", "bound_value": "100"},
                {"text": "B", "bound_direction": "upper", "bound_value": "500"},
            ],
        },
    }
    p = tmp_path / "pure_reasoning.yaml"
    p.write_text(yaml.dump(template))
    return p


class TestGenerateDispatchPureReasoning:
    @pytest.mark.asyncio
    async def test_pure_reasoning_writes_answer_key_without_corpus(
        self, pure_reasoning_template_yaml, tmp_workspace
    ):
        template = ExperimentTemplate.from_yaml(pure_reasoning_template_yaml)
        workspace = tmp_workspace / "workspace"

        # No SDK call should occur for pure_reasoning generation.
        with patch("agent_retrieval.generator.generate.generate_pool") as mock_pool, \
             patch("agent_retrieval.generator.generate.assemble_corpus") as mock_assemble, \
             patch("agent_retrieval.generator.generate.insert_payloads") as mock_insert:
            generated = await generate_experiment_v2(template, workspace)

        assert mock_pool.call_count == 0
        assert mock_assemble.call_count == 0
        assert mock_insert.call_count == 0

        ak_path = workspace / "judge" / "answer_keys" / "pure_reasoning__python_repo__n2.yaml"
        assert ak_path.exists()
        assert generated == ["pure_reasoning__python_repo__n2"]
```

- [ ] **Step 2: Run test to verify it fails.**

```bash
poetry run pytest tests/test_generate_dispatch.py -v
```

Expected: FAIL — the existing pipeline tries to generate a pool.

- [ ] **Step 3: Branch the dispatch.**

In `src/agent_retrieval/generator/generate.py`, replace the existing `generate_experiment_v2` with:

```python
from __future__ import annotations

from pathlib import Path

from agent_retrieval.generator.assembler import assemble_corpus
from agent_retrieval.generator.corpus_files import iter_corpus_files
from agent_retrieval.generator.grid import expand_grid
from agent_retrieval.generator.insertion import insert_payloads
from agent_retrieval.generator.insertion_fixed import insert_fixed_payloads
from agent_retrieval.generator.pool import generate_pool
from agent_retrieval.generator.pure_reasoning_gen import generate_pure_reasoning_cell
from agent_retrieval.schema.template import ExperimentTemplate


async def generate_experiment_v2(
    template: ExperimentTemplate,
    workspace_dir: Path,
    skip_existing: bool = True,
) -> list[str]:
    parametrisations = expand_grid(template)
    generated_ids: list[str] = []

    if template.experiment_type == "pure_reasoning":
        # No corpus, no insertion agent — just write answer keys.
        for param in parametrisations:
            pid = param.parametrisation_id
            answer_key_path = workspace_dir / "judge" / "answer_keys" / f"{pid}.yaml"
            if skip_existing and answer_key_path.exists():
                print(f"  Skipping {pid} (already exists)")
                continue
            generate_pure_reasoning_cell(
                template=template,
                parametrisation=param,
                answer_key_path=answer_key_path,
            )
            generated_ids.append(pid)
            print(f"  Done: {pid}")
        return generated_ids

    # Corpus-based experiments (single_needle, multi_chain, multi_reasoning, multi_retrieval).
    profiles_needed = {p.content_profile for p in parametrisations}
    for profile_name in profiles_needed:
        pool_dir = workspace_dir / "background_corpora" / profile_name
        if pool_dir.exists() and any(iter_corpus_files(pool_dir)):
            print(f"Background pool for '{profile_name}' already exists, skipping.")
            continue
        print(f"Ensuring background pool for '{profile_name}'...")
        await generate_pool(profile_name, pool_dir)

    for param in parametrisations:
        pid = param.parametrisation_id
        corpus_dir = workspace_dir / "runner" / "corpora" / pid
        answer_key_path = workspace_dir / "judge" / "answer_keys" / f"{pid}.yaml"

        if skip_existing and corpus_dir.exists() and answer_key_path.exists():
            print(f"  Skipping {pid} (already exists)")
            continue

        pool_dir = workspace_dir / "background_corpora" / param.content_profile
        print(f"  Assembling corpus for {pid}...")
        assemble_corpus(pool_dir, corpus_dir, param)

        print(f"  Inserting payloads for {pid}...")
        if template.experiment_type == "multi_retrieval":
            await insert_fixed_payloads(template, param, corpus_dir, answer_key_path)
        else:
            await insert_payloads(template, param, corpus_dir, answer_key_path)

        generated_ids.append(pid)
        print(f"  Done: {pid}")

    return generated_ids
```

- [ ] **Step 4: Run test to verify it passes.**

```bash
poetry run pytest tests/test_generate_dispatch.py -v
```

Expected: PASS.

- [ ] **Step 5: Run full test suite to verify no regressions.**

```bash
poetry run pytest -v
```

Expected: all PASS.

- [ ] **Step 6: Commit.**

```bash
git add src/agent_retrieval/generator/generate.py tests/test_generate_dispatch.py
git commit -m "$(cat <<'EOF'
generator: dispatch by experiment_type in generate_experiment_v2

multi_retrieval routes to the fixed-pool insertion path; pure_reasoning
skips corpus generation entirely. Existing types behave unchanged.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Runner — `pure_reasoning` skips corpus loading

**Why:** For `pure_reasoning`, the answer key's `expected_answers.question` already contains the full prompt (question + facts inlined). The runner needs to skip the corpus_dir step and not require a working directory of files.

**Files:**
- Modify: `src/agent_retrieval/runner/run.py`
- Test: `tests/test_runner_wiring.py` (extend)

- [ ] **Step 1: Inspect existing runner-wiring test to follow its mocking style.**

```bash
poetry run cat tests/test_runner_wiring.py | head -80
```

(This is read-only — no changes yet. Look at how `run_agent_session` is mocked.)

- [ ] **Step 2: Write failing test for pure_reasoning runner path.**

Append to `tests/test_runner_wiring.py` (or follow the existing test style — adapt the snippet below to match):

```python
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import yaml

from agent_retrieval.runner.run import run_batch
from agent_retrieval.runner.session import AgentResult
from agent_retrieval.schema.batch import BatchConfig


class TestRunnerPureReasoning:
    @pytest.mark.asyncio
    async def test_pure_reasoning_run_passes_question_without_corpus_dir_files(
        self, tmp_workspace
    ):
        """Pure-reasoning runs should send the question as the prompt and not
        need a populated corpus_dir."""
        workspace = tmp_workspace / "workspace"
        ak_path = workspace / "judge" / "answer_keys" / "pure_reasoning__python_repo__n2.yaml"
        ak_path.write_text(yaml.dump({
            "parametrisation_id": "pure_reasoning__python_repo__n2",
            "experiment_type": "pure_reasoning",
            "items": [{
                "item_id": "target_001",
                "inserted_text": "A",
                "context_summary": "x",
                "value": "100",
                "bound_direction": "lower",
            }],
            "expected_answers": {
                "question": "Inlined facts: 1. A. 2. B. Derive window.",
                "correctness": "ok",
                "completeness": "ok",
            },
            "rubric_criteria": [
                {"criterion": "endpoint_correctness", "weight": 1.0},
            ],
        }))

        batch = BatchConfig.model_validate({
            "batch_name": "test_pure",
            "experiments": [{
                "experiment_type": "pure_reasoning",
                "grid": {
                    "content_profile": ["python_repo"],
                    "n_items": [2],
                },
            }],
            "n_repeats": 1,
            "agent_model": "claude-haiku-4-5-20251001",
            "max_parallel": 1,
            "max_turns": 5,
            "allowed_tools": [],
        })

        captured_kwargs: dict = {}

        async def fake_run_session(**kwargs):
            captured_kwargs.update(kwargs)
            return AgentResult(
                response_text="window = [100, 500]",
                session_id="x", num_turns=1, total_cost_usd=0.0, usage={},
            )

        with patch(
            "agent_retrieval.runner.run.run_agent_session",
            side_effect=fake_run_session,
        ):
            await run_batch(batch, experiments_dir=tmp_workspace / "experiments",
                            workspace_dir=workspace)

        # The corpus_dir for pure_reasoning may be a non-existent path; the
        # session function should still be called with the question prompt.
        assert "Inlined facts" in captured_kwargs["question"]
```

(If your `BatchConfig` requires extra fields, adjust the construction.)

- [ ] **Step 3: Run test to verify it fails.**

```bash
poetry run pytest tests/test_runner_wiring.py -v -k pure_reasoning
```

Expected: FAIL — corpus_dir doesn't exist or run is rejected.

- [ ] **Step 4: Adjust runner to skip corpus_dir requirement for pure_reasoning.**

In `src/agent_retrieval/runner/run.py`, in the `run_one` closure, branch on the experiment type derived from the parametrisation ID. Replace the inner block:

```python
async def run_one(pid: str, run_id: str, run_dir: Path) -> None:
    nonlocal completed
    async with semaphore:
        corpus_dir = corpora_dir / pid
        ak_path = answer_keys_dir / f"{pid}.yaml"

        try:
            with open(ak_path) as f:
                ak = yaml.safe_load(f)
            question = ak["expected_answers"]["question"]
        except Exception as e:
            state_mgr.update_status(run_dir, "failed", error_message=f"bad answer key: {e}")
            completed += 1
            print(f"[{completed}/{total}] FAILED {pid} run {run_id}: bad answer key")
            return

        # pure_reasoning has no corpus; use a placeholder cwd for the SDK.
        is_pure_reasoning = pid.startswith("pure_reasoning__")
        session_corpus_dir = run_dir if is_pure_reasoning else corpus_dir

        state_mgr.update_status(run_dir, "running", started_at=datetime.now(timezone.utc).isoformat())

        try:
            result = await run_agent_session(
                question=question, corpus_dir=session_corpus_dir, model=batch.agent_model,
                allowed_tools=batch.allowed_tools, max_turns=batch.max_turns,
                run_id=run_id, run_dir=run_dir, effort_mode=batch.effort_mode,
            )
            response_path = run_dir / "response.json"
            response_path.write_text(json.dumps({
                "response_text": result.response_text,
                "session_id": result.session_id,
                "num_turns": result.num_turns,
                "total_cost_usd": result.total_cost_usd,
                "usage": result.usage,
            }, indent=2))
            state_mgr.update_status(run_dir, "completed", completed_at=datetime.now(timezone.utc).isoformat())
            completed += 1
            print(f"[{completed}/{total}] Completed {pid} run {run_id}")
        except Exception as e:
            state_mgr.update_status(run_dir, "failed", error_message=str(e))
            completed += 1
            print(f"[{completed}/{total}] FAILED {pid} run {run_id}: {e}")
```

The change is: detect `pure_reasoning__` prefix, use `run_dir` as the cwd (which exists, since the state manager created it). The agent never needs to actually read files — the question contains the full prompt.

- [ ] **Step 5: Run the test to verify it passes.**

```bash
poetry run pytest tests/test_runner_wiring.py -v -k pure_reasoning
```

Expected: PASS.

- [ ] **Step 6: Run full test suite to verify no regressions.**

```bash
poetry run pytest -v
```

Expected: all PASS.

- [ ] **Step 7: Commit.**

```bash
git add src/agent_retrieval/runner/run.py tests/test_runner_wiring.py
git commit -m "$(cat <<'EOF'
runner: pure_reasoning runs use run_dir as cwd (no corpus required)

For pure_reasoning experiments the answer key's question already
contains the full prompt with N facts inlined. The runner skips the
corpus directory and uses the run directory as a stand-in cwd.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: Author the `multi_retrieval` experiment YAML (16 items per profile)

**Why:** The fixed pools must be hand-authored to satisfy the constraints in the spec. The user reviews each pool before commit.

**Files:**
- Create: `experiments/multi_retrieval.yaml`

- [ ] **Step 1: Draft the YAML scaffold and the canary-deployment pool for python_repo.**

Authoring constraints (from spec):
- 16 numeric tuning parameters governing canary deployment / progressive rollout.
- Item names and locations must vary — no single grep token unifies them.
- Each item has `inserted_text`, `value`, `content_hint`.
- Plausibly insertable into the existing python_repo file structure.

Seed items (write at least these 4 verbatim, then draft 12 more following the same pattern):

```yaml
experiment_type: "multi_retrieval"

payload:
  item_type: "fact"

question_examples:
  python_repo:
    hard_contextual:
      question: |
        I have added N={n} numeric tuning parameters to this codebase that govern a canary deployment / progressive rollout system. Find each one and report it as `<verbatim line of code> — value: <numeric value>`. Report all {n}.
      answer: "List of {n} canary parameters with values"
  noir_fiction:
    hard_contextual:
      question: |
        I have added N={n} passages to this story that fix a specific time on the night before the murder. Find each one and report it as `<verbatim passage text> — time referenced: <e.g., 9:45 PM>`. Report all {n}.
      answer: "List of {n} timing passages with times"

rubric_criteria:
  - criterion: "recall"
    weight: 1.0
  - criterion: "precision"
    weight: 0.3

grid:
  content_profile: [python_repo, noir_fiction]
  corpus_token_count: [800000]
  discriminability: [hard]
  reference_clarity: [contextual]
  n_items: [2, 4, 8, 12, 16]

fixed_pool:
  python_repo:
    - inserted_text: "canary_traffic_split = 5  # initial percent of traffic routed to new version"
      value: "5"
      content_hint: "deploy/canary.py — initial canary traffic percentage"
    - inserted_text: "rollout_step_increment = 10  # percent points added per healthy interval"
      value: "10"
      content_hint: "deploy/canary.py — incremental traffic ramp step"
    - inserted_text: "regression_p99_multiplier = 1.5  # latency vs control to flag regression"
      value: "1.5"
      content_hint: "deploy/canary.py — p99 latency regression threshold"
    - inserted_text: "abort_error_rate = 0.02  # error rate threshold that triggers abort"
      value: "0.02"
      content_hint: "deploy/canary.py — auto-abort error rate ceiling"
    # ↓ 12 more entries to be added in Step 2 below.
  noir_fiction:
    - inserted_text: "The streetcar's last bell drifted up from the avenue at half-past ten, the night before the murder, sharp as a coin dropped on stone."
      value: "10:30 PM"
      content_hint: "ambient timing detail in early chapter"
    - inserted_text: "Mickey had stopped by the deli a little after nine the previous evening; the till receipt, dated and stamped, was still pinned to the cork board."
      value: "9:00 PM (night before)"
      content_hint: "physical evidence in case-file chapter"
    - inserted_text: "She remembered the radio playing the closing theme of the Westinghouse hour — that placed it sometime around eleven, the night before he died."
      value: "11:00 PM (night before)"
      content_hint: "witness recollection in mid-novel chapter"
    - inserted_text: "The chef closed the back door at quarter past midnight on the night the tabby got out — the night before the murder, by the sergeant's reckoning."
      value: "12:15 AM (night before, technically early morning)"
      content_hint: "minor character anecdote, mid-novel"
    # ↓ 12 more entries to be added in Step 2 below.
```

- [ ] **Step 2: Author the remaining 12 python_repo items and 12 noir_fiction items.**

Write the new entries into the YAML, replacing the placeholder comments. Each new python_repo entry must be a numeric configuration parameter related to canary/progressive rollout, with a unique name (no shared grep token across the 16), plausibly fitting alongside existing python_repo files (api/, workers/, deploy/). Suggested topics across the 12: holdback group size (percentage), synthetic-traffic ratio, evaluation window duration, auto-rollback latency threshold, drift tolerance, per-cohort blast radius cap, exposure ramp cooldown, health-check interval, canary→prod copy-over delay, probe success ratio threshold, feature flag exposure increment, stability hold duration.

Each new noir_fiction entry must reference a specific time on the night BEFORE the murder, use varied prose (don't repeat the literal phrase "night before" in every passage — vary by narrator, place, sensory cue), fit naturally into noir prose, and have an extractable time-of-day value. After authoring, both `fixed_pool.python_repo` and `fixed_pool.noir_fiction` should each contain exactly 16 entries.

- [ ] **Step 3: Validate the YAML loads cleanly.**

```bash
poetry run python -c "from agent_retrieval.schema.template import ExperimentTemplate; t = ExperimentTemplate.from_yaml('experiments/multi_retrieval.yaml'); print(len(t.fixed_pool['python_repo']), len(t.fixed_pool['noir_fiction']))"
```

Expected output: `16 16`

- [ ] **Step 4: Request user review.**

Print a confirmation message to the user listing the 16 python_repo items and the 16 noir_fiction items (or paste the YAML's `fixed_pool` block). Wait for user approval before committing.

If the user requests edits, apply them and re-run Step 3.

- [ ] **Step 5: Commit.**

```bash
git add experiments/multi_retrieval.yaml
git commit -m "$(cat <<'EOF'
experiments: add multi_retrieval template with 16-item fixed pools

python_repo: canary deployment / progressive rollout numeric parameters.
noir_fiction: prose passages fixing times on the night before the murder.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: Author the `pure_reasoning` experiment YAML (16 items per profile)

**Why:** Same as Task 10, for the second experiment. Authoring constraints differ.

**Files:**
- Create: `experiments/pure_reasoning.yaml`

- [ ] **Step 1: Draft the YAML scaffold + the seed items.**

Authoring constraints:
- python_repo: 16 numeric parameters, each implying a lower or upper bound on a database-migration window relative to a reference event.
- noir_fiction: 16 prose passages, each implying a lower or upper bound on the suspect being at-large on the night of the murder.
- The 16-item pool must be **contradiction-free**: `max(all 16 lowers) < min(all 16 uppers)` for each profile (proven in the spec to guarantee no subset can ever contradict).
- Roughly half lower-bounds and half upper-bounds per profile.
- Bound direction must be inferrable from the parameter's role / the evidence's language — NOT from its name (so `MIN_*` doesn't reliably mean lower-bound for the migration window).

Seed YAML:

```yaml
experiment_type: "pure_reasoning"

payload:
  item_type: "fact"

question_examples:
  python_repo:
    hard_contextual:
      question: |
        You are running a database migration. It must execute during a window of system quiescence. Below are {n} numeric configuration parameters from across the service. Each implies a constraint on *when* the migration can safely begin and end relative to a reference event (the most recent backup, the next traffic peak, the next certificate rotation, the latest replication lag spike, etc.).

        Some parameters establish a *lower bound* on the migration start time (must wait at least X seconds after the reference event).

        Others establish an *upper bound* on the migration end time (must complete before X seconds after the reference event).

        Below are the {n} facts:

        {facts_block}

        Derive the narrowest safe-migration window [earliest_start, latest_end] consistent with all {n} constraints. For each endpoint, cite the parameter that establishes it. Justify your classification of each parameter's bound direction.
      answer: "Window endpoints with citations"
  noir_fiction:
    hard_contextual:
      question: |
        You are given {n} pieces of evidence from a homicide investigation, each pertaining to events on the night of the murder. Some pieces establish that the suspect was at large (free, unaccounted-for) at or after a specific time; others establish that the suspect was off-the-streets (accounted-for, with an alibi) at or before a specific time.

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
  python_repo:
    - text: "REPLICATION_LAG_RECOVERY_S = 300  # post-traffic-peak settle time before risky writes"
      bound_direction: "lower"
      bound_value: "300"
      context_summary: "must wait at least 300s after the latest traffic-peak replication-lag spike"
    - text: "CERT_ROTATION_GRACE_S = 1800  # window between cert rotation and next cert checks"
      bound_direction: "lower"
      bound_value: "1800"
      context_summary: "must wait until at least 1800s after the most recent certificate rotation"
    - text: "BACKUP_WINDOW_OPEN_S = 7200  # next backup window starts at this offset"
      bound_direction: "upper"
      bound_value: "7200"
      context_summary: "migration must complete before 7200s — when the next backup window opens"
    - text: "PEAK_TRAFFIC_RAMP_BEGIN_S = 5400  # next peak begins at this offset"
      bound_direction: "upper"
      bound_value: "5400"
      context_summary: "migration must complete before the next peak ramp at 5400s"
    # ↓ 12 more entries to be added in Step 2 below.
  noir_fiction:
    - text: "The desk sergeant logged Costa's pickup at the precinct at five past nine — the suspect, demonstrably, was no longer at large."
      bound_direction: "upper"
      bound_value: "9:05 PM"
      context_summary: "police logged him in custody at 9:05 PM, so at-large window ends here"
    - text: "Mickey was last seen alive — and very much alone — by the doorman at the Astor at half past seven, the night of the killing."
      bound_direction: "lower"
      bound_value: "7:30 PM"
      context_summary: "last sighting alive at 7:30 PM, so the relevant timeline begins by then"
    - text: "Costa's lawyer arrived at the holding cell at quarter to ten; Costa had been there for the better part of an hour by then."
      bound_direction: "upper"
      bound_value: "9:45 PM (consistent with 9:05 PM lock-up)"
      context_summary: "redundant upper bound; arrived while suspect already in custody"
    - text: "The bartender at the Indigo, three blocks from where the body was found, served Costa a third gin at twenty after eight — Costa lingered another half hour, the bartender said, then left."
      bound_direction: "lower"
      bound_value: "8:50 PM"
      context_summary: "suspect at large at least until 8:50 PM (8:20 + 30min)"
    # ↓ 12 more entries to be added in Step 2 below.
```

- [ ] **Step 2: Author the remaining 12 python_repo items and 12 noir_fiction items.**

Replace the placeholder comments with the new entries.

For python_repo: the full 16-item pool must satisfy `max(all 16 lowers) < min(all 16 uppers)` (proof in spec — no subset can ever contradict). With the 4 seed entries the current `max_lower = 1800` (CERT_ROTATION) and `min_upper = 5400` (PEAK_TRAFFIC). Each new lower-bound entry must have `bound_value ≤ 5300` and each new upper-bound entry must have `bound_value ≥ max_lower + 100`. Suggested topics — lower bounds: replica catchup, DDL lock-cooldown, leader election quiesce, auto-vacuum tail completion, trace-flush buffer, in-flight-RPC drain; upper bounds: scheduled-job kickoff, autoscaler scale-down, session cookie rotation, metrics aggregation rollup, weekly-snapshot kickoff, on-call handoff. Roughly 6 lower / 6 upper to keep both classes well-represented for any random subset.

For noir_fiction: same contradiction-free constraint applies. Vary narrators, locations, sensory framing — the bound direction must be inferrable only by reading the language (don't telegraph with phrases like "at least" or "by then" in every entry). After authoring, verify:

```bash
poetry run python -c "
from agent_retrieval.schema.template import ExperimentTemplate
t = ExperimentTemplate.from_yaml('experiments/pure_reasoning.yaml')
for profile in ['python_repo', 'noir_fiction']:
    items = t.fixed_pool[profile]
    print(f'{profile}: {len(items)} items')
    lowers = [float(it['bound_value'].split()[0].replace(':', '.')) for it in items if it['bound_direction'] == 'lower']
    uppers = [float(it['bound_value'].split()[0].replace(':', '.')) for it in items if it['bound_direction'] == 'upper']
    print(f'  lowers: {len(lowers)}, max={max(lowers)}')
    print(f'  uppers: {len(uppers)}, min={min(uppers)}')
    print(f'  contradiction-free: {max(lowers) < min(uppers)}')
"
```

Expected output: `16 items`, `contradiction-free: True` for each profile.

(For noir_fiction, the time parsing in the snippet above is fragile — adapt it or write a small helper that normalizes "9:05 PM" → minutes-since-midnight to verify.)

- [ ] **Step 3: Request user review.**

Paste the `fixed_pool` block to the user. Wait for approval.

- [ ] **Step 4: Commit.**

```bash
git add experiments/pure_reasoning.yaml
git commit -m "$(cat <<'EOF'
experiments: add pure_reasoning template with 16-item fixed pools

python_repo: numeric parameters bounding a database migration window.
noir_fiction: prose passages bounding the suspect's at-large window.

Both pools authored to be contradiction-free per the design spec.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 12: End-to-end smoke test — generate one cell of `pure_reasoning`

**Why:** No LLM cost; validates the cheapest path before doing the same for `multi_retrieval`.

**Files:**
- (No code changes — runs the actual generator.)

- [ ] **Step 1: Run the generator for `pure_reasoning` only, smallest cell.**

Inspect the existing `scripts/generate_parallel.py` to understand the CLI shape:

```bash
poetry run python scripts/generate_parallel.py --help
```

Then run it scoped to the new experiment type. Adapt the flags as needed:

```bash
poetry run python scripts/generate_parallel.py --workers 1 --experiments pure_reasoning
```

Expected: 10 cells generated (5 n_items × 2 profiles). All complete without LLM calls.

- [ ] **Step 2: Sanity-check one of the generated answer keys.**

```bash
poetry run python -c "
from agent_retrieval.schema.answer_key import AnswerKey
ak = AnswerKey.from_yaml('workspace/judge/answer_keys/pure_reasoning__python_repo__n2.yaml')
print(f'parametrisation_id: {ak.parametrisation_id}')
print(f'item count: {len(ak.items)}')
print(f'question (first 200 chars):')
print(ak.expected_answers.question[:200])
print(f'item 1 bound_direction: {ak.items[0].bound_direction}, value: {ak.items[0].value}')
"
```

Expected: `pure_reasoning__python_repo__n2`, `item count: 2`, question contains the inlined facts, items have bound_direction set.

- [ ] **Step 3: Commit.**

(No code changes; nothing to commit. The generated answer keys live under `workspace/` which is gitignored — confirm by running `git status` and ensuring no untracked source files.)

```bash
git status
```

If clean (only workspace artifacts), nothing to commit.

---

## Task 13: End-to-end smoke test — generate one cell of `multi_retrieval`

**Why:** Validates the LLM-driven insertion path for the new fixed-pool flow. Costs one model call per cell, so scope it to a single small cell first.

**Files:**
- (No code changes — runs the actual generator with a filter.)

- [ ] **Step 1: Confirm the python_repo background pool exists.**

```bash
ls workspace/background_corpora/python_repo/ | head -3
```

Expected: at least a few `.py` files. If empty, the pool generator will run first (incurs cost). If you want to limit blast radius, run only n=2:

- [ ] **Step 2: Run the generator scoped to `multi_retrieval__python_repo__800k__hard__contextual__n2`.**

Adapt the runner to filter to one parametrisation. The simplest path: temporarily edit the grid in `experiments/multi_retrieval.yaml` to `n_items: [2]` and `content_profile: [python_repo]`, run, then revert. (Don't commit the temporary edit.)

```bash
# (temporarily edit experiments/multi_retrieval.yaml)
poetry run python scripts/generate_parallel.py --workers 1 --experiments multi_retrieval
# (revert experiments/multi_retrieval.yaml)
git checkout experiments/multi_retrieval.yaml
```

Expected: one corpus assembled, two pre-authored canary needles inserted, answer key written.

- [ ] **Step 3: Sanity-check the generated answer key.**

```bash
poetry run python -c "
from agent_retrieval.schema.answer_key import AnswerKey
ak = AnswerKey.from_yaml('workspace/judge/answer_keys/multi_retrieval__python_repo__800k__hard__contextual__n2.yaml')
for it in ak.items:
    print(f'{it.item_id}: {it.inserted_text!r} (value={it.value}) at {it.file_path}:{it.line_range}')
"
```

Expected: 2 items with their pre-authored `inserted_text` and `value` fields, each placed in a real file with a concrete `line_range`.

- [ ] **Step 4: Verify the inserted text appears verbatim in the corpus file.**

```bash
poetry run python -c "
from agent_retrieval.schema.answer_key import AnswerKey
ak = AnswerKey.from_yaml('workspace/judge/answer_keys/multi_retrieval__python_repo__800k__hard__contextual__n2.yaml')
import os
for it in ak.items:
    path = os.path.join('workspace/runner/corpora', ak.parametrisation_id, it.file_path)
    text = open(path).read()
    assert it.inserted_text in text, f'MISSING: {it.inserted_text!r} not in {path}'
    print(f'OK: {it.item_id} verbatim in {path}')
"
```

Expected: `OK:` lines for both items.

- [ ] **Step 5: Commit.**

(No code changes; verify clean tree.)

```bash
git status
```

Expected: clean (only workspace artifacts).

---

## Task 14: Update changelog / docs (optional)

**Why:** If the project tracks experiment types in a top-level docs file, mention the two new ones. Otherwise this task is a no-op.

**Files:**
- Possibly modify: `README.md`, `docs/superpowers/specs/2026-04-03-experiment-design-v2.md` if it has a Type Taxonomy table.

- [ ] **Step 1: Search for an experiment-type taxonomy.**

```bash
grep -rn "experiment_type" docs/ README.md 2>/dev/null | head
```

- [ ] **Step 2: If a taxonomy table exists, append two rows.**

For example, in `docs/superpowers/specs/2026-04-03-experiment-design-v2.md`'s Experiment Type Taxonomy table:

| Type | What the agent must do | Key challenge |
|---|---|---|
| `multi_retrieval` | Find N category-coherent pre-authored items in a large corpus and report each verbatim | Pure retrieval+retention without reasoning |
| `pure_reasoning` | Reason across N facts handed in the prompt to derive a structured interval-shaped answer | Pure cross-fact reasoning without retrieval |

- [ ] **Step 3: Commit if anything was changed.**

```bash
git add <modified files>
git commit -m "$(cat <<'EOF'
docs: register multi_retrieval and pure_reasoning in type taxonomy

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

If no taxonomy table exists, skip.

---

## Self-review checklist (run after all tasks complete)

- [ ] All 14 tasks committed.
- [ ] `poetry run pytest -v` shows the same number of passing tests as the pre-flight baseline plus the new tests added in tasks 1–9.
- [ ] `experiments/multi_retrieval.yaml` validates with `ExperimentTemplate.from_yaml`.
- [ ] `experiments/pure_reasoning.yaml` validates and is contradiction-free per the spec.
- [ ] `workspace/judge/answer_keys/pure_reasoning__python_repo__n2.yaml` exists and parses.
- [ ] `workspace/judge/answer_keys/multi_retrieval__python_repo__800k__hard__contextual__n2.yaml` exists, parses, and the inserted text appears verbatim in the corresponding corpus file.

---

## Out of scope for this plan

- Running full batches of either experiment against an agent model (those are batch YAMLs and runner invocations, separate from generation).
- Analysis updates / new figures for the new experiment types.
- Adapting the judge prompt to better evaluate the new rubric criteria. The current scoring infrastructure reads rubric_criteria from the answer key and asks the judge to score each — it should work as-is, but a follow-up tuning pass on the judge prompt may be desirable after first results land.
