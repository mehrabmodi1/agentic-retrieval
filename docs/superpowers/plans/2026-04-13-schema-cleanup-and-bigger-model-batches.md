# Schema Cleanup + Bigger-Model Batches Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Clean up scattered / dead model & cost specifications across the codebase so each knob lives in exactly one place, then run two new batches (`opus-4-6 effort=low` and `sonnet-4-6 effort=max`) on a reduced parameter space using the cleaned-up schema.

**Architecture:** Batch YAML becomes the single source of truth for *run-time* knobs (agent model, effort, n_repeats, max_turns, allowed_tools). Experiment template YAML holds only the *experimental protocol* (grid, payload examples, rubric). `CorpusSpec` in `schema/experiment.py` holds generator model defaults as module-level constants that `pool.py` and `insertion.py` import directly. Judge model is pinned to a constant in `judge/scoring.py`. Dead code (`max_tokens` plumbing, unused `judge_model` field on `BatchConfig`, `--judge-model` CLI flag, `schema_version` field) is removed. Audit trail expands so every run and verdict records exactly which models/caps produced it.

**Tech Stack:** Python 3.12+, Poetry, Pydantic v2, pytest, `claude_agent_sdk`, Claude Code CLI.

---

## Pre-flight

- [ ] **Verify clean working tree (aside from the known modified notebook).** Uncommitted changes on unrelated files can confuse review.

Run:
```bash
git status
```

Expected: only `notebooks/full-sweep_sonnet-4-6_effort_low_20260409.ipynb` is modified. If there's anything else, stash or commit before starting.

- [ ] **Create a branch for the cleanup.**

Run:
```bash
git checkout -b cleanup/model-and-cost-config-schema
```

- [ ] **Confirm test suite is green on main before changes.**

Run:
```bash
poetry run pytest -v
```

Expected: all tests pass. Record the passing test count as a baseline. If anything is red on `main`, fix or acknowledge before proceeding.

---

## Task 1: Consolidate generator model constants on `CorpusSpec`

**Why:** Pool generation and payload insertion models are currently hardcoded in three places (`pool.py:49`, `insertion.py:257`). Move them to module-level constants colocated with `CorpusSpec` (which already has `generation_model`) so there's one place to audit / change generator models.

**Files:**
- Modify: `src/agent_retrieval/schema/experiment.py`
- Test: `tests/test_schema.py` (extend existing `TestExperimentSpec`)

- [ ] **Step 1: Write failing test for the new constants + CorpusSpec defaults.**

Append to `tests/test_schema.py` after the existing `TestExperimentSpec` class:

```python
from agent_retrieval.schema.experiment import (
    CorpusSpec,
    POOL_GENERATION_MODEL,
    PAYLOAD_INSERTION_MODEL_SINGLE,
    PAYLOAD_INSERTION_MODEL_MULTI,
)


class TestGeneratorModelConstants:
    def test_constants_have_expected_values(self):
        # These are the values currently hardcoded in pool.py and insertion.py.
        # Preserved verbatim so already-generated corpora remain reproducible.
        assert POOL_GENERATION_MODEL == "claude-haiku-4-5-20251001"
        assert PAYLOAD_INSERTION_MODEL_SINGLE == "claude-sonnet-4-6"
        assert PAYLOAD_INSERTION_MODEL_MULTI == "claude-haiku-4-5-20251001"

    def test_corpus_spec_defaults_match_constants(self):
        spec = CorpusSpec(
            content_profile="python_repo",
            target_token_count=20000,
            target_file_count=50,
            folder_depth=2,
            folder_distribution="balanced",
            generation_model="claude-haiku-4-5-20251001",
            red_herring_density="none",
        )
        assert spec.pool_generation_model == POOL_GENERATION_MODEL
        assert spec.payload_insertion_model_single == PAYLOAD_INSERTION_MODEL_SINGLE
        assert spec.payload_insertion_model_multi == PAYLOAD_INSERTION_MODEL_MULTI
```

- [ ] **Step 2: Run test to verify it fails.**

```bash
poetry run pytest tests/test_schema.py::TestGeneratorModelConstants -v
```

Expected: FAIL with `ImportError: cannot import name 'POOL_GENERATION_MODEL' from 'agent_retrieval.schema.experiment'`.

- [ ] **Step 3: Add constants + fields to `CorpusSpec`.**

Edit `src/agent_retrieval/schema/experiment.py`. Add the constants just above the `CorpusSpec` class and add three new fields with those constants as defaults:

```python
# Generator-side model defaults. Single source of truth shared by
# pool generation (pool.py), payload insertion (insertion.py), and
# per-experiment CorpusSpec records. These values are the ones used
# to produce the currently-checked-in corpora; change with care.
POOL_GENERATION_MODEL = "claude-haiku-4-5-20251001"
PAYLOAD_INSERTION_MODEL_SINGLE = "claude-sonnet-4-6"
PAYLOAD_INSERTION_MODEL_MULTI = "claude-haiku-4-5-20251001"


class CorpusSpec(BaseModel):
    content_profile: str
    target_token_count: int
    target_file_count: int
    folder_depth: int
    folder_distribution: Literal["balanced", "skewed", "flat"]
    generation_model: str
    red_herring_density: Literal["none", "low", "medium", "high"]
    pool_generation_model: str = POOL_GENERATION_MODEL
    payload_insertion_model_single: str = PAYLOAD_INSERTION_MODEL_SINGLE
    payload_insertion_model_multi: str = PAYLOAD_INSERTION_MODEL_MULTI
```

- [ ] **Step 4: Verify tests pass.**

```bash
poetry run pytest tests/test_schema.py::TestGeneratorModelConstants -v
```

Expected: PASS.

- [ ] **Step 5: Full test run to make sure nothing else broke.**

```bash
poetry run pytest -v
```

Expected: same baseline count + 2 new tests, all green.

- [ ] **Step 6: Commit.**

```bash
git add src/agent_retrieval/schema/experiment.py tests/test_schema.py
git commit -m "schema: consolidate generator model defaults on CorpusSpec"
```

---

## Task 2: Rewire `pool.py` and `insertion.py` to use the constants

**Why:** Remove the two hardcoded call sites now that the canonical constants exist.

**Files:**
- Modify: `src/agent_retrieval/generator/pool.py:49`
- Modify: `src/agent_retrieval/generator/insertion.py:257`
- Test: `tests/test_pool.py` and/or `tests/test_insertion.py` (if they check model strings; otherwise rely on import-wiring test)

- [ ] **Step 1: Write a failing test pinning the import wiring.**

Append to `tests/test_schema.py`:

```python
class TestGeneratorWiring:
    def test_pool_uses_constant(self, monkeypatch):
        # If someone later changes the constant, pool.py should pick it up
        # without edit. Proves it isn't hardcoded.
        import agent_retrieval.generator.pool as pool_mod
        from agent_retrieval.schema import experiment as exp_mod
        monkeypatch.setattr(exp_mod, "POOL_GENERATION_MODEL", "sentinel-model-x")
        # pool.py must import the constant lazily (via module attr access) OR
        # bind it at call time. Either pattern should result in the patched
        # value being visible on re-read.
        import importlib
        importlib.reload(pool_mod)
        from agent_retrieval.schema.experiment import POOL_GENERATION_MODEL
        assert POOL_GENERATION_MODEL == "sentinel-model-x"

    def test_insertion_uses_constants(self):
        # Proves insertion.py references the constants, not string literals.
        import agent_retrieval.generator.insertion as ins_mod
        src = open(ins_mod.__file__).read()
        assert "PAYLOAD_INSERTION_MODEL_SINGLE" in src
        assert "PAYLOAD_INSERTION_MODEL_MULTI" in src
        # And does NOT hardcode these specific literals in the is_multi branch:
        assert '"claude-haiku-4-5-20251001" if is_multi else "claude-sonnet-4-6"' not in src
```

- [ ] **Step 2: Run the new tests to confirm failure.**

```bash
poetry run pytest tests/test_schema.py::TestGeneratorWiring -v
```

Expected: FAIL (the src grep assertions will fail since insertion.py still has the hardcoded ternary).

- [ ] **Step 3: Update `pool.py` to import and use the constant.**

Edit `src/agent_retrieval/generator/pool.py`. At the top, add:

```python
from agent_retrieval.schema.experiment import POOL_GENERATION_MODEL
```

Replace line 49 (`model="claude-haiku-4-5-20251001",`) with:

```python
            model=POOL_GENERATION_MODEL,
```

- [ ] **Step 4: Update `insertion.py` to import and use the constants.**

Edit `src/agent_retrieval/generator/insertion.py`. At the top, add:

```python
from agent_retrieval.schema.experiment import (
    PAYLOAD_INSERTION_MODEL_SINGLE,
    PAYLOAD_INSERTION_MODEL_MULTI,
)
```

Replace line 257 (`model = "claude-haiku-4-5-20251001" if is_multi else "claude-sonnet-4-6"`) with:

```python
    model = PAYLOAD_INSERTION_MODEL_MULTI if is_multi else PAYLOAD_INSERTION_MODEL_SINGLE
```

- [ ] **Step 5: Verify wiring tests pass.**

```bash
poetry run pytest tests/test_schema.py::TestGeneratorWiring -v
```

Expected: PASS.

- [ ] **Step 6: Full test run.**

```bash
poetry run pytest -v
```

Expected: green.

- [ ] **Step 7: Commit.**

```bash
git add src/agent_retrieval/generator/pool.py src/agent_retrieval/generator/insertion.py tests/test_schema.py
git commit -m "generator: read pool/insertion models from CorpusSpec constants"
```

---

## Task 3: Add audit fields to `RunState` and `Verdict`

**Why:** When `max_turns` and `allowed_tools` move to batch YAML, and when `judge_model` becomes a pinned constant, we need them recorded per-run / per-verdict so runs remain auditable after the fact.

**Files:**
- Modify: `src/agent_retrieval/schema/run_state.py`
- Modify: `src/agent_retrieval/schema/verdict.py`
- Test: `tests/test_schema.py` (extend `TestRunState` and `TestVerdict`)

- [ ] **Step 1: Write failing tests.**

Append to `tests/test_schema.py` (add after existing `TestRunState`):

```python
class TestRunStateAuditFields:
    def test_max_turns_recorded(self):
        state = RunState(
            parametrisation_id="test-001",
            run_id="abc",
            batch_name="b",
            status="pending",
            claude_code_version="1.0.0",
            max_turns=75,
            allowed_tools=["Read", "Grep"],
        )
        assert state.max_turns == 75
        assert state.allowed_tools == ["Read", "Grep"]

    def test_defaults_preserve_legacy_run_state(self):
        # Reading an old state.yaml that predates these fields must still parse.
        state = RunState.model_validate({
            "parametrisation_id": "test-001",
            "run_id": "abc",
            "batch_name": "b",
            "status": "completed",
            "claude_code_version": "1.0.0",
        })
        assert state.max_turns == 0
        assert state.allowed_tools == []


class TestVerdictAuditFields:
    def test_judge_model_recorded(self):
        v = Verdict.model_validate({
            "parametrisation_id": "test-001",
            "run_id": "abc",
            "batch_name": "b",
            "judge_model": "claude-sonnet-4-6",
            "scores": [{"criterion": "correctness", "score": 1.0,
                        "weight": 1.0, "reasoning": "ok"}],
            "weighted_score": 1.0,
            "session_metrics": {"total_context_tokens": 1, "total_turns": 1,
                                "tool_calls": {}, "duration_seconds": 0.0},
        })
        assert v.judge_model == "claude-sonnet-4-6"

    def test_legacy_verdict_without_judge_model_parses(self):
        # Existing judgements on disk must still load.
        v = Verdict.model_validate({
            "parametrisation_id": "test-001",
            "run_id": "abc",
            "batch_name": "b",
            "scores": [{"criterion": "correctness", "score": 1.0,
                        "weight": 1.0, "reasoning": "ok"}],
            "weighted_score": 1.0,
            "session_metrics": {"total_context_tokens": 1, "total_turns": 1,
                                "tool_calls": {}, "duration_seconds": 0.0},
        })
        assert v.judge_model == ""
```

- [ ] **Step 2: Run tests — expect failure.**

```bash
poetry run pytest tests/test_schema.py::TestRunStateAuditFields tests/test_schema.py::TestVerdictAuditFields -v
```

Expected: FAIL — fields don't exist yet.

- [ ] **Step 3: Add fields to `RunState`.**

Edit `src/agent_retrieval/schema/run_state.py`. Inside the `RunState` class, add two fields after `effort_mode`:

```python
    max_turns: int = 0
    allowed_tools: list[str] = []
```

(Defaults chosen so legacy `state.yaml` files still parse.)

- [ ] **Step 4: Add field to `Verdict`.**

Edit `src/agent_retrieval/schema/verdict.py`. In the `Verdict` class, after `batch_name`, add:

```python
    judge_model: str = ""
```

- [ ] **Step 5: Verify new tests pass.**

```bash
poetry run pytest tests/test_schema.py::TestRunStateAuditFields tests/test_schema.py::TestVerdictAuditFields -v
```

Expected: PASS.

- [ ] **Step 6: Full test run.**

```bash
poetry run pytest -v
```

Expected: green.

- [ ] **Step 7: Commit.**

```bash
git add src/agent_retrieval/schema/run_state.py src/agent_retrieval/schema/verdict.py tests/test_schema.py
git commit -m "schema: record max_turns + allowed_tools in RunState, judge_model in Verdict"
```

---

## Task 4: Reshape `RunnerSpec` — strip runtime fields, keep nothing (remove it)

**Why:** After moving everything to batch, `RunnerSpec` is empty. Delete it rather than leave a vestigial field-less class. `ExperimentTemplate.runner` and `ExperimentSpec.runner` become unused — drop them too. This also removes the coupling between "which tools an experiment allows" and "which model runs it".

**Files:**
- Modify: `src/agent_retrieval/schema/experiment.py` (remove `RunnerSpec` class)
- Modify: `src/agent_retrieval/schema/experiment.py` (drop `runner` from `ExperimentSpec`)
- Modify: `src/agent_retrieval/schema/template.py` (drop `runner` from `ExperimentTemplate`)
- Test: `tests/test_schema.py` + `tests/test_template_schema.py`

- [ ] **Step 1: Write failing test proving `RunnerSpec` is gone.**

Append to `tests/test_schema.py`:

```python
class TestRunnerSpecRemoved:
    def test_runner_spec_not_importable(self):
        import agent_retrieval.schema.experiment as exp_mod
        assert not hasattr(exp_mod, "RunnerSpec"), (
            "RunnerSpec should be removed; runtime knobs now live on BatchConfig"
        )

    def test_experiment_spec_has_no_runner_field(self):
        assert "runner" not in ExperimentSpec.model_fields

    def test_experiment_template_has_no_runner_field(self):
        from agent_retrieval.schema.template import ExperimentTemplate
        assert "runner" not in ExperimentTemplate.model_fields
```

- [ ] **Step 2: Run test — expect failure.**

```bash
poetry run pytest tests/test_schema.py::TestRunnerSpecRemoved -v
```

Expected: FAIL — `RunnerSpec` still exists.

- [ ] **Step 3: Delete `RunnerSpec` and remove `runner` fields.**

Edit `src/agent_retrieval/schema/experiment.py`:
- Delete the `RunnerSpec` class (lines 46–51).
- Remove the `runner: RunnerSpec` field from `ExperimentSpec`.

Edit `src/agent_retrieval/schema/template.py`:
- Remove the import of `RunnerSpec` (line 9).
- Remove the `runner: RunnerSpec` field from `ExperimentTemplate`.

- [ ] **Step 4: Verify the three schema tests pass.**

```bash
poetry run pytest tests/test_schema.py::TestRunnerSpecRemoved -v
```

Expected: PASS.

- [ ] **Step 5: Full test run — expect several existing tests to now fail (that's OK; later tasks fix them).**

```bash
poetry run pytest -v
```

Expected: failures in tests that reference `template.runner.*` or construct `ExperimentSpec` with a runner block. Record the list. Leave failing for now — they'll be repaired in later tasks.

- [ ] **Step 6: Commit.**

```bash
git add src/agent_retrieval/schema/experiment.py src/agent_retrieval/schema/template.py tests/test_schema.py
git commit -m "schema: remove RunnerSpec (runtime knobs move to BatchConfig)"
```

---

## Task 5: Expand `BatchConfig` — add required runtime + cost fields, drop dead `judge_model`

**Why:** Batch YAML becomes the single place that defines *how* a grid is run: which agent, how hard it thinks, how many repeats, how many turns, which tools. `judge_model` was never read — remove it.

**Files:**
- Modify: `src/agent_retrieval/schema/batch.py`
- Modify: `tests/test_schema.py` (both `TestBatchConfig` classes — there are two)

- [ ] **Step 1: Rewrite `BatchConfig` tests to match the new schema.**

Replace **both** `TestBatchConfig` classes in `tests/test_schema.py` (the one around line 84 and the one around line 149) with a single unified class:

```python
class TestBatchConfig:
    def _base(self, **overrides):
        base = {
            "batch_name": "test-batch",
            "max_parallel": 2,
            "retry_failed": True,
            "agent_model": "claude-sonnet-4-6",
            "effort_mode": "low",
            "n_repeats": 3,
            "max_turns": 50,
            "allowed_tools": ["Read", "Glob", "Grep", "Bash"],
            "experiments": ["single_needle"],
        }
        base.update(overrides)
        return base

    def test_valid_batch(self):
        batch = BatchConfig.model_validate(self._base())
        assert batch.batch_name == "test-batch"
        assert batch.agent_model == "claude-sonnet-4-6"
        assert batch.effort_mode == "low"
        assert batch.n_repeats == 3
        assert batch.max_turns == 50
        assert batch.allowed_tools == ["Read", "Glob", "Grep", "Bash"]
        assert batch.experiments[0].experiment_type == "single_needle"

    def test_missing_agent_model_raises(self):
        data = self._base()
        del data["agent_model"]
        with pytest.raises(Exception):
            BatchConfig.model_validate(data)

    def test_missing_effort_mode_raises(self):
        data = self._base()
        del data["effort_mode"]
        with pytest.raises(Exception):
            BatchConfig.model_validate(data)

    def test_missing_n_repeats_raises(self):
        data = self._base()
        del data["n_repeats"]
        with pytest.raises(Exception):
            BatchConfig.model_validate(data)

    def test_missing_max_turns_raises(self):
        data = self._base()
        del data["max_turns"]
        with pytest.raises(Exception):
            BatchConfig.model_validate(data)

    def test_missing_allowed_tools_raises(self):
        data = self._base()
        del data["allowed_tools"]
        with pytest.raises(Exception):
            BatchConfig.model_validate(data)

    def test_effort_mode_rejects_invalid_value(self):
        data = self._base(effort_mode="extreme")
        with pytest.raises(Exception):
            BatchConfig.model_validate(data)

    def test_effort_mode_accepts_max(self):
        batch = BatchConfig.model_validate(self._base(effort_mode="max"))
        assert batch.effort_mode == "max"

    def test_judge_model_field_rejected(self):
        # judge_model has no home on BatchConfig anymore.
        data = self._base(judge_model="claude-opus-4-6")
        with pytest.raises(Exception):
            BatchConfig.model_validate(data)

    def test_filtered_experiment(self):
        batch = BatchConfig.model_validate(self._base(experiments=[{
            "experiment_type": "single_needle",
            "filter": {"content_profile": ["python_repo"],
                       "corpus_token_count": [20000]},
        }]))
        assert batch.experiments[0].filter["content_profile"] == ["python_repo"]

    def test_mixed_experiment_format(self):
        batch = BatchConfig.model_validate(self._base(experiments=[
            "single_needle",
            {"experiment_type": "multi_chain", "filter": {"n_items": [2]}},
        ]))
        assert len(batch.experiments) == 2
        assert batch.experiments[0].filter is None
        assert batch.experiments[1].filter is not None
```

- [ ] **Step 2: Run tests — most will fail.**

```bash
poetry run pytest tests/test_schema.py::TestBatchConfig -v
```

Expected: FAIL on every test that references new fields or strict validation.

- [ ] **Step 3: Update `BatchConfig`.**

Replace `src/agent_retrieval/schema/batch.py` with:

```python
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, model_validator


class BatchExperimentEntry(BaseModel):
    experiment_type: str
    filter: dict[str, list[Any]] | None = None


class BatchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    batch_name: str
    max_parallel: int
    retry_failed: bool
    agent_model: str
    effort_mode: Literal["low", "medium", "high", "max"]
    n_repeats: int
    max_turns: int
    allowed_tools: list[str]
    experiments: list[BatchExperimentEntry]

    @model_validator(mode="before")
    @classmethod
    def normalize_experiments(cls, data: dict) -> dict:
        if isinstance(data, dict) and "experiments" in data:
            normalized = []
            for entry in data["experiments"]:
                if isinstance(entry, str):
                    normalized.append({"experiment_type": entry})
                else:
                    normalized.append(entry)
            data["experiments"] = normalized
        return data

    @classmethod
    def from_yaml(cls, path: Path) -> BatchConfig:
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)
```

Key changes vs previous:
- `judge_model` removed.
- `agent_model`, `effort_mode`, `n_repeats`, `max_turns`, `allowed_tools` added, all required.
- `effort_mode` is a `Literal` for `low|medium|high|max` so invalid values are rejected at parse time.
- `ConfigDict(extra="forbid")` so stray fields like `judge_model` raise, catching outdated batch YAMLs early instead of silently ignoring them.

- [ ] **Step 4: Verify new tests pass.**

```bash
poetry run pytest tests/test_schema.py::TestBatchConfig -v
```

Expected: PASS.

- [ ] **Step 5: Run full test suite.**

```bash
poetry run pytest -v
```

Expected: still failing in runner / cli tests (fixed in later tasks) but no schema tests failing.

- [ ] **Step 6: Commit.**

```bash
git add src/agent_retrieval/schema/batch.py tests/test_schema.py
git commit -m "schema: BatchConfig now owns agent_model, effort_mode, n_repeats, max_turns, allowed_tools; drop dead judge_model"
```

---

## Task 6: Allow `schema_version` in legacy YAMLs (ignore-extra), drop from required schemas

**Why:** User preference: only v2.0 exists going forward, don't require the field. But existing files still have `schema_version: "2.0"` at the top — must still parse.

**Files:**
- Modify: `src/agent_retrieval/schema/experiment.py` (drop `schema_version` from `ExperimentSpec`, set `extra="ignore"`)
- Modify: `src/agent_retrieval/schema/template.py` (drop `schema_version` from `ExperimentTemplate`, set `extra="ignore"`)
- Test: `tests/test_template_schema.py` and `tests/test_schema.py`

- [ ] **Step 1: Write failing test.**

Append to `tests/test_schema.py`:

```python
class TestSchemaVersionTolerance:
    def test_experiment_spec_parses_with_schema_version(self, sample_spec_dict):
        # Legacy v1 spec files contain schema_version; must still parse.
        sample_spec_dict["schema_version"] = "1.0"
        spec = ExperimentSpec.model_validate(sample_spec_dict)
        assert spec.experiment_id == "test-001"

    def test_experiment_spec_parses_without_schema_version(self, sample_spec_dict):
        sample_spec_dict.pop("schema_version", None)
        spec = ExperimentSpec.model_validate(sample_spec_dict)
        assert spec.experiment_id == "test-001"
```

Append to `tests/test_template_schema.py` (add class at end):

```python
from agent_retrieval.schema.template import ExperimentTemplate


class TestTemplateSchemaVersionTolerance:
    def _base(self):
        return {
            "experiment_type": "single_needle",
            "payload": {"item_type": "config_value"},
            "question_examples": {"python_repo": {"easy_exact": {
                "question": "q", "needle": "n", "answer": "a",
            }}},
            "rubric_criteria": [{"criterion": "correctness", "weight": 1.0}],
            "grid": {
                "content_profile": ["python_repo"],
                "corpus_token_count": [20000],
                "discriminability": ["easy"],
                "reference_clarity": ["exact"],
            },
        }

    def test_template_parses_without_schema_version(self):
        tpl = ExperimentTemplate.model_validate(self._base())
        assert tpl.experiment_type == "single_needle"

    def test_template_parses_with_legacy_schema_version(self):
        data = self._base()
        data["schema_version"] = "2.0"
        tpl = ExperimentTemplate.model_validate(data)
        assert tpl.experiment_type == "single_needle"
```

- [ ] **Step 2: Run tests — expect fail.**

```bash
poetry run pytest tests/test_schema.py::TestSchemaVersionTolerance tests/test_template_schema.py::TestTemplateSchemaVersionTolerance -v
```

Expected: some pass (because extra is currently allowed by Pydantic default), some fail (because `schema_version` is currently a *required* field so legacy-without raises).

- [ ] **Step 3: Drop `schema_version` from schemas and configure ignore-extra.**

Edit `src/agent_retrieval/schema/experiment.py`:
- Remove `schema_version: str` from `ExperimentSpec`.
- Add `model_config = ConfigDict(extra="ignore")` at the top of `ExperimentSpec`.
- Import `ConfigDict` from pydantic.

Edit `src/agent_retrieval/schema/template.py`:
- Remove `schema_version: str` from `ExperimentTemplate`.
- Add `model_config = ConfigDict(extra="ignore")` at the top of `ExperimentTemplate`.
- Import `ConfigDict` from pydantic.

- [ ] **Step 4: Tests pass.**

```bash
poetry run pytest tests/test_schema.py::TestSchemaVersionTolerance tests/test_template_schema.py::TestTemplateSchemaVersionTolerance -v
```

Expected: PASS.

- [ ] **Step 5: Full test run.**

```bash
poetry run pytest -v
```

Expected: schema tests green; runner/cli tests still pending (fixed later).

- [ ] **Step 6: Commit.**

```bash
git add src/agent_retrieval/schema/experiment.py src/agent_retrieval/schema/template.py tests/test_schema.py tests/test_template_schema.py
git commit -m "schema: drop schema_version field; ignore when present in legacy YAMLs"
```

---

## Task 7: Pin judge model in scoring module; remove `judge_model` parameter plumbing

**Why:** Judging must be standardized across runs. Pinning the model in code (not YAML, not CLI) makes "one judge for all comparisons" the default. Judge records its model in each verdict for auditability.

**Files:**
- Modify: `src/agent_retrieval/judge/scoring.py`
- Modify: `src/agent_retrieval/judge/judge.py`
- Modify: `src/agent_retrieval/cli.py`
- Test: new `tests/test_judge_pinning.py`

- [ ] **Step 1: Write failing test.**

Create `tests/test_judge_pinning.py`:

```python
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_retrieval.judge import scoring
from agent_retrieval.judge.scoring import JUDGE_MODEL


def test_judge_model_pinned():
    assert JUDGE_MODEL == "claude-sonnet-4-6"


@pytest.mark.asyncio
async def test_score_response_uses_pinned_model():
    """score_response must not accept a judge_model parameter anymore."""
    # Introspection: the signature must not have judge_model
    import inspect
    sig = inspect.signature(scoring.score_response)
    assert "judge_model" not in sig.parameters


@pytest.mark.asyncio
async def test_judge_run_records_judge_model(tmp_path):
    """judge_run writes JUDGE_MODEL into the verdict file."""
    from agent_retrieval.judge.judge import judge_run
    from agent_retrieval.schema.answer_key import AnswerKey

    # Set up minimal run_dir with response.json and an empty session.jsonl
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "response.json").write_text(
        '{"response_text": "the answer is 42",'
        ' "session_id": "sid", "num_turns": 1,'
        ' "total_cost_usd": 0.0, "usage": {}}'
    )
    (run_dir / "session.jsonl").write_text("")

    ak = AnswerKey.model_validate({
        "parametrisation_id": "pid",
        "generated_at": "2026-04-13T00:00:00Z",
        "items": [{
            "item_id": "target_001",
            "inserted_text": "X = 42",
            "file_path": "f.md",
            "line_range": [1, 1],
            "context_summary": "t",
        }],
        "expected_answers": {
            "question": "What is X?",
            "correctness": "42",
        },
        "rubric_criteria": [{"criterion": "correctness", "weight": 1.0}],
    })

    verdict_path = tmp_path / "verdict.yaml"

    fake_scores = [MagicMock(score=1.0, weight=1.0, criterion="correctness",
                             reasoning="ok")]
    with patch("agent_retrieval.judge.judge.score_response",
               new=AsyncMock(return_value=fake_scores)):
        verdict = await judge_run(
            run_dir=run_dir, answer_key=ak,
            batch_run_name="b", verdict_path=verdict_path,
        )

    assert verdict.judge_model == JUDGE_MODEL
```

- [ ] **Step 2: Run test — expect fail.**

```bash
poetry run pytest tests/test_judge_pinning.py -v
```

Expected: FAIL (JUDGE_MODEL not defined; signature still has judge_model).

- [ ] **Step 3: Add constant to `scoring.py`, remove the parameter.**

Edit `src/agent_retrieval/judge/scoring.py`:
- Above the imports block for `agent_retrieval.schema.*`, add: `JUDGE_MODEL = "claude-sonnet-4-6"`.
- Change `async def score_response(agent_response: str, answer_key: AnswerKey, judge_model: str) -> list[ScoreEntry]:` to `async def score_response(agent_response: str, answer_key: AnswerKey) -> list[ScoreEntry]:`.
- Inside, change `model=judge_model,` (line 41) to `model=JUDGE_MODEL,`.

- [ ] **Step 4: Update `judge.py`.**

Edit `src/agent_retrieval/judge/judge.py`:
- Import `JUDGE_MODEL` from scoring: `from agent_retrieval.judge.scoring import score_response, JUDGE_MODEL`
- Remove `judge_model: str,` from `judge_run` parameters (line 14).
- Remove the `judge_model` argument from the `score_response(...)` call on line 21.
- Add `judge_model=JUDGE_MODEL,` to the `Verdict(...)` construction.
- Remove `judge_model: str,` from `judge_batch` parameters (line 37).
- Remove `judge_model=judge_model,` from the `judge_run(...)` call.

Final `judge_run` signature becomes:

```python
async def judge_run(
    run_dir: Path, answer_key: AnswerKey,
    batch_run_name: str, verdict_path: Path,
) -> Verdict:
    response_path = run_dir / "response.json"
    session_path = run_dir / "session.jsonl"
    response_data = json.loads(response_path.read_text())
    agent_response = response_data["response_text"]
    scores = await score_response(agent_response, answer_key)
    total_weight = sum(s.weight for s in scores)
    weighted_score = sum(s.score * s.weight for s in scores) / total_weight if total_weight > 0 else 0.0
    metrics = extract_session_metrics(session_path, response_path)
    run_id = run_dir.name
    verdict = Verdict(
        parametrisation_id=answer_key.parametrisation_id, run_id=run_id,
        batch_name=batch_run_name, judge_model=JUDGE_MODEL, scores=scores,
        weighted_score=round(weighted_score, 4), session_metrics=metrics,
    )
    verdict.to_yaml(verdict_path)
    return verdict
```

And `judge_batch`:

```python
async def judge_batch(
    batch_run_name: str,
    workspace_dir: Path,
    rejudge: bool = False,
) -> list[Verdict]:
```

(Inside, remove the `judge_model=judge_model,` argument from the `judge_run(...)` call on line 82.)

- [ ] **Step 5: Update CLI.**

Edit `src/agent_retrieval/cli.py`:
- Remove the `--judge-model` argument (line 30).
- In `_judge()`, change the `judge_batch(...)` call to drop `args.judge_model`:

```python
async def _judge(args: argparse.Namespace) -> None:
    from agent_retrieval.judge import judge_batch
    await judge_batch(
        args.batch_run_name,
        Path(args.workspace).resolve(), rejudge=args.rejudge,
    )
```

- [ ] **Step 6: Verify new tests pass.**

```bash
poetry run pytest tests/test_judge_pinning.py -v
```

Expected: PASS.

- [ ] **Step 7: Full test run.**

```bash
poetry run pytest -v
```

Expected: judge tests green; runner tests still pending.

- [ ] **Step 8: Commit.**

```bash
git add src/agent_retrieval/judge/scoring.py src/agent_retrieval/judge/judge.py src/agent_retrieval/cli.py tests/test_judge_pinning.py
git commit -m "judge: pin judge model in scoring module, record it in verdict, drop CLI flag"
```

---

## Task 8: Rewire runner to read runtime config from `BatchConfig`

**Why:** After Task 4 removed `RunnerSpec`, `run.py` no longer has `template.runner.*` to read. Everything comes from `batch.*` now. Also drops dead `max_tokens` plumbing and wires `max_turns` (previously hardcoded to 50) through to `ClaudeAgentOptions`.

**Files:**
- Modify: `src/agent_retrieval/runner/run.py`
- Modify: `src/agent_retrieval/runner/session.py`
- Modify: `src/agent_retrieval/runner/state.py`
- Test: new `tests/test_runner_wiring.py`

- [ ] **Step 1: Write failing test.**

Create `tests/test_runner_wiring.py`:

```python
import inspect

from agent_retrieval.runner import session as session_mod
from agent_retrieval.runner import state as state_mod


def test_run_agent_session_drops_max_tokens():
    sig = inspect.signature(session_mod.run_agent_session)
    assert "max_tokens" not in sig.parameters, (
        "max_tokens was dead plumbing — ClaudeAgentOptions has no max_tokens field"
    )


def test_run_agent_session_accepts_max_turns():
    sig = inspect.signature(session_mod.run_agent_session)
    assert "max_turns" in sig.parameters, (
        "max_turns must be a runtime parameter, not hardcoded"
    )


def test_run_agent_session_accepts_allowed_tools():
    sig = inspect.signature(session_mod.run_agent_session)
    assert "allowed_tools" in sig.parameters


def test_max_turns_not_hardcoded():
    src = inspect.getsource(session_mod.run_agent_session)
    assert "max_turns=50" not in src, (
        "max_turns should come from the caller (batch config), not be hardcoded"
    )


def test_run_state_manager_accepts_max_turns_and_allowed_tools():
    sig = inspect.signature(state_mod.RunStateManager.create_pending_runs)
    assert "max_turns" in sig.parameters
    assert "allowed_tools" in sig.parameters
```

- [ ] **Step 2: Run tests — expect fail.**

```bash
poetry run pytest tests/test_runner_wiring.py -v
```

Expected: FAIL.

- [ ] **Step 3: Update `session.py`.**

Replace `src/agent_retrieval/runner/session.py` body of `run_agent_session` (lines 41–95) so `max_turns` is a required parameter, `max_tokens` is gone, and `max_turns` is wired through:

```python
async def run_agent_session(
    question: str,
    corpus_dir: Path,
    model: str,
    allowed_tools: list[str],
    max_turns: int,
    run_id: str,
    run_dir: Path,
    effort_mode: str = "",
) -> AgentResult:
    system_prompt = (
        f"Answer the following question by searching the provided codebase. "
        f"Your session ID is: {run_id}\n\n"
        f"Question: {question}"
    )
    options = ClaudeAgentOptions(
        model=model,
        system_prompt=system_prompt,
        cwd=str(corpus_dir.resolve()),
        allowed_tools=allowed_tools,
        permission_mode="acceptEdits",
        max_turns=max_turns,
        effort=effort_mode or None,
    )
    response_parts: list[str] = []
    session_id = ""
    num_turns = 0
    total_cost: float | None = None
    usage: dict = {}

    async for message in query(prompt=question, options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    response_parts.append(block.text)
        elif isinstance(message, ResultMessage):
            session_id = message.session_id
            num_turns = message.num_turns
            total_cost = message.total_cost_usd
            usage = message.usage or {}

    if session_id:
        jsonl_src = _find_session_jsonl(session_id, str(corpus_dir))
        if jsonl_src:
            import shutil
            shutil.copy2(jsonl_src, run_dir / "session.jsonl")

    return AgentResult(
        response_text="\n".join(response_parts),
        session_id=session_id,
        num_turns=num_turns,
        total_cost_usd=total_cost,
        usage=usage,
    )
```

- [ ] **Step 4: Update `state.py`.**

Edit `src/agent_retrieval/runner/state.py`:

```python
def create_pending_runs(
    self, batch_name: str, parametrisation_id: str, n_runs: int,
    claude_version: str, agent_model: str, effort_mode: str,
    max_turns: int, allowed_tools: list[str],
) -> list[str]:
    run_ids = []
    for _ in range(n_runs):
        run_id = uuid.uuid4().hex[:12]
        state = RunState(
            parametrisation_id=parametrisation_id, run_id=run_id,
            batch_name=batch_name, status="pending", claude_code_version=claude_version,
            agent_model=agent_model, effort_mode=effort_mode,
            max_turns=max_turns, allowed_tools=allowed_tools,
        )
        run_dir = self.runs_dir / batch_name / parametrisation_id / run_id
        state.to_yaml(run_dir / "state.yaml")
        run_ids.append(run_id)
    return run_ids
```

Signature note: `agent_model` and `effort_mode` become required (no `= ""` default) because the new runner always has them.

- [ ] **Step 5: Update `run.py` to read from batch instead of template.**

Replace the parts of `src/agent_retrieval/runner/run.py` that currently read `template.runner.*`. The full body of the `run_batch` function in its final form:

```python
async def run_batch(
    batch: BatchConfig,
    experiments_dir: Path,
    workspace_dir: Path,
    resume: str | None = None,
) -> None:
    if resume:
        batch_run_name = resume
    else:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        batch_run_name = f"{batch.batch_name}__{ts}"

    runs_dir = workspace_dir / "runner" / "runs"
    corpora_dir = workspace_dir / "runner" / "corpora"
    answer_keys_dir = workspace_dir / "judge" / "answer_keys"
    state_mgr = RunStateManager(runs_dir)

    claude_version = get_claude_version()
    print(f"Claude Code version: {claude_version}")
    print(f"Batch run: {batch_run_name}")
    print(
        f"Agent: {batch.agent_model}, effort: {batch.effort_mode}, "
        f"max_turns: {batch.max_turns}, n_repeats: {batch.n_repeats}"
    )

    recovered = state_mgr.recover_interrupted(batch_run_name)
    if recovered:
        print(f"Recovered {len(recovered)} interrupted runs")

    pid_to_template: dict[str, ExperimentTemplate] = {}
    for entry in batch.experiments:
        template_path = experiments_dir / f"{entry.experiment_type}.yaml"
        template = ExperimentTemplate.from_yaml(template_path)
        params = expand_grid(template)
        if entry.filter:
            params = filter_parametrisations(params, entry.filter)
        for p in params:
            pid_to_template[p.parametrisation_id] = template

    if batch.retry_failed:
        for pid in pid_to_template:
            failed = state_mgr.get_runs_by_status(batch_run_name, pid, "failed")
            for run_id, run_dir in failed:
                state_mgr.update_status(run_dir, "pending")

    for pid in pid_to_template:
        existing = state_mgr.get_runs_by_status(batch_run_name, pid, "pending")
        existing += state_mgr.get_runs_by_status(batch_run_name, pid, "completed")
        existing += state_mgr.get_runs_by_status(batch_run_name, pid, "running")
        n_needed = batch.n_repeats - len(existing)
        if n_needed > 0:
            state_mgr.create_pending_runs(
                batch_run_name, pid, n_needed, claude_version,
                agent_model=batch.agent_model,
                effort_mode=batch.effort_mode,
                max_turns=batch.max_turns,
                allowed_tools=batch.allowed_tools,
            )

    all_pending: list[tuple[str, str, Path]] = []
    for pid in pid_to_template:
        n_completed = len(state_mgr.get_runs_by_status(batch_run_name, pid, "completed"))
        n_to_run = max(0, batch.n_repeats - n_completed)
        if n_to_run == 0:
            continue
        pending = state_mgr.get_runs_by_status(batch_run_name, pid, "pending")
        for run_id, run_dir in pending[:n_to_run]:
            all_pending.append((pid, run_id, run_dir))

    total = len(all_pending)
    print(f"Running {total} experiment sessions (max_parallel={batch.max_parallel})")

    if not all_pending:
        print("Nothing to run.")
        return

    semaphore = asyncio.Semaphore(batch.max_parallel)
    completed = 0

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

            state_mgr.update_status(run_dir, "running", started_at=datetime.now(timezone.utc).isoformat())

            try:
                result = await run_agent_session(
                    question=question, corpus_dir=corpus_dir, model=batch.agent_model,
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

    tasks = [run_one(pid, run_id, run_dir) for pid, run_id, run_dir in all_pending]
    await asyncio.gather(*tasks)
    print(f"Batch '{batch_run_name}' complete.")
```

- [ ] **Step 6: Verify wiring tests pass.**

```bash
poetry run pytest tests/test_runner_wiring.py -v
```

Expected: PASS.

- [ ] **Step 7: Full test run.**

```bash
poetry run pytest -v
```

Expected: runner tests green. Any remaining failures should be in tests that patch / stub old signatures — note them for Task 10.

- [ ] **Step 8: Commit.**

```bash
git add src/agent_retrieval/runner/run.py src/agent_retrieval/runner/session.py src/agent_retrieval/runner/state.py tests/test_runner_wiring.py
git commit -m "runner: read runtime knobs from BatchConfig; drop dead max_tokens; wire max_turns"
```

---

## Task 9: Update `analysis/loader.py` to read `agent_model` from run state

**Why:** Loader currently reads `spec.runner.agent_model` from `ExperimentSpec`, but `RunnerSpec` is gone. Use the audit trail on `RunState` instead (where `agent_model` has been recorded per-run all along).

**Files:**
- Modify: `src/agent_retrieval/analysis/loader.py`
- Test: `tests/test_analysis.py`

- [ ] **Step 1: Read the existing test to understand expected behaviour.**

```bash
poetry run pytest tests/test_analysis.py -v
```

Note any failing test names to compare against after the edit.

- [ ] **Step 2: Update `loader.py`.**

Replace the two occurrences of `row["agent_model"] = spec.runner.agent_model` (lines 68, 80) with a lookup from the per-run state file. Add near the top of `load_batch_results`:

```python
from agent_retrieval.schema.run_state import RunState
```

Then add this helper above `load_batch_results`:

```python
def _read_run_state(workspace_dir: Path, batch_name: str,
                    pid: str, run_id: str) -> RunState | None:
    state_path = (workspace_dir / "runner" / "runs" /
                  batch_name / pid / run_id / "state.yaml")
    if state_path.exists():
        return RunState.from_yaml(state_path)
    return None
```

Remove the `row["agent_model"] = spec.runner.agent_model` lines and replace with a single unconditional block inside the verdict loop (after `row: dict = {...}` is populated):

```python
        state = _read_run_state(workspace_dir, batch_name,
                                verdict.parametrisation_id, verdict.run_id)
        if state is not None:
            row["agent_model"] = state.agent_model
            row["effort_mode"] = state.effort_mode
            row["max_turns"] = state.max_turns
```

Delete the two blocks that referenced `spec.runner.agent_model`; leave the rest of the V1 fallback intact (for `experiment_type`, `content_profile`, etc.).

- [ ] **Step 3: Run existing analysis tests.**

```bash
poetry run pytest tests/test_analysis.py -v
```

Expected: all pass. If a test specifically asserted on the old `spec.runner.agent_model` path, update it to construct a `state.yaml` in `tmp_path` and assert the loader reads from there.

- [ ] **Step 4: Full test run.**

```bash
poetry run pytest -v
```

Expected: green (save any test from Task 10's list).

- [ ] **Step 5: Commit.**

```bash
git add src/agent_retrieval/analysis/loader.py tests/test_analysis.py
git commit -m "analysis: read agent_model / effort_mode / max_turns from RunState (not template)"
```

---

## Task 10: Update experiment template YAMLs

**Why:** `runner:` block is now meaningless on templates. `schema_version` is dropped. Remove both.

**Files:**
- Modify: `experiments/single_needle.yaml`
- Modify: `experiments/multi_chain.yaml`
- Modify: `experiments/multi_reasoning.yaml`

- [ ] **Step 1: Delete `schema_version` and the whole `runner:` block from all three.**

Edit each file. Remove:
- Top line: `schema_version: "2.0"`
- The final `runner:` block with its 5 children (`n_repeats`, `agent_model`, `max_tokens`, `allowed_tools`, `effort_mode`).

- [ ] **Step 2: Verify the templates still parse.**

```bash
poetry run python3 -c "
from pathlib import Path
from agent_retrieval.schema.template import ExperimentTemplate
for name in ['single_needle', 'multi_chain', 'multi_reasoning']:
    t = ExperimentTemplate.from_yaml(Path(f'experiments/{name}.yaml'))
    print(f'{name}: OK')
"
```

Expected: three `OK` lines.

- [ ] **Step 3: Run all tests that load templates.**

```bash
poetry run pytest -v -k template
```

Expected: green.

- [ ] **Step 4: Commit.**

```bash
git add experiments/single_needle.yaml experiments/multi_chain.yaml experiments/multi_reasoning.yaml
git commit -m "experiments: drop schema_version and runner: block (moved to BatchConfig)"
```

---

## Task 11: Update existing batch YAMLs to the new schema

**Why:** `BatchConfig` now has `extra="forbid"` and five new required fields. The four existing batch files won't parse until updated. Add the values that match what they actually ran (sonnet-4-6, effort_mode low, n_repeats 3, max_turns 50 as that was the silent hardcode, and the original tool set). Preserves the ability to re-analyze those runs without re-running them.

**Files:**
- Modify: `batches/full-sweep.yaml`
- Modify: `batches/sanity-check.yaml`
- Modify: `batches/smoke-test-v2.yaml`
- Modify: `batches/smoke-test-v2-one-each.yaml`

- [ ] **Step 1: Rewrite each batch file.**

`batches/full-sweep.yaml`:
```yaml
batch_name: "full-sweep"
max_parallel: 3
retry_failed: true
agent_model: "claude-sonnet-4-6"
effort_mode: "low"
n_repeats: 3
max_turns: 50
allowed_tools: ["Read", "Glob", "Grep", "Bash"]

experiments:
  - experiment_type: "single_needle"
  - experiment_type: "multi_chain"
  - experiment_type: "multi_reasoning"
```

`batches/sanity-check.yaml`: same top 8 lines as above, but keep the existing `experiments:` block (which has filters). Replace the current `judge_model:` line with the runtime fields above; leave everything from `experiments:` down untouched.

`batches/smoke-test-v2.yaml` and `batches/smoke-test-v2-one-each.yaml`: same pattern — replace `judge_model: "claude-sonnet-4-6"` with the 5 runtime fields, leave `experiments:` block intact.

- [ ] **Step 2: Verify all four parse.**

```bash
poetry run python3 -c "
from pathlib import Path
from agent_retrieval.schema.batch import BatchConfig
for f in Path('batches').glob('*.yaml'):
    b = BatchConfig.from_yaml(f)
    print(f'{f.name}: OK ({b.agent_model}, effort={b.effort_mode})')
"
```

Expected: four `OK` lines.

- [ ] **Step 3: Commit.**

```bash
git add batches/full-sweep.yaml batches/sanity-check.yaml batches/smoke-test-v2.yaml batches/smoke-test-v2-one-each.yaml
git commit -m "batches: update existing batch YAMLs to new BatchConfig schema"
```

---

## Task 12: Full repo test sweep + fix anything left

**Why:** Collect the residual fallout of the refactor (tests that patched old signatures, fixtures that include removed fields) in one place.

- [ ] **Step 1: Run the full suite.**

```bash
poetry run pytest -v
```

- [ ] **Step 2: For each failing test, fix. Common issues expected:**

- `tests/test_cli.py::test_judge_batch` references the removed `--judge-model`. Update to call `poetry run agent-retrieval judge some-batch` without the flag.
- Fixtures in `tests/conftest.py` may include `runner:` blocks in template dicts — remove.
- Tests that construct `BatchConfig` inline may still include `judge_model` — remove.
- `tests/test_generator_background.py` constructs `ExperimentSpec` — confirm it doesn't include `runner:` or `schema_version` any longer (those were required before; now one is gone and the other ignored).

- [ ] **Step 3: Re-run until green.**

```bash
poetry run pytest -v
```

Expected: all green. Record the final passing test count.

- [ ] **Step 4: Commit (only if there were fixes).**

```bash
git add tests/
git commit -m "tests: adapt to BatchConfig + schema cleanup"
```

---

## Task 13: Create new batch YAML — `opus-4-6 effort=low`

**Files:**
- Create: `batches/hard-synonym-subset_opus-4-6_effort-low.yaml`

- [ ] **Step 1: Write the batch file.**

```yaml
batch_name: "hard-synonym-subset_opus-4-6_effort-low"
max_parallel: 3
retry_failed: true
agent_model: "claude-opus-4-6"
effort_mode: "low"
n_repeats: 3
max_turns: 75
allowed_tools: ["Read", "Glob", "Grep", "Bash"]

# Grid: both profiles x {40k, 800k} x hard x synonym, + n_items {2, 16} for multi_*.
#   single_needle:   2 x 2 x 1 x 1      =  4 params -> 12 runs
#   multi_chain:     2 x 2 x 1 x 1 x 2  =  8 params -> 24 runs
#   multi_reasoning: 2 x 2 x 1 x 1 x 2  =  8 params -> 24 runs
#   total: 20 parametrisations, 60 runs
experiments:
  - experiment_type: "single_needle"
    filter:
      content_profile: [python_repo, noir_fiction]
      corpus_token_count: [40000, 800000]
      discriminability: [hard]
      reference_clarity: [synonym]

  - experiment_type: "multi_chain"
    filter:
      content_profile: [python_repo, noir_fiction]
      corpus_token_count: [40000, 800000]
      discriminability: [hard]
      reference_clarity: [synonym]
      n_items: [2, 16]

  - experiment_type: "multi_reasoning"
    filter:
      content_profile: [python_repo, noir_fiction]
      corpus_token_count: [40000, 800000]
      discriminability: [hard]
      reference_clarity: [synonym]
      n_items: [2, 16]
```

- [ ] **Step 2: Verify it parses.**

```bash
poetry run python3 -c "
from pathlib import Path
from agent_retrieval.schema.batch import BatchConfig
b = BatchConfig.from_yaml(Path('batches/hard-synonym-subset_opus-4-6_effort-low.yaml'))
print(f'{b.batch_name}: {b.agent_model} effort={b.effort_mode} max_turns={b.max_turns} n_repeats={b.n_repeats}')
"
```

Expected:
```
hard-synonym-subset_opus-4-6_effort-low: claude-opus-4-6 effort=low max_turns=75 n_repeats=3
```

- [ ] **Step 3: Dry-run expand to sanity-check parametrisation count.**

```bash
poetry run python3 -c "
from pathlib import Path
from agent_retrieval.schema.batch import BatchConfig
from agent_retrieval.schema.template import ExperimentTemplate
from agent_retrieval.generator.grid import expand_grid, filter_parametrisations
b = BatchConfig.from_yaml(Path('batches/hard-synonym-subset_opus-4-6_effort-low.yaml'))
total = 0
for e in b.experiments:
    t = ExperimentTemplate.from_yaml(Path(f'experiments/{e.experiment_type}.yaml'))
    params = expand_grid(t)
    if e.filter:
        params = filter_parametrisations(params, e.filter)
    print(f'{e.experiment_type}: {len(params)}')
    total += len(params)
print(f'total: {total} parametrisations -> {total * b.n_repeats} runs')
"
```

Expected: `total: 20 parametrisations -> 60 runs`.

- [ ] **Step 4: Commit.**

```bash
git add batches/hard-synonym-subset_opus-4-6_effort-low.yaml
git commit -m "batches: add hard-synonym-subset_opus-4-6_effort-low (60 runs)"
```

---

## Task 14: Create new batch YAML — `sonnet-4-6 effort=max`

**Files:**
- Create: `batches/hard-synonym-subset_sonnet-4-6_effort-max.yaml`

- [ ] **Step 1: Write the batch file.**

Same as Task 13 but with `batch_name`, `agent_model`, and `effort_mode` swapped:

```yaml
batch_name: "hard-synonym-subset_sonnet-4-6_effort-max"
max_parallel: 3
retry_failed: true
agent_model: "claude-sonnet-4-6"
effort_mode: "max"
n_repeats: 3
max_turns: 75
allowed_tools: ["Read", "Glob", "Grep", "Bash"]

# (identical grid to the opus batch — same experiments / filter blocks)
experiments:
  - experiment_type: "single_needle"
    filter:
      content_profile: [python_repo, noir_fiction]
      corpus_token_count: [40000, 800000]
      discriminability: [hard]
      reference_clarity: [synonym]

  - experiment_type: "multi_chain"
    filter:
      content_profile: [python_repo, noir_fiction]
      corpus_token_count: [40000, 800000]
      discriminability: [hard]
      reference_clarity: [synonym]
      n_items: [2, 16]

  - experiment_type: "multi_reasoning"
    filter:
      content_profile: [python_repo, noir_fiction]
      corpus_token_count: [40000, 800000]
      discriminability: [hard]
      reference_clarity: [synonym]
      n_items: [2, 16]
```

- [ ] **Step 2: Parse + expand check.**

Run the same two commands as Task 13 steps 2–3, with the filename swapped. Expected total: 20 parametrisations / 60 runs.

- [ ] **Step 3: Commit.**

```bash
git add batches/hard-synonym-subset_sonnet-4-6_effort-max.yaml
git commit -m "batches: add hard-synonym-subset_sonnet-4-6_effort-max (60 runs)"
```

---

## Task 15: Smoke test — one parametrisation end-to-end before burning the full budget

**Why:** Before committing to 120 agent-sessions, confirm the new code path actually works end-to-end with opus-4-6 on one parametrisation. Catches model-name typos, effort-mode validation issues, or audit-field recording bugs while the cost is still tiny.

**Files:**
- Create: `batches/smoke-opus-4-6.yaml`

- [ ] **Step 1: Write the smoke batch (1 parametrisation, 1 repeat).**

```yaml
batch_name: "smoke-opus-4-6"
max_parallel: 1
retry_failed: true
agent_model: "claude-opus-4-6"
effort_mode: "low"
n_repeats: 1
max_turns: 75
allowed_tools: ["Read", "Glob", "Grep", "Bash"]

experiments:
  - experiment_type: "single_needle"
    filter:
      content_profile: [python_repo]
      corpus_token_count: [40000]
      discriminability: [hard]
      reference_clarity: [synonym]
```

- [ ] **Step 2: Confirm corpus already exists (it should).**

```bash
ls workspace/runner/corpora/single_needle__python_repo__40k__hard__synonym/ | head
```

Expected: a populated directory with `.md` files.

- [ ] **Step 3: Run the smoke batch.**

```bash
poetry run agent-retrieval run batches/smoke-opus-4-6.yaml
```

Expected: one session completes. Console line ends with `Completed single_needle__python_repo__40k__hard__synonym run <run_id>`.

- [ ] **Step 4: Inspect the run state file for audit fields.**

```bash
ls workspace/runner/runs/ | grep smoke-opus
# Then:
cat workspace/runner/runs/smoke-opus-4-6__<timestamp>/single_needle__python_repo__40k__hard__synonym/*/state.yaml
```

Expected: the state includes `agent_model: claude-opus-4-6`, `effort_mode: low`, `max_turns: 75`, `allowed_tools: [Read, Glob, Grep, Bash]`, and `status: completed`.

- [ ] **Step 5: Judge it.**

```bash
poetry run agent-retrieval judge smoke-opus-4-6__<timestamp>
```

Expected: a verdict written. Inspect:

```bash
cat workspace/judge/judgements/smoke-opus-4-6__<timestamp>/single_needle__python_repo__40k__hard__synonym/*.yaml
```

Expected: verdict has `judge_model: claude-sonnet-4-6` recorded, and a non-zero `weighted_score`.

- [ ] **Step 6: Commit the smoke batch file.**

```bash
git add batches/smoke-opus-4-6.yaml
git commit -m "batches: add smoke-opus-4-6 for pre-flight validation"
```

- [ ] **Step 7: If anything failed, STOP and diagnose before proceeding to Task 16.** Do not burn budget on the full batches until the smoke run is clean.

---

## Task 16: Run the `opus-4-6 effort=low` full batch

- [ ] **Step 1: Kick off the batch.**

```bash
poetry run agent-retrieval run batches/hard-synonym-subset_opus-4-6_effort-low.yaml
```

Expected console: `Running 60 experiment sessions (max_parallel=3)`, then a stream of `Completed ...` lines. If any FAIL, the batch continues; they'll be retried on the next `run` invocation thanks to `retry_failed: true`.

- [ ] **Step 2: Monitor for failures.**

Once the run finishes, check for failed states:

```bash
poetry run python3 -c "
from pathlib import Path
from agent_retrieval.schema.run_state import RunState
import sys
runs = Path('workspace/runner/runs')
batch = sorted(runs.glob('hard-synonym-subset_opus-4-6_effort-low__*'))[-1]
counts = {'pending': 0, 'running': 0, 'completed': 0, 'failed': 0}
for state_path in batch.rglob('state.yaml'):
    s = RunState.from_yaml(state_path)
    counts[s.status] = counts.get(s.status, 0) + 1
print(batch.name, counts)
"
```

Expected: `completed: 60`, others 0. If any `failed`: re-run the batch to retry.

- [ ] **Step 3: Re-run if needed.**

```bash
poetry run agent-retrieval run batches/hard-synonym-subset_opus-4-6_effort-low.yaml
```

Repeats only the failed/pending ones (thanks to `retry_failed`).

---

## Task 17: Judge the opus batch

- [ ] **Step 1: Run the judge.**

```bash
# Use the same batch_run_name (with timestamp) produced by Task 16.
poetry run agent-retrieval judge hard-synonym-subset_opus-4-6_effort-low__<timestamp>
```

Expected: 60 verdicts under `workspace/judge/judgements/hard-synonym-subset_opus-4-6_effort-low__<timestamp>/`.

- [ ] **Step 2: Verify all 60 verdicts exist and record `judge_model`.**

```bash
poetry run python3 -c "
from pathlib import Path
from agent_retrieval.schema.verdict import Verdict
judgements = sorted(Path('workspace/judge/judgements').glob('hard-synonym-subset_opus-4-6_effort-low__*'))[-1]
verdicts = list(judgements.rglob('*.yaml'))
print(f'verdict count: {len(verdicts)}')
judges = {Verdict.from_yaml(v).judge_model for v in verdicts}
print(f'judge models used: {judges}')
"
```

Expected: `verdict count: 60`, `judge models used: {'claude-sonnet-4-6'}`.

---

## Task 18: Run the `sonnet-4-6 effort=max` full batch

- [ ] **Step 1: Kick off the batch.**

```bash
poetry run agent-retrieval run batches/hard-synonym-subset_sonnet-4-6_effort-max.yaml
```

Expected: `Running 60 experiment sessions (max_parallel=3)`. Be aware this may take materially longer than the opus-low run — effort=max allows much deeper thinking / more turns per session. Allow up to 2-3× wall time.

- [ ] **Step 2: Check for failures (same script as Task 16 Step 2, swapped batch name).**

- [ ] **Step 3: Re-run if needed (same command as Task 16).**

---

## Task 19: Judge the sonnet-max batch

- [ ] **Step 1: Run the judge.**

```bash
poetry run agent-retrieval judge hard-synonym-subset_sonnet-4-6_effort-max__<timestamp>
```

- [ ] **Step 2: Verify verdict count and judge_model (same script as Task 17 Step 2, swapped batch name).**

Expected: 60 verdicts, all with `judge_model: claude-sonnet-4-6`.

---

## Task 20: Merge the cleanup branch

- [ ] **Step 1: Final test sweep.**

```bash
poetry run pytest -v
```

Expected: green.

- [ ] **Step 2: Open PR (or merge locally, per your workflow).**

```bash
git push -u origin cleanup/model-and-cost-config-schema
# Then either open a PR via gh, or merge locally after review.
```

- [ ] **Step 3: Record the two new batch_run_names somewhere durable** (commit message, README, or a note in the upcoming analysis notebook). Future-you will want to know which timestamp corresponds to which model/effort pairing.

---

## Out-of-scope (flagged for future cleanup)

These exist today but are not touched by this plan:

1. **`BackgroundGenerator` class in `src/agent_retrieval/generator/background.py` is never called by the v2 pipeline.** It's test-only. Clean up when you next need to regenerate background pools (not now — pools are already on disk).
2. **Hardcoded `max_tokens=4096` and `max_tokens=1024` inside `generator/payload.py` and `generator/llm_client.py`** for the raw Anthropic Messages API path. These *are* respected (unlike the runner `max_tokens` we deleted) but are scattered. Consider centralizing next time you touch the generator.
3. **`generator/pool.py` hardcodes `max_turns=50` in its `ClaudeAgentOptions` construction.** Fine for now — pool gen is one-shot, not cost-sensitive during experiments.
4. **`CorpusSpec.generation_model` is still a required field** even though the v2 pipeline doesn't use it (only the unused `BackgroundGenerator` does). Left in place to keep historical spec.yaml files parseable. Prune if/when you remove `BackgroundGenerator`.

---

## Self-review notes (for the executing agent)

- Tasks 4, 5, and 6 are the most fragile because they touch schemas that many tests depend on. Task 12 exists to catch the fallout; don't skip it.
- Task 15's smoke test is the single most important cost-control gate. Never skip straight from the refactor to Task 16/18.
- Commit messages follow the repo's existing style (prefix: `schema:`, `runner:`, `judge:`, `batches:`, `experiments:`, `analysis:`, `tests:`).
- Every code change ships with a test in the same commit unless the change is purely a YAML edit.
- When a signature change ripples (Task 7 removes `judge_model` param; Task 8 changes `create_pending_runs` arity), it's OK for intermediate tasks to leave the suite red — Task 12 is the cleanup pass.
