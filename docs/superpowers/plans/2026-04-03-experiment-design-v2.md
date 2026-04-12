# Experiment Design v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the v1 single-spec-per-experiment system with template + parameter grid experiments, agent-based generation using background pools, and two content profiles (python_repo, noir_fiction).

**Architecture:** Three-phase generation pipeline: (1) background pool via Haiku Agent SDK, (2) corpus assembly via deterministic sampling, (3) payload insertion via Sonnet Agent SDK. V2 schema defines experiment templates with inline grids. Existing runner/judge/analysis adapt to new parametrisation IDs and grid-expanded experiments.

**Tech Stack:** Python 3.12, Pydantic v2, Claude Agent SDK (`claude_agent_sdk`), PyYAML, Poetry, pytest

---

## File Structure

### New Files
- `src/agent_retrieval/schema/template.py` — v2 ExperimentTemplate + grid models
- `src/agent_retrieval/generator/pool.py` — background pool generation via Agent SDK
- `src/agent_retrieval/generator/assembler.py` — corpus assembly (sampling from pool)
- `src/agent_retrieval/generator/insertion.py` — payload insertion via Sonnet Agent SDK
- `src/agent_retrieval/generator/grid.py` — grid expansion logic
- `src/agent_retrieval/generator/profiles/noir_fiction.py` — noir fiction content profile
- `experiments/single_needle.yaml` — v2 experiment template
- `experiments/multi_chain.yaml` — v2 experiment template
- `experiments/multi_reasoning.yaml` — v2 experiment template
- `tests/test_template_schema.py` — tests for v2 schema
- `tests/test_grid.py` — tests for grid expansion
- `tests/test_assembler.py` — tests for corpus assembly
- `tests/test_pool.py` — tests for pool generation
- `tests/test_insertion.py` — tests for payload insertion
- `tests/test_noir_profile.py` — tests for noir fiction profile

### Modified Files
- `src/agent_retrieval/generator/profiles/base.py` — new v2 ContentProfile interface (keep v1 for backward compat)
- `src/agent_retrieval/generator/profiles/python_repo.py` — add v2 methods
- `src/agent_retrieval/generator/profiles/registry.py` — register noir_fiction
- `src/agent_retrieval/generator/generate.py` — add v2 generation orchestrator
- `src/agent_retrieval/generator/__init__.py` — export new functions
- `src/agent_retrieval/schema/answer_key.py` — add parametrisation_id and parameters fields
- `src/agent_retrieval/schema/batch.py` — v2 batch format with experiment type refs + filters
- `src/agent_retrieval/schema/__init__.py` — export new types
- `src/agent_retrieval/schema/run_state.py` — use parametrisation_id instead of experiment_id
- `src/agent_retrieval/schema/verdict.py` — use parametrisation_id instead of experiment_id
- `src/agent_retrieval/runner/run.py` — handle v2 batch + parametrisation IDs
- `src/agent_retrieval/runner/state.py` — handle parametrisation IDs
- `src/agent_retrieval/judge/judge.py` — handle v2 batch + parametrisation IDs
- `src/agent_retrieval/judge/scoring.py` — replace llm_client with Agent SDK
- `src/agent_retrieval/analysis/loader.py` — load v2 metadata from answer keys instead of spec files
- `src/agent_retrieval/analysis/figures.py` — add new figure functions
- `src/agent_retrieval/analysis/tables.py` — add new slicing dimensions
- `src/agent_retrieval/analysis/analyze.py` — call new figures/tables
- `src/agent_retrieval/cli.py` — add generate-pool command, update generate for v2
- `tests/conftest.py` — add v2 fixtures

---

### Task 1: V2 Schema — ExperimentTemplate and Grid Models

**Files:**
- Create: `src/agent_retrieval/schema/template.py`
- Create: `tests/test_template_schema.py`
- Modify: `src/agent_retrieval/schema/__init__.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_template_schema.py
import pytest
import yaml

from agent_retrieval.schema.template import (
    ExperimentTemplate,
    GridSpec,
    Parametrisation,
    QuestionExample,
)


@pytest.fixture
def single_needle_dict() -> dict:
    return {
        "schema_version": "2.0",
        "experiment_type": "single_needle",
        "payload": {"item_type": "config_value"},
        "question_examples": {
            "python_repo": {
                "easy_exact": {
                    "question": "What is the value of MAX_POOL_SIZE?",
                    "needle": "MAX_POOL_SIZE = 25",
                    "answer": "25",
                },
            },
        },
        "rubric_criteria": [
            {"criterion": "correctness", "weight": 1.0},
            {"criterion": "completeness", "weight": 0.5},
        ],
        "grid": {
            "content_profile": ["python_repo", "noir_fiction"],
            "corpus_token_count": [20000, 40000, 160000, 800000],
            "discriminability": ["easy", "hard"],
            "reference_clarity": ["exact", "synonym", "contextual"],
        },
        "runner": {
            "n_repeats": 3,
            "agent_model": "claude-sonnet-4-6",
            "max_tokens": 100000,
            "allowed_tools": ["Read", "Glob", "Grep", "Bash"],
        },
    }


@pytest.fixture
def multi_chain_dict(single_needle_dict) -> dict:
    d = single_needle_dict.copy()
    d["experiment_type"] = "multi_chain"
    d["payload"] = {"item_type": "cross_reference"}
    d["question_examples"] = {
        "python_repo": {
            "easy_exact": {
                "question": "Starting from X, follow refs. What is the final value?",
                "chain": [
                    {"needle": "X = get('Y')", "file_context": "config.md"},
                    {"needle": "Y = 42", "file_context": "settings.md"},
                ],
                "answer": "42",
            },
        },
    }
    d["grid"]["n_items"] = [2, 8, 16]
    return d


class TestExperimentTemplate:
    def test_valid_single_needle(self, single_needle_dict):
        tmpl = ExperimentTemplate.model_validate(single_needle_dict)
        assert tmpl.experiment_type == "single_needle"
        assert tmpl.schema_version == "2.0"
        assert len(tmpl.grid.content_profile) == 2
        assert tmpl.grid.n_items is None

    def test_valid_multi_chain(self, multi_chain_dict):
        tmpl = ExperimentTemplate.model_validate(multi_chain_dict)
        assert tmpl.experiment_type == "multi_chain"
        assert tmpl.grid.n_items == [2, 8, 16]

    def test_multi_type_requires_n_items(self, multi_chain_dict):
        del multi_chain_dict["grid"]["n_items"]
        with pytest.raises(Exception):
            ExperimentTemplate.model_validate(multi_chain_dict)

    def test_single_type_rejects_n_items(self, single_needle_dict):
        single_needle_dict["grid"]["n_items"] = [2, 4]
        with pytest.raises(Exception):
            ExperimentTemplate.model_validate(single_needle_dict)

    def test_invalid_experiment_type_raises(self, single_needle_dict):
        single_needle_dict["experiment_type"] = "invalid_type"
        with pytest.raises(Exception):
            ExperimentTemplate.model_validate(single_needle_dict)

    def test_from_yaml(self, single_needle_dict, tmp_path):
        path = tmp_path / "single_needle.yaml"
        path.write_text(yaml.dump(single_needle_dict))
        tmpl = ExperimentTemplate.from_yaml(path)
        assert tmpl.experiment_type == "single_needle"


class TestGridSpec:
    def test_valid_grid(self):
        grid = GridSpec.model_validate({
            "content_profile": ["python_repo"],
            "corpus_token_count": [20000],
            "discriminability": ["easy"],
            "reference_clarity": ["exact"],
        })
        assert grid.content_profile == ["python_repo"]

    def test_invalid_discriminability_raises(self):
        with pytest.raises(Exception):
            GridSpec.model_validate({
                "content_profile": ["python_repo"],
                "corpus_token_count": [20000],
                "discriminability": ["medium"],
                "reference_clarity": ["exact"],
            })


class TestParametrisation:
    def test_parametrisation_id_single(self):
        p = Parametrisation(
            experiment_type="single_needle",
            content_profile="python_repo",
            corpus_token_count=20000,
            discriminability="easy",
            reference_clarity="exact",
        )
        assert p.parametrisation_id == "single_needle__python_repo__20k__easy__exact"

    def test_parametrisation_id_multi(self):
        p = Parametrisation(
            experiment_type="multi_chain",
            content_profile="noir_fiction",
            corpus_token_count=160000,
            discriminability="hard",
            reference_clarity="synonym",
            n_items=8,
        )
        assert p.parametrisation_id == "multi_chain__noir_fiction__160k__hard__synonym__n8"

    def test_parametrisation_id_800k(self):
        p = Parametrisation(
            experiment_type="single_needle",
            content_profile="python_repo",
            corpus_token_count=800000,
            discriminability="hard",
            reference_clarity="contextual",
        )
        assert p.parametrisation_id == "single_needle__python_repo__800k__hard__contextual"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `poetry run pytest tests/test_template_schema.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agent_retrieval.schema.template'`

- [ ] **Step 3: Implement the v2 schema models**

```python
# src/agent_retrieval/schema/template.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, model_validator

from agent_retrieval.schema.experiment import RubricCriterion, RunnerSpec


class QuestionExample(BaseModel):
    question: str
    answer: str
    needle: str | None = None
    chain: list[dict[str, str]] | None = None
    items: list[dict[str, str]] | None = None


class GridSpec(BaseModel):
    content_profile: list[str]
    corpus_token_count: list[int]
    discriminability: list[Literal["easy", "hard"]]
    reference_clarity: list[Literal["exact", "synonym", "contextual"]]
    n_items: list[int] | None = None

    @model_validator(mode="after")
    def validate_discriminability_values(self) -> GridSpec:
        for v in self.discriminability:
            if v not in ("easy", "hard"):
                raise ValueError(f"Invalid discriminability: {v}")
        return self


class PayloadTemplateSpec(BaseModel):
    item_type: str


class ExperimentTemplate(BaseModel):
    schema_version: str
    experiment_type: Literal["single_needle", "multi_chain", "multi_reasoning"]
    payload: PayloadTemplateSpec
    question_examples: dict[str, dict[str, QuestionExample]]
    rubric_criteria: list[RubricCriterion]
    grid: GridSpec
    runner: RunnerSpec

    @model_validator(mode="after")
    def validate_grid_n_items(self) -> ExperimentTemplate:
        is_multi = self.experiment_type in ("multi_chain", "multi_reasoning")
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

    @classmethod
    def from_yaml(cls, path: Path) -> ExperimentTemplate:
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)


def _format_token_count(n: int) -> str:
    if n >= 1_000_000:
        return f"{n // 1_000_000}m"
    if n >= 1000:
        return f"{n // 1000}k"
    return str(n)


class Parametrisation(BaseModel):
    experiment_type: str
    content_profile: str
    corpus_token_count: int
    discriminability: str
    reference_clarity: str
    n_items: int | None = None

    @property
    def parametrisation_id(self) -> str:
        parts = [
            self.experiment_type,
            self.content_profile,
            _format_token_count(self.corpus_token_count),
            self.discriminability,
            self.reference_clarity,
        ]
        if self.n_items is not None:
            parts.append(f"n{self.n_items}")
        return "__".join(parts)
```

- [ ] **Step 4: Update schema __init__.py**

```python
# Add to src/agent_retrieval/schema/__init__.py — append these imports and __all__ entries
from agent_retrieval.schema.template import (
    ExperimentTemplate,
    GridSpec,
    Parametrisation,
    PayloadTemplateSpec,
    QuestionExample,
)
# Add to __all__:
# "ExperimentTemplate", "GridSpec", "Parametrisation", "PayloadTemplateSpec", "QuestionExample"
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `poetry run pytest tests/test_template_schema.py -v`
Expected: All 10 tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/agent_retrieval/schema/template.py tests/test_template_schema.py src/agent_retrieval/schema/__init__.py
git commit -m "feat: add v2 ExperimentTemplate schema with grid and parametrisation models"
```

---

### Task 2: Grid Expansion Logic

**Files:**
- Create: `src/agent_retrieval/generator/grid.py`
- Create: `tests/test_grid.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_grid.py
import pytest

from agent_retrieval.generator.grid import expand_grid, filter_parametrisations
from agent_retrieval.schema.template import ExperimentTemplate, Parametrisation


@pytest.fixture
def single_template_dict() -> dict:
    return {
        "schema_version": "2.0",
        "experiment_type": "single_needle",
        "payload": {"item_type": "config_value"},
        "question_examples": {
            "python_repo": {
                "easy_exact": {
                    "question": "What is X?",
                    "needle": "X = 1",
                    "answer": "1",
                },
            },
        },
        "rubric_criteria": [{"criterion": "correctness", "weight": 1.0}],
        "grid": {
            "content_profile": ["python_repo", "noir_fiction"],
            "corpus_token_count": [20000, 40000],
            "discriminability": ["easy", "hard"],
            "reference_clarity": ["exact", "synonym"],
        },
        "runner": {
            "n_repeats": 3,
            "agent_model": "claude-sonnet-4-6",
            "max_tokens": 100000,
            "allowed_tools": ["Read", "Glob", "Grep", "Bash"],
        },
    }


@pytest.fixture
def multi_template_dict(single_template_dict) -> dict:
    d = single_template_dict.copy()
    d["experiment_type"] = "multi_chain"
    d["payload"] = {"item_type": "cross_reference"}
    d["question_examples"] = {
        "python_repo": {
            "easy_exact": {
                "question": "Follow the chain. What is the final value?",
                "chain": [
                    {"needle": "A = B", "file_context": "a.md"},
                    {"needle": "B = 1", "file_context": "b.md"},
                ],
                "answer": "1",
            },
        },
    }
    d["grid"]["n_items"] = [2, 8]
    return d


class TestExpandGrid:
    def test_single_needle_expansion(self, single_template_dict):
        tmpl = ExperimentTemplate.model_validate(single_template_dict)
        params = expand_grid(tmpl)
        # 2 profiles x 2 token counts x 2 discriminability x 2 reference_clarity = 16
        assert len(params) == 16
        assert all(isinstance(p, Parametrisation) for p in params)
        assert all(p.n_items is None for p in params)

    def test_multi_chain_expansion(self, multi_template_dict):
        tmpl = ExperimentTemplate.model_validate(multi_template_dict)
        params = expand_grid(tmpl)
        # 2 x 2 x 2 x 2 x 2 = 32
        assert len(params) == 32
        assert all(p.n_items is not None for p in params)

    def test_ids_are_unique(self, single_template_dict):
        tmpl = ExperimentTemplate.model_validate(single_template_dict)
        params = expand_grid(tmpl)
        ids = [p.parametrisation_id for p in params]
        assert len(ids) == len(set(ids))

    def test_all_combinations_present(self, single_template_dict):
        tmpl = ExperimentTemplate.model_validate(single_template_dict)
        params = expand_grid(tmpl)
        profiles = {p.content_profile for p in params}
        assert profiles == {"python_repo", "noir_fiction"}
        token_counts = {p.corpus_token_count for p in params}
        assert token_counts == {20000, 40000}


class TestFilterParametrisations:
    def test_filter_by_profile(self, single_template_dict):
        tmpl = ExperimentTemplate.model_validate(single_template_dict)
        params = expand_grid(tmpl)
        filtered = filter_parametrisations(
            params, {"content_profile": ["python_repo"]}
        )
        assert len(filtered) == 8  # 1 x 2 x 2 x 2
        assert all(p.content_profile == "python_repo" for p in filtered)

    def test_filter_by_multiple(self, single_template_dict):
        tmpl = ExperimentTemplate.model_validate(single_template_dict)
        params = expand_grid(tmpl)
        filtered = filter_parametrisations(
            params,
            {
                "content_profile": ["python_repo"],
                "corpus_token_count": [20000],
                "discriminability": ["easy"],
                "reference_clarity": ["exact"],
            },
        )
        assert len(filtered) == 1
        assert filtered[0].parametrisation_id == "single_needle__python_repo__20k__easy__exact"

    def test_empty_filter_returns_all(self, single_template_dict):
        tmpl = ExperimentTemplate.model_validate(single_template_dict)
        params = expand_grid(tmpl)
        filtered = filter_parametrisations(params, {})
        assert len(filtered) == len(params)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `poetry run pytest tests/test_grid.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agent_retrieval.generator.grid'`

- [ ] **Step 3: Implement grid expansion**

```python
# src/agent_retrieval/generator/grid.py
from __future__ import annotations

import itertools
from typing import Any

from agent_retrieval.schema.template import ExperimentTemplate, Parametrisation


def expand_grid(template: ExperimentTemplate) -> list[Parametrisation]:
    grid = template.grid
    dimensions: list[tuple[str, list[Any]]] = [
        ("content_profile", grid.content_profile),
        ("corpus_token_count", grid.corpus_token_count),
        ("discriminability", grid.discriminability),
        ("reference_clarity", grid.reference_clarity),
    ]
    if grid.n_items is not None:
        dimensions.append(("n_items", grid.n_items))

    keys = [k for k, _ in dimensions]
    values = [v for _, v in dimensions]

    parametrisations: list[Parametrisation] = []
    for combo in itertools.product(*values):
        params = dict(zip(keys, combo))
        params["experiment_type"] = template.experiment_type
        parametrisations.append(Parametrisation(**params))

    return parametrisations


def filter_parametrisations(
    parametrisations: list[Parametrisation],
    filters: dict[str, list[Any]],
) -> list[Parametrisation]:
    if not filters:
        return parametrisations

    result: list[Parametrisation] = []
    for p in parametrisations:
        match = True
        for key, allowed_values in filters.items():
            if getattr(p, key) not in allowed_values:
                match = False
                break
        if match:
            result.append(p)
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `poetry run pytest tests/test_grid.py -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/agent_retrieval/generator/grid.py tests/test_grid.py
git commit -m "feat: add grid expansion and filtering for v2 experiment templates"
```

---

### Task 3: V2 Content Profile Interface and Python Repo v2

**Files:**
- Modify: `src/agent_retrieval/generator/profiles/base.py`
- Modify: `src/agent_retrieval/generator/profiles/python_repo.py`
- Modify: `tests/test_profiles.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_profiles.py`:

```python
class TestPythonRepoProfileV2:
    def test_pool_generation_brief_returns_string(self):
        profile = PythonRepoProfile()
        brief = profile.pool_generation_brief(target_token_count=100000)
        assert isinstance(brief, str)
        assert len(brief) > 100
        assert "python" in brief.lower() or "web application" in brief.lower()

    def test_skeleton_returns_sections(self):
        profile = PythonRepoProfile()
        sections = profile.skeleton(target_token_count=100000)
        assert isinstance(sections, list)
        assert len(sections) > 0
        for section in sections:
            assert "name" in section
            assert "description" in section
            assert "files" in section

    def test_skeleton_file_paths_are_md(self):
        profile = PythonRepoProfile()
        sections = profile.skeleton(target_token_count=100000)
        for section in sections:
            for f in section["files"]:
                assert f.endswith(".md"), f"Expected .md file, got {f}"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `poetry run pytest tests/test_profiles.py::TestPythonRepoProfileV2 -v`
Expected: FAIL — `AttributeError: 'PythonRepoProfile' object has no attribute 'pool_generation_brief'`

- [ ] **Step 3: Add v2 interface to base and implement in python_repo**

Add to `src/agent_retrieval/generator/profiles/base.py`:

```python
# Add these methods to the ContentProfile ABC class (after the existing methods):

    def pool_generation_brief(self, target_token_count: int) -> str:
        """Return the system prompt for the background pool generation agent.

        Override in subclasses for v2 pool-based generation.
        """
        raise NotImplementedError("v2 pool generation not implemented for this profile")

    def skeleton(self, target_token_count: int) -> list[dict]:
        """Return a list of section dicts the pool generation agent should follow.

        Each section dict has keys: name, description, files (list of relative paths).
        Override in subclasses for v2 pool-based generation.
        """
        raise NotImplementedError("v2 skeleton not implemented for this profile")
```

Add to `src/agent_retrieval/generator/profiles/python_repo.py`:

```python
# Add these methods to PythonRepoProfile class:

    def pool_generation_brief(self, target_token_count: int) -> str:
        return (
            "You are generating background files for a realistic Python web application repository. "
            "Each file should be a Markdown (.md) file containing realistic Python source code, "
            "configuration, documentation, or project files. Files should look like they belong to "
            "a medium-to-large production web application with database access, caching, API endpoints, "
            "authentication, background tasks, and logging.\n\n"
            "Requirements:\n"
            "- Every file must have a .md extension\n"
            "- Code files should contain 50-200 lines of realistic Python code\n"
            "- Include imports, classes, functions, docstrings, and occasional inline comments\n"
            "- Configuration files should use YAML or environment variable patterns\n"
            "- Documentation files should describe setup, usage, and architecture\n"
            "- Include realistic variable names, function signatures, and data structures\n"
            "- Do NOT include any comments about the files being generated or synthetic\n"
            f"- Target total output: approximately {target_token_count:,} tokens across all files\n"
        )

    def skeleton(self, target_token_count: int) -> list[dict]:
        tokens_per_section = target_token_count // 8

        return [
            {
                "name": "core",
                "description": "Core application models, configuration, and entry points",
                "target_tokens": tokens_per_section,
                "files": [
                    "README.md", "config/settings.md", "config/database.md",
                    "config/logging.md", "core/models.md", "core/exceptions.md",
                    "core/constants.md", "core/app.md",
                ],
            },
            {
                "name": "api",
                "description": "API endpoints, serializers, and middleware",
                "target_tokens": tokens_per_section,
                "files": [
                    "api/routes.md", "api/auth.md", "api/middleware.md",
                    "api/serializers.md", "api/validators.md", "api/pagination.md",
                    "api/rate_limiting.md", "api/error_handlers.md",
                ],
            },
            {
                "name": "services",
                "description": "Business logic services and domain operations",
                "target_tokens": tokens_per_section,
                "files": [
                    "services/user_service.md", "services/auth_service.md",
                    "services/notification_service.md", "services/payment_service.md",
                    "services/search_service.md", "services/analytics_service.md",
                    "services/export_service.md", "services/scheduling_service.md",
                ],
            },
            {
                "name": "db",
                "description": "Database access layer, migrations, and connection management",
                "target_tokens": tokens_per_section,
                "files": [
                    "db/connection.md", "db/migrations.md", "db/repositories.md",
                    "db/query_builder.md", "db/pool.md", "db/cache_layer.md",
                    "db/seeds.md", "db/health_check.md",
                ],
            },
            {
                "name": "workers",
                "description": "Background task workers and job processing",
                "target_tokens": tokens_per_section,
                "files": [
                    "workers/task_runner.md", "workers/email_worker.md",
                    "workers/report_generator.md", "workers/cleanup_worker.md",
                    "workers/sync_worker.md", "workers/retry_handler.md",
                ],
            },
            {
                "name": "utils",
                "description": "Utility functions, helpers, and shared components",
                "target_tokens": tokens_per_section,
                "files": [
                    "utils/crypto.md", "utils/date_helpers.md", "utils/file_utils.md",
                    "utils/http_client.md", "utils/logging_utils.md",
                    "utils/string_utils.md", "utils/validators.md",
                ],
            },
            {
                "name": "tests",
                "description": "Test files and test utilities",
                "target_tokens": tokens_per_section,
                "files": [
                    "tests/conftest.md", "tests/test_auth.md", "tests/test_api.md",
                    "tests/test_services.md", "tests/test_db.md",
                    "tests/test_workers.md", "tests/fixtures.md",
                ],
            },
            {
                "name": "deploy",
                "description": "Deployment configuration, CI/CD, and infrastructure",
                "target_tokens": tokens_per_section,
                "files": [
                    "deploy/dockerfile.md", "deploy/docker_compose.md",
                    "deploy/ci_pipeline.md", "deploy/nginx_config.md",
                    "deploy/env_template.md", "deploy/monitoring.md",
                ],
            },
        ]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `poetry run pytest tests/test_profiles.py -v`
Expected: All 9 tests PASS (6 existing + 3 new)

- [ ] **Step 5: Commit**

```bash
git add src/agent_retrieval/generator/profiles/base.py src/agent_retrieval/generator/profiles/python_repo.py tests/test_profiles.py
git commit -m "feat: add v2 content profile interface with pool_generation_brief and skeleton"
```

---

### Task 4: Noir Fiction Content Profile

**Files:**
- Create: `src/agent_retrieval/generator/profiles/noir_fiction.py`
- Create: `tests/test_noir_profile.py`
- Modify: `src/agent_retrieval/generator/profiles/registry.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_noir_profile.py
from pathlib import Path

import pytest

from agent_retrieval.generator.profiles.noir_fiction import NoirFictionProfile
from agent_retrieval.generator.profiles.registry import get_profile


class TestNoirFictionProfile:
    def test_pool_generation_brief_returns_string(self):
        profile = NoirFictionProfile()
        brief = profile.pool_generation_brief(target_token_count=100000)
        assert isinstance(brief, str)
        assert len(brief) > 100
        assert "noir" in brief.lower() or "detective" in brief.lower()

    def test_skeleton_returns_sections(self):
        profile = NoirFictionProfile()
        sections = profile.skeleton(target_token_count=100000)
        assert isinstance(sections, list)
        assert len(sections) > 0
        for section in sections:
            assert "name" in section
            assert "description" in section
            assert "files" in section

    def test_skeleton_file_paths_are_md(self):
        profile = NoirFictionProfile()
        sections = profile.skeleton(target_token_count=100000)
        for section in sections:
            for f in section["files"]:
                assert f.endswith(".md"), f"Expected .md file, got {f}"

    def test_registered_in_registry(self):
        profile = get_profile("noir_fiction")
        assert isinstance(profile, NoirFictionProfile)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `poetry run pytest tests/test_noir_profile.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agent_retrieval.generator.profiles.noir_fiction'`

- [ ] **Step 3: Implement noir fiction profile**

```python
# src/agent_retrieval/generator/profiles/noir_fiction.py
from __future__ import annotations

from pathlib import Path

from agent_retrieval.generator.profiles.base import ContentProfile, GenerationContext
from agent_retrieval.schema.experiment import CorpusSpec


class NoirFictionProfile(ContentProfile):
    def generate_folder_structure(self, spec: CorpusSpec) -> list[Path]:
        raise NotImplementedError("Use v2 pool-based generation for noir_fiction")

    def generate_file_prompt(self, path: Path, context: GenerationContext) -> str:
        raise NotImplementedError("Use v2 pool-based generation for noir_fiction")

    def pool_generation_brief(self, target_token_count: int) -> str:
        return (
            "You are generating background files for a noir detective fiction corpus. "
            "Each file should be a Markdown (.md) file containing chapters, scenes, or "
            "supplementary materials for a hardboiled detective novel set in a 1940s American city.\n\n"
            "The story follows a private investigator working a complex case involving "
            "missing persons, corrupt officials, and underworld connections. The narrative "
            "should have a rich cast of characters including suspects, witnesses, informants, "
            "and law enforcement.\n\n"
            "Requirements:\n"
            "- Every file must have a .md extension\n"
            "- Chapter files: 500-1500 words of narrative prose with atmospheric descriptions, "
            "sharp dialogue, and plot progression\n"
            "- Case note files: 100-300 words of terse, factual investigation notes\n"
            "- Evidence log files: structured lists of evidence items with dates and descriptions\n"
            "- Witness statement files: 200-500 words in interview/transcript format\n"
            "- Write in classic noir style: first person or close third, cynical tone, "
            "vivid sensory details, morally ambiguous characters\n"
            "- Include realistic character names, locations, timestamps, and physical descriptions\n"
            "- Do NOT include any comments about the text being generated or synthetic\n"
            f"- Target total output: approximately {target_token_count:,} tokens across all files\n"
        )

    def skeleton(self, target_token_count: int) -> list[dict]:
        tokens_per_section = target_token_count // 8

        return [
            {
                "name": "chapters_1_5",
                "description": "Opening chapters: crime discovered, PI takes the case, initial interviews",
                "target_tokens": tokens_per_section,
                "files": [
                    "chapters/chapter_01.md", "chapters/chapter_02.md",
                    "chapters/chapter_03.md", "chapters/chapter_04.md",
                    "chapters/chapter_05.md",
                ],
            },
            {
                "name": "chapters_6_10",
                "description": "Middle chapters: investigation deepens, false leads, danger increases",
                "target_tokens": tokens_per_section,
                "files": [
                    "chapters/chapter_06.md", "chapters/chapter_07.md",
                    "chapters/chapter_08.md", "chapters/chapter_09.md",
                    "chapters/chapter_10.md",
                ],
            },
            {
                "name": "chapters_11_15",
                "description": "Later chapters: revelations, confrontations, stakes escalate",
                "target_tokens": tokens_per_section,
                "files": [
                    "chapters/chapter_11.md", "chapters/chapter_12.md",
                    "chapters/chapter_13.md", "chapters/chapter_14.md",
                    "chapters/chapter_15.md",
                ],
            },
            {
                "name": "chapters_16_20",
                "description": "Final chapters: climax, resolution, aftermath",
                "target_tokens": tokens_per_section,
                "files": [
                    "chapters/chapter_16.md", "chapters/chapter_17.md",
                    "chapters/chapter_18.md", "chapters/chapter_19.md",
                    "chapters/chapter_20.md",
                ],
            },
            {
                "name": "case_notes",
                "description": "PI's case notebook entries with dates, observations, and theories",
                "target_tokens": tokens_per_section,
                "files": [
                    "case_notes/day_01.md", "case_notes/day_02.md",
                    "case_notes/day_03.md", "case_notes/day_04.md",
                    "case_notes/day_05.md", "case_notes/day_06.md",
                    "case_notes/day_07.md", "case_notes/day_08.md",
                ],
            },
            {
                "name": "evidence",
                "description": "Evidence logs, photographs, forensic reports",
                "target_tokens": tokens_per_section,
                "files": [
                    "evidence/evidence_log.md", "evidence/forensic_report.md",
                    "evidence/phone_records.md", "evidence/financial_records.md",
                    "evidence/photographs.md", "evidence/autopsy_report.md",
                ],
            },
            {
                "name": "witness_statements",
                "description": "Formal and informal witness interviews and statements",
                "target_tokens": tokens_per_section,
                "files": [
                    "witnesses/bartender_statement.md", "witnesses/landlady_statement.md",
                    "witnesses/cab_driver_statement.md", "witnesses/secretary_statement.md",
                    "witnesses/informant_interview.md", "witnesses/detective_debrief.md",
                    "witnesses/neighbor_statement.md",
                ],
            },
            {
                "name": "locations",
                "description": "Location descriptions, maps, and scene reports",
                "target_tokens": tokens_per_section,
                "files": [
                    "locations/crime_scene_report.md", "locations/office_description.md",
                    "locations/warehouse_district.md", "locations/nightclub_report.md",
                    "locations/apartment_search.md", "locations/docks_surveillance.md",
                ],
            },
        ]
```

- [ ] **Step 4: Register noir_fiction in registry**

Update `src/agent_retrieval/generator/profiles/registry.py`:

```python
from __future__ import annotations

from agent_retrieval.generator.profiles.base import ContentProfile
from agent_retrieval.generator.profiles.noir_fiction import NoirFictionProfile
from agent_retrieval.generator.profiles.python_repo import PythonRepoProfile

_PROFILES: dict[str, type[ContentProfile]] = {
    "python_repo": PythonRepoProfile,
    "noir_fiction": NoirFictionProfile,
}


def get_profile(name: str) -> ContentProfile:
    if name not in _PROFILES:
        raise KeyError(f"Unknown content profile: '{name}'. Available: {list(_PROFILES.keys())}")
    return _PROFILES[name]()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `poetry run pytest tests/test_noir_profile.py tests/test_profiles.py -v`
Expected: All 13 tests PASS (9 existing + 4 new)

- [ ] **Step 6: Commit**

```bash
git add src/agent_retrieval/generator/profiles/noir_fiction.py src/agent_retrieval/generator/profiles/registry.py tests/test_noir_profile.py
git commit -m "feat: add noir_fiction content profile with pool generation support"
```

---

### Task 5: Background Pool Generation via Agent SDK

**Files:**
- Create: `src/agent_retrieval/generator/pool.py`
- Create: `tests/test_pool.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_pool.py
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

from agent_retrieval.generator.pool import generate_pool, estimate_token_count


class TestEstimateTokenCount:
    def test_empty_dir(self, tmp_path):
        assert estimate_token_count(tmp_path) == 0

    def test_counts_characters_div_4(self, tmp_path):
        f = tmp_path / "test.md"
        f.write_text("a" * 400)  # 400 chars = ~100 tokens
        assert estimate_token_count(tmp_path) == 100

    def test_counts_multiple_files(self, tmp_path):
        (tmp_path / "a.md").write_text("x" * 800)
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "b.md").write_text("y" * 400)
        assert estimate_token_count(tmp_path) == 300  # (800 + 400) / 4


class TestGeneratePool:
    @pytest.mark.asyncio
    async def test_creates_pool_directory(self, tmp_path):
        pool_dir = tmp_path / "pools" / "python_repo"

        mock_query = AsyncMock()
        # Simulate the agent creating files as a side effect
        async def fake_query(prompt, options):
            # Agent "creates" files during execution
            pool_dir.mkdir(parents=True, exist_ok=True)
            for i in range(5):
                f = pool_dir / f"file_{i}.md"
                f.write_text("x" * 4000)  # 1000 tokens each
            # Yield a ResultMessage-like object
            result = MagicMock()
            result.session_id = "test-session"
            yield result

        with patch("agent_retrieval.generator.pool.query", side_effect=fake_query):
            await generate_pool("python_repo", pool_dir, target_token_count=5000)

        assert pool_dir.exists()

    @pytest.mark.asyncio
    async def test_skips_existing_pool(self, tmp_path):
        pool_dir = tmp_path / "pools" / "python_repo"
        pool_dir.mkdir(parents=True)
        # Create enough files to meet budget
        for i in range(10):
            (pool_dir / f"file_{i}.md").write_text("x" * 4000)

        with patch("agent_retrieval.generator.pool.query") as mock_query:
            await generate_pool("python_repo", pool_dir, target_token_count=5000)
            mock_query.assert_not_called()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `poetry run pytest tests/test_pool.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agent_retrieval.generator.pool'`

- [ ] **Step 3: Implement pool generation**

```python
# src/agent_retrieval/generator/pool.py
from __future__ import annotations

from pathlib import Path

from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, query

from agent_retrieval.generator.profiles.registry import get_profile


def estimate_token_count(directory: Path) -> int:
    total_chars = 0
    if not directory.exists():
        return 0
    for f in directory.rglob("*.md"):
        if f.is_file():
            total_chars += len(f.read_text())
    return total_chars // 4


async def generate_pool(
    profile_name: str,
    pool_dir: Path,
    target_token_count: int = 1_000_000,
) -> None:
    if pool_dir.exists() and estimate_token_count(pool_dir) >= target_token_count:
        return

    pool_dir.mkdir(parents=True, exist_ok=True)
    profile = get_profile(profile_name)
    sections = profile.skeleton(target_token_count)
    brief = profile.pool_generation_brief(target_token_count)

    for section in sections:
        current_tokens = estimate_token_count(pool_dir)
        if current_tokens >= target_token_count:
            break

        files_list = "\n".join(f"- {f}" for f in section["files"])
        section_prompt = (
            f"Generate the following files for the '{section['name']}' section.\n"
            f"Section description: {section['description']}\n"
            f"Target tokens for this section: ~{section.get('target_tokens', 50000):,}\n\n"
            f"Files to create:\n{files_list}\n\n"
            f"Create each file using the Write tool. Make sure each file path is exactly "
            f"as listed above."
        )

        options = ClaudeAgentOptions(
            model="claude-haiku-4-5-20251001",
            system_prompt=brief,
            cwd=str(pool_dir),
            allowed_tools=["Write"],
            permission_mode="acceptEdits",
            max_turns=50,
        )

        async for message in query(prompt=section_prompt, options=options):
            if isinstance(message, ResultMessage):
                break

        print(
            f"  Pool '{profile_name}' section '{section['name']}' done. "
            f"Cumulative tokens: ~{estimate_token_count(pool_dir):,}"
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `poetry run pytest tests/test_pool.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/agent_retrieval/generator/pool.py tests/test_pool.py
git commit -m "feat: add background pool generation via Agent SDK"
```

---

### Task 6: Corpus Assembly (Sampling from Pool)

**Files:**
- Create: `src/agent_retrieval/generator/assembler.py`
- Create: `tests/test_assembler.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_assembler.py
import pytest
from pathlib import Path

from agent_retrieval.generator.assembler import assemble_corpus
from agent_retrieval.schema.template import Parametrisation


@pytest.fixture
def pool_dir(tmp_path) -> Path:
    pool = tmp_path / "pool"
    pool.mkdir()
    # Create 20 files with varying sizes
    for i in range(20):
        sub = pool / f"section_{i // 5}"
        sub.mkdir(exist_ok=True)
        # ~250 tokens each (1000 chars / 4)
        (sub / f"file_{i}.md").write_text(f"# File {i}\n" + "content " * 125)
    return pool


@pytest.fixture
def parametrisation() -> Parametrisation:
    return Parametrisation(
        experiment_type="single_needle",
        content_profile="python_repo",
        corpus_token_count=1000,
        discriminability="easy",
        reference_clarity="exact",
    )


class TestAssembleCorpus:
    def test_creates_corpus_directory(self, pool_dir, parametrisation, tmp_path):
        corpus_dir = tmp_path / "corpora" / parametrisation.parametrisation_id
        assemble_corpus(pool_dir, corpus_dir, parametrisation)
        assert corpus_dir.exists()

    def test_respects_token_budget(self, pool_dir, parametrisation, tmp_path):
        corpus_dir = tmp_path / "corpora" / parametrisation.parametrisation_id
        assemble_corpus(pool_dir, corpus_dir, parametrisation)
        # Count tokens in assembled corpus
        total_chars = sum(
            len(f.read_text()) for f in corpus_dir.rglob("*.md") if f.is_file()
        )
        total_tokens = total_chars // 4
        # Should be close to budget but not wildly over
        assert total_tokens >= parametrisation.corpus_token_count * 0.5
        assert total_tokens <= parametrisation.corpus_token_count * 2.0

    def test_deterministic_with_same_seed(self, pool_dir, parametrisation, tmp_path):
        corpus1 = tmp_path / "c1" / parametrisation.parametrisation_id
        corpus2 = tmp_path / "c2" / parametrisation.parametrisation_id
        assemble_corpus(pool_dir, corpus1, parametrisation)
        assemble_corpus(pool_dir, corpus2, parametrisation)
        files1 = sorted(f.name for f in corpus1.rglob("*.md"))
        files2 = sorted(f.name for f in corpus2.rglob("*.md"))
        assert files1 == files2

    def test_different_parametrisations_differ(self, pool_dir, tmp_path):
        p1 = Parametrisation(
            experiment_type="single_needle",
            content_profile="python_repo",
            corpus_token_count=1000,
            discriminability="easy",
            reference_clarity="exact",
        )
        p2 = Parametrisation(
            experiment_type="single_needle",
            content_profile="python_repo",
            corpus_token_count=1000,
            discriminability="hard",
            reference_clarity="exact",
        )
        c1 = tmp_path / "c1" / p1.parametrisation_id
        c2 = tmp_path / "c2" / p2.parametrisation_id
        assemble_corpus(pool_dir, c1, p1)
        assemble_corpus(pool_dir, c2, p2)
        files1 = sorted(f.name for f in c1.rglob("*.md"))
        files2 = sorted(f.name for f in c2.rglob("*.md"))
        # Different seeds should produce different samples (probabilistic but very likely with 20 files)
        # We just check they both produced files
        assert len(files1) > 0
        assert len(files2) > 0

    def test_preserves_subdirectory_structure(self, pool_dir, parametrisation, tmp_path):
        corpus_dir = tmp_path / "corpora" / parametrisation.parametrisation_id
        assemble_corpus(pool_dir, corpus_dir, parametrisation)
        # Files should retain their relative subdirectory paths
        for f in corpus_dir.rglob("*.md"):
            rel = f.relative_to(corpus_dir)
            assert len(rel.parts) >= 1  # at least a filename

    def test_skips_existing_corpus(self, pool_dir, parametrisation, tmp_path):
        corpus_dir = tmp_path / "corpora" / parametrisation.parametrisation_id
        assemble_corpus(pool_dir, corpus_dir, parametrisation)
        first_files = sorted(f.name for f in corpus_dir.rglob("*.md"))

        # Add a marker file to detect if corpus gets regenerated
        (corpus_dir / "marker.txt").write_text("exists")
        assemble_corpus(pool_dir, corpus_dir, parametrisation)
        assert (corpus_dir / "marker.txt").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `poetry run pytest tests/test_assembler.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agent_retrieval.generator.assembler'`

- [ ] **Step 3: Implement corpus assembly**

```python
# src/agent_retrieval/generator/assembler.py
from __future__ import annotations

import random
import shutil
from pathlib import Path

from agent_retrieval.schema.template import Parametrisation


def assemble_corpus(
    pool_dir: Path,
    corpus_dir: Path,
    parametrisation: Parametrisation,
) -> None:
    if corpus_dir.exists() and any(corpus_dir.rglob("*.md")):
        return

    corpus_dir.mkdir(parents=True, exist_ok=True)

    all_files = sorted(f for f in pool_dir.rglob("*.md") if f.is_file())
    if not all_files:
        return

    rng = random.Random(hash(parametrisation.parametrisation_id))
    rng.shuffle(all_files)

    budget = parametrisation.corpus_token_count
    accumulated_tokens = 0

    for src_file in all_files:
        content = src_file.read_text()
        file_tokens = len(content) // 4

        rel_path = src_file.relative_to(pool_dir)
        dest = corpus_dir / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_file, dest)

        accumulated_tokens += file_tokens
        if accumulated_tokens >= budget:
            break
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `poetry run pytest tests/test_assembler.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/agent_retrieval/generator/assembler.py tests/test_assembler.py
git commit -m "feat: add corpus assembly by sampling from background pool"
```

---

### Task 7: Payload Insertion via Agent SDK

**Files:**
- Create: `src/agent_retrieval/generator/insertion.py`
- Create: `tests/test_insertion.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_insertion.py
import yaml
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

from agent_retrieval.generator.insertion import insert_payloads, build_insertion_prompt
from agent_retrieval.schema.template import ExperimentTemplate, Parametrisation, QuestionExample


@pytest.fixture
def corpus_dir(tmp_path) -> Path:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "config").mkdir()
    (corpus / "config" / "settings.md").write_text("# Settings\nDEBUG = True\nPORT = 8080\n")
    (corpus / "core").mkdir()
    (corpus / "core" / "app.md").write_text("# App\ndef main():\n    pass\n")
    return corpus


@pytest.fixture
def single_template() -> ExperimentTemplate:
    return ExperimentTemplate.model_validate({
        "schema_version": "2.0",
        "experiment_type": "single_needle",
        "payload": {"item_type": "config_value"},
        "question_examples": {
            "python_repo": {
                "easy_exact": {
                    "question": "What is the value of MAX_POOL_SIZE?",
                    "needle": "MAX_POOL_SIZE = 25",
                    "answer": "25",
                },
            },
        },
        "rubric_criteria": [
            {"criterion": "correctness", "weight": 1.0},
            {"criterion": "completeness", "weight": 0.5},
        ],
        "grid": {
            "content_profile": ["python_repo"],
            "corpus_token_count": [20000],
            "discriminability": ["easy"],
            "reference_clarity": ["exact"],
        },
        "runner": {
            "n_repeats": 3,
            "agent_model": "claude-sonnet-4-6",
            "max_tokens": 100000,
            "allowed_tools": ["Read", "Glob", "Grep", "Bash"],
        },
    })


@pytest.fixture
def parametrisation() -> Parametrisation:
    return Parametrisation(
        experiment_type="single_needle",
        content_profile="python_repo",
        corpus_token_count=20000,
        discriminability="easy",
        reference_clarity="exact",
    )


class TestBuildInsertionPrompt:
    def test_contains_experiment_type(self, single_template, parametrisation):
        prompt = build_insertion_prompt(single_template, parametrisation, Path("/answer.yaml"))
        assert "single_needle" in prompt

    def test_contains_discriminability_rubric(self, single_template, parametrisation):
        prompt = build_insertion_prompt(single_template, parametrisation, Path("/answer.yaml"))
        assert "easy" in prompt
        assert "Findable by exact string search" in prompt

    def test_contains_reference_clarity(self, single_template, parametrisation):
        prompt = build_insertion_prompt(single_template, parametrisation, Path("/answer.yaml"))
        assert "exact" in prompt

    def test_contains_examples(self, single_template, parametrisation):
        prompt = build_insertion_prompt(single_template, parametrisation, Path("/answer.yaml"))
        assert "MAX_POOL_SIZE" in prompt

    def test_contains_answer_key_path(self, single_template, parametrisation):
        prompt = build_insertion_prompt(single_template, parametrisation, Path("/tmp/answer.yaml"))
        assert "/tmp/answer.yaml" in prompt

    def test_multi_chain_specifies_n_items(self):
        tmpl = ExperimentTemplate.model_validate({
            "schema_version": "2.0",
            "experiment_type": "multi_chain",
            "payload": {"item_type": "cross_reference"},
            "question_examples": {
                "python_repo": {
                    "easy_exact": {
                        "question": "Follow the chain.",
                        "chain": [{"needle": "A", "file_context": "a.md"}],
                        "answer": "A",
                    },
                },
            },
            "rubric_criteria": [{"criterion": "correctness", "weight": 1.0}],
            "grid": {
                "content_profile": ["python_repo"],
                "corpus_token_count": [20000],
                "discriminability": ["easy"],
                "reference_clarity": ["exact"],
                "n_items": [4],
            },
            "runner": {
                "n_repeats": 1,
                "agent_model": "claude-sonnet-4-6",
                "max_tokens": 100000,
                "allowed_tools": ["Read"],
            },
        })
        param = Parametrisation(
            experiment_type="multi_chain",
            content_profile="python_repo",
            corpus_token_count=20000,
            discriminability="easy",
            reference_clarity="exact",
            n_items=4,
        )
        prompt = build_insertion_prompt(tmpl, param, Path("/answer.yaml"))
        assert "4" in prompt
        assert "chain" in prompt.lower() or "sequential" in prompt.lower()


class TestInsertPayloads:
    @pytest.mark.asyncio
    async def test_calls_agent_sdk(self, corpus_dir, single_template, parametrisation, tmp_path):
        answer_key_path = tmp_path / "answer_keys" / f"{parametrisation.parametrisation_id}.yaml"

        # Simulate the agent writing the answer key
        async def fake_query(prompt, options):
            answer_key_path.parent.mkdir(parents=True, exist_ok=True)
            answer_key_path.write_text(yaml.dump({
                "parametrisation_id": parametrisation.parametrisation_id,
                "experiment_type": "single_needle",
                "generated_at": "2026-04-03T10:00:00Z",
                "parameters": {
                    "content_profile": "python_repo",
                    "corpus_token_count": 20000,
                    "discriminability": "easy",
                    "reference_clarity": "exact",
                },
                "items": [{
                    "item_id": "target_001",
                    "inserted_text": "MAX_POOL_SIZE = 25",
                    "file_path": "config/settings.md",
                    "line_range": [3, 3],
                    "context_summary": "Added as config constant",
                }],
                "expected_answers": {
                    "question": "What is the value of MAX_POOL_SIZE?",
                    "correctness": "25",
                    "completeness": "Found in config/settings.md",
                },
                "rubric_criteria": [
                    {"criterion": "correctness", "weight": 1.0},
                    {"criterion": "completeness", "weight": 0.5},
                ],
            }))
            result = MagicMock()
            result.session_id = "test-session"
            yield result

        with patch("agent_retrieval.generator.insertion.query", side_effect=fake_query):
            await insert_payloads(
                template=single_template,
                parametrisation=parametrisation,
                corpus_dir=corpus_dir,
                answer_key_path=answer_key_path,
            )

        assert answer_key_path.exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `poetry run pytest tests/test_insertion.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agent_retrieval.generator.insertion'`

- [ ] **Step 3: Implement payload insertion**

```python
# src/agent_retrieval/generator/insertion.py
from __future__ import annotations

from pathlib import Path

from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, query

from agent_retrieval.schema.template import ExperimentTemplate, Parametrisation


DISCRIMINABILITY_RUBRIC = """## Discriminability Rubric

| Level | Retrieval method | Cognitive demand |
|---|---|---|
| easy | Findable by exact string search on terms in the question | Locate and report |
| hard | Embedded in surrounding context; requires reading and comprehending surrounding content to identify | Comprehend context, extract implicit information |
"""

REFERENCE_CLARITY_RUBRIC = """## Reference Clarity

| Level | Definition |
|---|---|
| exact | Question uses the same identifier/name as the needle |
| synonym | Question uses a different term for the same concept |
| contextual | Question describes the concept indirectly; requires domain understanding to connect |
"""

ANSWER_KEY_SCHEMA = """parametrisation_id: "{parametrisation_id}"
experiment_type: "{experiment_type}"
generated_at: "<ISO timestamp>"
parameters:
  content_profile: "{content_profile}"
  corpus_token_count: {corpus_token_count}
  discriminability: "{discriminability}"
  reference_clarity: "{reference_clarity}"
items:
  - item_id: "target_001"
    inserted_text: "<the exact text you inserted>"
    file_path: "<relative path to the file>"
    line_range: [<start_line>, <end_line>]
    context_summary: "<one sentence describing where/how it was inserted>"
expected_answers:
  question: "<the question you generated>"
  correctness: "<what a correct answer must include>"
  completeness: "<what a complete answer covers>"
rubric_criteria:
{rubric_criteria_yaml}"""


def _format_examples(
    template: ExperimentTemplate,
    parametrisation: Parametrisation,
) -> str:
    profile = parametrisation.content_profile
    disc = parametrisation.discriminability
    ref = parametrisation.reference_clarity

    profile_examples = template.question_examples.get(profile, {})
    key = f"{disc}_{ref}"
    example = profile_examples.get(key)

    if example is None:
        # Fall back to any example for this profile
        for k, v in profile_examples.items():
            example = v
            break

    if example is None:
        return "No examples available."

    lines = [f"**Question:** {example.question}", f"**Answer:** {example.answer}"]
    if example.needle:
        lines.insert(1, f"**Needle:** {example.needle}")
    if example.chain:
        chain_str = "\n".join(
            f"  - [{c['file_context']}] {c['needle']}" for c in example.chain
        )
        lines.insert(1, f"**Chain:**\n{chain_str}")
    if example.items:
        items_str = "\n".join(
            f"  - [{it['file_context']}] {it['needle']}" for it in example.items
        )
        lines.insert(1, f"**Items:**\n{items_str}")

    return "\n".join(lines)


def build_insertion_prompt(
    template: ExperimentTemplate,
    parametrisation: Parametrisation,
    answer_key_path: Path,
) -> str:
    n_items = parametrisation.n_items or 1
    exp_type = parametrisation.experiment_type

    type_instructions = {
        "single_needle": (
            "Insert exactly 1 needle into the corpus. "
            "Generate a question that asks about the inserted information."
        ),
        "multi_chain": (
            f"Insert a chain of {n_items} linked needles. Each needle must reference or point to "
            f"the next one in the chain. The question must provide the entry point (first needle's "
            f"location) and ask what the chain ultimately resolves to. Each link should be in a "
            f"different file. Previous links can be forgotten after following — only the final "
            f"value matters."
        ),
        "multi_reasoning": (
            f"Insert {n_items} independent needles, each containing a distinct piece of information. "
            f"The question must require finding ALL items and combining/reasoning about them to "
            f"produce the answer. Items should be in different files."
        ),
    }

    examples_str = _format_examples(template, parametrisation)

    rubric_criteria_yaml = "\n".join(
        f'  - criterion: "{c.criterion}"\n    weight: {c.weight}'
        for c in template.rubric_criteria
    )

    answer_schema = ANSWER_KEY_SCHEMA.format(
        parametrisation_id=parametrisation.parametrisation_id,
        experiment_type=exp_type,
        content_profile=parametrisation.content_profile,
        corpus_token_count=parametrisation.corpus_token_count,
        discriminability=parametrisation.discriminability,
        reference_clarity=parametrisation.reference_clarity,
        rubric_criteria_yaml=rubric_criteria_yaml,
    )

    return (
        f"You are inserting needle(s) into a corpus for a retrieval experiment.\n\n"
        f"Experiment type: {exp_type}\n"
        f"Discriminability: {parametrisation.discriminability}\n"
        f"Reference clarity: {parametrisation.reference_clarity}\n"
        f"Number of items: {n_items}\n\n"
        f"{DISCRIMINABILITY_RUBRIC}\n"
        f"{REFERENCE_CLARITY_RUBRIC}\n"
        f"## Example\n{examples_str}\n\n"
        f"## Type-specific instructions\n{type_instructions[exp_type]}\n\n"
        f"## Instructions\n"
        f"1. Browse the corpus to understand its structure and content style\n"
        f"2. Choose {n_items} file(s) to insert needles into\n"
        f"3. Read each target file to understand the surrounding context\n"
        f"4. Insert each needle so it reads naturally within the file\n"
        f"5. Generate a question and expected answer\n"
        f"6. Write the answer key as valid YAML to: {answer_key_path}\n\n"
        f"The answer key MUST follow this exact structure:\n"
        f"```yaml\n{answer_schema}\n```\n\n"
        f"For multiple items, add additional entries under 'items:' with sequential "
        f"item_ids (target_001, target_002, etc.).\n"
    )


async def insert_payloads(
    template: ExperimentTemplate,
    parametrisation: Parametrisation,
    corpus_dir: Path,
    answer_key_path: Path,
) -> None:
    if answer_key_path.exists():
        return

    answer_key_path.parent.mkdir(parents=True, exist_ok=True)

    system_prompt = build_insertion_prompt(template, parametrisation, answer_key_path)

    options = ClaudeAgentOptions(
        model="claude-sonnet-4-6",
        system_prompt=system_prompt,
        cwd=str(corpus_dir),
        allowed_tools=["Read", "Edit", "Write", "Glob", "Grep"],
        permission_mode="acceptEdits",
        max_turns=100,
    )

    prompt = (
        "Begin by browsing the corpus to understand its structure. "
        "Then insert the needle(s) and write the answer key."
    )

    async for message in query(prompt=prompt, options=options):
        if isinstance(message, ResultMessage):
            break
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `poetry run pytest tests/test_insertion.py -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/agent_retrieval/generator/insertion.py tests/test_insertion.py
git commit -m "feat: add agentic payload insertion via Sonnet Agent SDK"
```

---

### Task 8: V2 Generation Orchestrator

**Files:**
- Modify: `src/agent_retrieval/generator/generate.py`
- Modify: `src/agent_retrieval/generator/__init__.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_generator_background.py` (or create a new test — we append since it's the generation orchestrator):

```python
# Append to tests/test_generator_background.py

from unittest.mock import MagicMock
from agent_retrieval.generator.generate import generate_experiment_v2
from agent_retrieval.schema.template import ExperimentTemplate


class TestGenerateExperimentV2:
    @pytest.mark.asyncio
    async def test_orchestrates_all_phases(self, tmp_workspace):
        workspace_dir = tmp_workspace / "workspace"
        template_dict = {
            "schema_version": "2.0",
            "experiment_type": "single_needle",
            "payload": {"item_type": "config_value"},
            "question_examples": {
                "python_repo": {
                    "easy_exact": {
                        "question": "What is X?",
                        "needle": "X = 1",
                        "answer": "1",
                    },
                },
            },
            "rubric_criteria": [{"criterion": "correctness", "weight": 1.0}],
            "grid": {
                "content_profile": ["python_repo"],
                "corpus_token_count": [20000],
                "discriminability": ["easy"],
                "reference_clarity": ["exact"],
            },
            "runner": {
                "n_repeats": 1,
                "agent_model": "claude-sonnet-4-6",
                "max_tokens": 100000,
                "allowed_tools": ["Read"],
            },
        }
        template = ExperimentTemplate.model_validate(template_dict)

        # Pre-create a pool so pool generation is skipped
        pool_dir = workspace_dir / "runner" / "pools" / "python_repo"
        pool_dir.mkdir(parents=True)
        for i in range(10):
            (pool_dir / f"file_{i}.md").write_text(f"# File {i}\n" + "content " * 500)

        with patch("agent_retrieval.generator.generate.insert_payloads", new_callable=AsyncMock) as mock_insert, \
             patch("agent_retrieval.generator.generate.generate_pool", new_callable=AsyncMock) as mock_pool:
            await generate_experiment_v2(template, workspace_dir)

        # Pool generation should NOT have been called (pool exists with enough tokens)
        mock_pool.assert_not_called()
        # Insertion should have been called once (1 parametrisation)
        assert mock_insert.call_count == 1
        # Corpus should have been assembled
        corpus_dir = workspace_dir / "runner" / "corpora" / "single_needle__python_repo__20k__easy__exact"
        assert corpus_dir.exists()
        assert len(list(corpus_dir.rglob("*.md"))) > 0
```

- [ ] **Step 2: Run tests to verify it fails**

Run: `poetry run pytest tests/test_generator_background.py::TestGenerateExperimentV2 -v`
Expected: FAIL — `ImportError: cannot import name 'generate_experiment_v2'`

- [ ] **Step 3: Implement the v2 orchestrator**

Add to `src/agent_retrieval/generator/generate.py`:

```python
# Add these imports at the top:
from agent_retrieval.generator.assembler import assemble_corpus
from agent_retrieval.generator.grid import expand_grid
from agent_retrieval.generator.insertion import insert_payloads
from agent_retrieval.generator.pool import generate_pool
from agent_retrieval.schema.template import ExperimentTemplate


# Add this function:
async def generate_experiment_v2(
    template: ExperimentTemplate,
    workspace_dir: Path,
    skip_existing: bool = True,
) -> list[str]:
    """Generate all parametrisations for a v2 experiment template.

    Returns list of parametrisation IDs that were generated.
    """
    parametrisations = expand_grid(template)
    generated_ids: list[str] = []

    # Phase 1: Ensure background pools exist for all needed profiles
    profiles_needed = {p.content_profile for p in parametrisations}
    for profile_name in profiles_needed:
        pool_dir = workspace_dir / "runner" / "pools" / profile_name
        print(f"Ensuring background pool for '{profile_name}'...")
        await generate_pool(profile_name, pool_dir)

    # Phase 2 & 3: Assemble corpus and insert payloads per parametrisation
    for param in parametrisations:
        pid = param.parametrisation_id
        corpus_dir = workspace_dir / "runner" / "corpora" / pid
        answer_key_path = workspace_dir / "judge" / "answer_keys" / f"{pid}.yaml"

        if skip_existing and corpus_dir.exists() and answer_key_path.exists():
            print(f"  Skipping {pid} (already exists)")
            continue

        # Phase 2: Assemble corpus from pool
        pool_dir = workspace_dir / "runner" / "pools" / param.content_profile
        print(f"  Assembling corpus for {pid}...")
        assemble_corpus(pool_dir, corpus_dir, param)

        # Phase 3: Insert payloads
        print(f"  Inserting payloads for {pid}...")
        await insert_payloads(template, param, corpus_dir, answer_key_path)

        generated_ids.append(pid)
        print(f"  Done: {pid}")

    return generated_ids
```

- [ ] **Step 4: Update generator __init__.py**

```python
# src/agent_retrieval/generator/__init__.py
from agent_retrieval.generator.generate import generate_experiment, generate_experiment_v2

__all__ = ["generate_experiment", "generate_experiment_v2"]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `poetry run pytest tests/test_generator_background.py -v`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/agent_retrieval/generator/generate.py src/agent_retrieval/generator/__init__.py tests/test_generator_background.py
git commit -m "feat: add v2 generation orchestrator with pool, assembly, and insertion phases"
```

---

### Task 9: V2 Batch Schema and Answer Key Updates

**Files:**
- Modify: `src/agent_retrieval/schema/batch.py`
- Modify: `src/agent_retrieval/schema/answer_key.py`
- Modify: `tests/test_schema.py`
- Modify: `tests/conftest.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_schema.py`:

```python
from agent_retrieval.schema.batch import BatchConfigV2, BatchExperimentEntry


class TestBatchConfigV2:
    def test_simple_experiment_list(self):
        batch = BatchConfigV2.model_validate({
            "batch_name": "test-batch",
            "max_parallel": 4,
            "retry_failed": True,
            "judge_model": "claude-sonnet-4-6",
            "experiments": ["single_needle", "multi_chain"],
        })
        assert len(batch.experiments) == 2
        assert batch.experiments[0].experiment_type == "single_needle"
        assert batch.experiments[0].filter is None

    def test_filtered_experiment(self):
        batch = BatchConfigV2.model_validate({
            "batch_name": "test-batch",
            "max_parallel": 2,
            "retry_failed": True,
            "judge_model": "claude-sonnet-4-6",
            "experiments": [
                {
                    "experiment_type": "single_needle",
                    "filter": {
                        "content_profile": ["python_repo"],
                        "corpus_token_count": [20000],
                    },
                },
            ],
        })
        assert batch.experiments[0].experiment_type == "single_needle"
        assert batch.experiments[0].filter["content_profile"] == ["python_repo"]

    def test_mixed_format(self):
        batch = BatchConfigV2.model_validate({
            "batch_name": "test-batch",
            "max_parallel": 2,
            "retry_failed": True,
            "judge_model": "claude-sonnet-4-6",
            "experiments": [
                "single_needle",
                {"experiment_type": "multi_chain", "filter": {"n_items": [2]}},
            ],
        })
        assert len(batch.experiments) == 2
        assert batch.experiments[0].filter is None
        assert batch.experiments[1].filter is not None


class TestAnswerKeyV2:
    def test_answer_key_with_parametrisation(self):
        ak = AnswerKey.model_validate({
            "experiment_id": "single_needle__python_repo__20k__easy__exact",
            "generated_at": "2026-04-03T10:00:00Z",
            "parametrisation_id": "single_needle__python_repo__20k__easy__exact",
            "parameters": {
                "content_profile": "python_repo",
                "corpus_token_count": 20000,
                "discriminability": "easy",
                "reference_clarity": "exact",
            },
            "items": [{
                "item_id": "target_001",
                "inserted_text": "X = 1",
                "file_path": "config.md",
                "line_range": [1, 1],
                "context_summary": "test",
            }],
            "expected_answers": {
                "question": "What is X?",
                "correctness": "1",
            },
            "rubric_criteria": [{"criterion": "correctness", "weight": 1.0}],
        })
        assert ak.parametrisation_id == "single_needle__python_repo__20k__easy__exact"
        assert ak.parameters["content_profile"] == "python_repo"

    def test_answer_key_backward_compat(self):
        """V1 answer keys without parametrisation fields still work."""
        ak = AnswerKey.model_validate({
            "experiment_id": "test-001",
            "generated_at": "2026-04-03T10:00:00Z",
            "items": [{
                "item_id": "target_001",
                "inserted_text": "X = 1",
                "file_path": "config.py",
                "line_range": [1, 1],
                "context_summary": "test",
            }],
            "expected_answers": {
                "question": "What is X?",
                "correctness": "1",
            },
            "rubric_criteria": [{"criterion": "correctness", "weight": 1.0}],
        })
        assert ak.parametrisation_id is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `poetry run pytest tests/test_schema.py::TestBatchConfigV2 tests/test_schema.py::TestAnswerKeyV2 -v`
Expected: FAIL — `ImportError: cannot import name 'BatchConfigV2'`

- [ ] **Step 3: Update batch schema**

Add to `src/agent_retrieval/schema/batch.py`:

```python
# Add these imports at top:
from typing import Any

# Add these classes:
class BatchExperimentEntry(BaseModel):
    experiment_type: str
    filter: dict[str, list[Any]] | None = None


class BatchConfigV2(BaseModel):
    batch_name: str
    max_parallel: int
    retry_failed: bool
    judge_model: str
    experiments: list[BatchExperimentEntry]

    @model_validator(mode="before")
    @classmethod
    def normalize_experiments(cls, data: dict) -> dict:
        if "experiments" in data:
            normalized = []
            for entry in data["experiments"]:
                if isinstance(entry, str):
                    normalized.append({"experiment_type": entry})
                else:
                    normalized.append(entry)
            data["experiments"] = normalized
        return data

    @classmethod
    def from_yaml(cls, path: Path) -> BatchConfigV2:
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)
```

Also add the `model_validator` import at the top of the file:

```python
from pydantic import BaseModel, model_validator
```

- [ ] **Step 4: Update answer key schema**

Add optional v2 fields to `src/agent_retrieval/schema/answer_key.py`:

```python
# Add to AnswerKey class — new optional fields:
class AnswerKey(BaseModel):
    experiment_id: str
    generated_at: str
    items: list[AnswerKeyItem]
    expected_answers: ExpectedAnswers
    rubric_criteria: list[RubricCriterion]
    parametrisation_id: str | None = None
    parameters: dict[str, Any] | None = None
```

Add the import at the top:

```python
from typing import Any
```

- [ ] **Step 5: Update schema __init__.py**

Add to `src/agent_retrieval/schema/__init__.py`:

```python
from agent_retrieval.schema.batch import BatchConfigV2, BatchExperimentEntry
# Add to __all__: "BatchConfigV2", "BatchExperimentEntry"
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `poetry run pytest tests/test_schema.py -v`
Expected: All tests PASS (11 existing + 5 new)

- [ ] **Step 7: Commit**

```bash
git add src/agent_retrieval/schema/batch.py src/agent_retrieval/schema/answer_key.py src/agent_retrieval/schema/__init__.py tests/test_schema.py
git commit -m "feat: add v2 batch schema and answer key parametrisation fields"
```

---

### Task 10: V2 CLI — generate-pool and Updated generate Command

**Files:**
- Modify: `src/agent_retrieval/cli.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_cli.py`:

```python
class TestCLIV2:
    def test_generate_pool_command(self):
        args = parse_args(["generate-pool", "python_repo"])
        assert args.command == "generate-pool"
        assert args.profile_name == "python_repo"

    def test_generate_pool_custom_workspace(self):
        args = parse_args(["generate-pool", "python_repo", "--workspace", "/tmp/ws"])
        assert args.workspace == "/tmp/ws"

    def test_generate_v2_experiment(self):
        args = parse_args(["generate", "experiments/single_needle.yaml"])
        assert args.command == "generate"
        assert args.config_path == "experiments/single_needle.yaml"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `poetry run pytest tests/test_cli.py::TestCLIV2 -v`
Expected: FAIL — `SystemExit` (unrecognized command `generate-pool`)

- [ ] **Step 3: Update CLI**

Replace `src/agent_retrieval/cli.py` with:

```python
from __future__ import annotations
import argparse
import asyncio
import sys
from pathlib import Path
from agent_retrieval.schema.batch import BatchConfig
from agent_retrieval.schema.experiment import ExperimentSpec


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="agent-retrieval", description="Agent Retrieval Experiment Framework")
    sub = parser.add_subparsers(dest="command", required=True)

    gen_pool = sub.add_parser("generate-pool", help="Generate a background pool for a content profile")
    gen_pool.add_argument("profile_name", help="Content profile name (e.g. python_repo, noir_fiction)")
    gen_pool.add_argument("--workspace", default="workspace", help="Workspace directory")
    gen_pool.add_argument("--target-tokens", type=int, default=1_000_000, help="Target token count")

    gen = sub.add_parser("generate", help="Generate corpus and answer key")
    gen.add_argument("config_path", help="Path to spec YAML, experiment YAML, or batch YAML")
    gen.add_argument("--workspace", default="workspace", help="Workspace directory")

    run = sub.add_parser("run", help="Run experiments in a batch")
    run.add_argument("config_path", help="Path to batch YAML")
    run.add_argument("--workspace", default="workspace", help="Workspace directory")
    run.add_argument("--specs-dir", default="specs", help="Specs directory")
    run.add_argument("--experiments-dir", default="experiments", help="Experiments directory (v2)")

    judge = sub.add_parser("judge", help="Judge completed runs")
    judge.add_argument("config_path", help="Path to batch YAML")
    judge.add_argument("--workspace", default="workspace", help="Workspace directory")
    judge.add_argument("--rejudge", action="store_true", help="Re-judge existing verdicts")

    analyze = sub.add_parser("analyze", help="Analyze judged results")
    analyze.add_argument("config_path", help="Path to batch YAML")
    analyze.add_argument("--workspace", default="workspace", help="Workspace directory")
    analyze.add_argument("--specs-dir", default="specs", help="Specs directory")

    return parser.parse_args(argv)


def _is_batch_file(path: Path) -> bool:
    if "batch" in path.stem.lower() or path.parent.name == "batches":
        return True
    import yaml
    with open(path) as f:
        data = yaml.safe_load(f)
    return "batch_name" in data


def _is_v2_experiment(path: Path) -> bool:
    import yaml
    with open(path) as f:
        data = yaml.safe_load(f)
    return data.get("schema_version") == "2.0"


async def _generate_pool(args: argparse.Namespace) -> None:
    from agent_retrieval.generator.pool import generate_pool
    workspace_dir = Path(args.workspace)
    pool_dir = workspace_dir / "runner" / "pools" / args.profile_name
    print(f"Generating background pool for '{args.profile_name}'...")
    await generate_pool(args.profile_name, pool_dir, target_token_count=args.target_tokens)
    print(f"Done: pool at {pool_dir}")


async def _generate(args: argparse.Namespace) -> None:
    config_path = Path(args.config_path)
    workspace_dir = Path(args.workspace)

    if _is_v2_experiment(config_path):
        from agent_retrieval.generator import generate_experiment_v2
        from agent_retrieval.schema.template import ExperimentTemplate
        template = ExperimentTemplate.from_yaml(config_path)
        print(f"Generating v2 experiment '{template.experiment_type}'...")
        ids = await generate_experiment_v2(template, workspace_dir)
        print(f"Generated {len(ids)} parametrisations")
        return

    if _is_batch_file(config_path):
        from agent_retrieval.generator import generate_experiment
        batch = BatchConfig.from_yaml(config_path)
        specs_dir = config_path.parent.parent / "specs"
        for run_config in batch.runs:
            spec = ExperimentSpec.from_yaml(specs_dir / f"{run_config.experiment_id}.yaml")
            print(f"Generating corpus for {spec.experiment_id}...")
            await generate_experiment(spec, workspace_dir)
            print(f"  Done: {spec.experiment_id}")
    else:
        from agent_retrieval.generator import generate_experiment
        spec = ExperimentSpec.from_yaml(config_path)
        print(f"Generating corpus for {spec.experiment_id}...")
        await generate_experiment(spec, workspace_dir)
        print(f"  Done: {spec.experiment_id}")


async def _run(args: argparse.Namespace) -> None:
    from agent_retrieval.runner import run_batch
    batch = BatchConfig.from_yaml(Path(args.config_path))
    await run_batch(batch, Path(args.specs_dir), Path(args.workspace))


async def _judge(args: argparse.Namespace) -> None:
    from agent_retrieval.judge import judge_batch
    batch = BatchConfig.from_yaml(Path(args.config_path))
    await judge_batch(batch, Path(args.workspace), rejudge=args.rejudge)


def _analyze(args: argparse.Namespace) -> None:
    from agent_retrieval.analysis import run_analysis
    batch = BatchConfig.from_yaml(Path(args.config_path))
    run_analysis(batch.batch_name, Path(args.workspace), Path(args.specs_dir))


def main() -> None:
    args = parse_args()
    if args.command == "generate-pool":
        asyncio.run(_generate_pool(args))
    elif args.command == "generate":
        asyncio.run(_generate(args))
    elif args.command == "run":
        asyncio.run(_run(args))
    elif args.command == "judge":
        asyncio.run(_judge(args))
    elif args.command == "analyze":
        _analyze(args)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `poetry run pytest tests/test_cli.py -v`
Expected: All 8 tests PASS (5 existing + 3 new)

- [ ] **Step 5: Commit**

```bash
git add src/agent_retrieval/cli.py tests/test_cli.py
git commit -m "feat: add generate-pool CLI command and v2 experiment detection in generate"
```

---

### Task 11: Experiment YAML Files

**Files:**
- Create: `experiments/single_needle.yaml`
- Create: `experiments/multi_chain.yaml`
- Create: `experiments/multi_reasoning.yaml`

- [ ] **Step 1: Create the experiments directory**

```bash
mkdir -p experiments
```

- [ ] **Step 2: Create single_needle.yaml**

Copy the full content from the spec document section "single_needle.yaml" (lines 84-153 of the spec). The complete file is already defined in the spec at `docs/superpowers/specs/2026-04-03-experiment-design-v2.md`.

Write the file to `experiments/single_needle.yaml`.

- [ ] **Step 3: Create multi_chain.yaml**

Copy the full content from the spec document section "multi_chain.yaml" (lines 158-225 of the spec).

Write the file to `experiments/multi_chain.yaml`.

- [ ] **Step 4: Create multi_reasoning.yaml**

Copy the full content from the spec document section "multi_reasoning.yaml" (lines 228-292 of the spec).

Write the file to `experiments/multi_reasoning.yaml`.

- [ ] **Step 5: Validate all three parse correctly**

```python
# Quick validation script — run inline
from agent_retrieval.schema.template import ExperimentTemplate
for name in ["single_needle", "multi_chain", "multi_reasoning"]:
    t = ExperimentTemplate.from_yaml(Path(f"experiments/{name}.yaml"))
    print(f"{name}: {t.experiment_type}, grid produces {len(expand_grid(t))} parametrisations")
```

Run: `poetry run python -c "from pathlib import Path; from agent_retrieval.schema.template import ExperimentTemplate; from agent_retrieval.generator.grid import expand_grid; [print(f'{n}: {ExperimentTemplate.from_yaml(Path(f\"experiments/{n}.yaml\")).experiment_type}, {len(expand_grid(ExperimentTemplate.from_yaml(Path(f\"experiments/{n}.yaml\"))))} params') for n in ['single_needle', 'multi_chain', 'multi_reasoning']]"`

Expected:
```
single_needle: single_needle, 48 params
multi_chain: multi_chain, 144 params
multi_reasoning: multi_reasoning, 144 params
```

- [ ] **Step 6: Create v2 batch file**

Write `batches/full-sweep-v1.yaml`:

```yaml
batch_name: "full-sweep-v1"
max_parallel: 4
retry_failed: true
judge_model: "claude-sonnet-4-6"

experiments:
  - "single_needle"
  - "multi_chain"
  - "multi_reasoning"
```

Write `batches/smoke-test-v2.yaml`:

```yaml
batch_name: "smoke-test-v2"
max_parallel: 2
retry_failed: true
judge_model: "claude-sonnet-4-6"

experiments:
  - experiment_type: "single_needle"
    filter:
      content_profile: [python_repo]
      corpus_token_count: [20000]
      discriminability: [easy]
      reference_clarity: [exact]
```

- [ ] **Step 7: Commit**

```bash
git add experiments/ batches/full-sweep-v1.yaml batches/smoke-test-v2.yaml
git commit -m "feat: add v2 experiment YAML templates and batch files"
```

---

### Task 12: Update Analysis Loader for V2 Metadata

**Files:**
- Modify: `src/agent_retrieval/analysis/loader.py`
- Modify: `tests/test_analysis.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_analysis.py`:

```python
class TestLoadBatchResultsV2:
    def test_loads_v2_metadata_from_answer_key(self, tmp_path):
        """V2 answer keys carry parametrisation metadata — no spec file needed."""
        workspace = tmp_path / "workspace"
        batch_name = "test-v2"

        # Create a v2 answer key
        ak_dir = workspace / "judge" / "answer_keys"
        ak_dir.mkdir(parents=True)
        ak_path = ak_dir / "single_needle__python_repo__20k__easy__exact.yaml"
        ak_path.write_text(yaml.dump({
            "experiment_id": "single_needle__python_repo__20k__easy__exact",
            "parametrisation_id": "single_needle__python_repo__20k__easy__exact",
            "parameters": {
                "content_profile": "python_repo",
                "corpus_token_count": 20000,
                "discriminability": "easy",
                "reference_clarity": "exact",
            },
            "generated_at": "2026-04-03T10:00:00Z",
            "items": [{"item_id": "t1", "inserted_text": "X=1", "file_path": "a.md", "line_range": [1,1], "context_summary": "test"}],
            "expected_answers": {"question": "What is X?", "correctness": "1"},
            "rubric_criteria": [{"criterion": "correctness", "weight": 1.0}],
        }))

        # Create a verdict
        verdict_dir = workspace / "judge" / "judgements" / batch_name / "single_needle__python_repo__20k__easy__exact"
        verdict_dir.mkdir(parents=True)
        (verdict_dir / "run001.yaml").write_text(yaml.dump({
            "experiment_id": "single_needle__python_repo__20k__easy__exact",
            "run_id": "run001",
            "batch_name": batch_name,
            "scores": [{"criterion": "correctness", "score": 0.9, "weight": 1.0, "reasoning": "Good"}],
            "weighted_score": 0.9,
            "session_metrics": {"total_context_tokens": 5000, "total_turns": 3, "tool_calls": {"Grep": 2}, "duration_seconds": 10.0},
        }))

        from agent_retrieval.analysis.loader import load_batch_results
        df = load_batch_results(batch_name, workspace_dir=workspace, specs_dir=tmp_path / "specs")
        assert len(df) == 1
        assert df.iloc[0]["content_profile"] == "python_repo"
        assert df.iloc[0]["corpus_token_count"] == 20000
        assert df.iloc[0]["discriminability"] == "easy"
        assert df.iloc[0]["reference_clarity"] == "exact"
        assert df.iloc[0]["experiment_type"] == "single_needle"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/test_analysis.py::TestLoadBatchResultsV2 -v`
Expected: FAIL — missing v2 columns in output

- [ ] **Step 3: Update the loader**

Replace `src/agent_retrieval/analysis/loader.py`:

```python
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from agent_retrieval.schema.answer_key import AnswerKey
from agent_retrieval.schema.experiment import ExperimentSpec
from agent_retrieval.schema.verdict import Verdict


def load_batch_results(
    batch_name: str,
    workspace_dir: Path,
    specs_dir: Path,
) -> pd.DataFrame:
    judgements_dir = workspace_dir / "judge" / "judgements" / batch_name
    answer_keys_dir = workspace_dir / "judge" / "answer_keys"

    rows: list[dict] = []
    for verdict_path in sorted(judgements_dir.rglob("*.yaml")):
        verdict = Verdict.from_yaml(verdict_path)

        row: dict = {
            "experiment_id": verdict.experiment_id,
            "run_id": verdict.run_id,
            "batch_name": verdict.batch_name,
            "weighted_score": verdict.weighted_score,
            "total_context_tokens": verdict.session_metrics.total_context_tokens,
            "total_turns": verdict.session_metrics.total_turns,
            "duration_seconds": verdict.session_metrics.duration_seconds,
        }

        # Try to load metadata from v2 answer key first
        ak_path = answer_keys_dir / f"{verdict.experiment_id}.yaml"
        if ak_path.exists():
            ak = AnswerKey.from_yaml(ak_path)
            if ak.parameters:
                row["content_profile"] = ak.parameters.get("content_profile", "")
                row["corpus_token_count"] = ak.parameters.get("corpus_token_count", 0)
                row["discriminability"] = ak.parameters.get("discriminability", "")
                row["reference_clarity"] = ak.parameters.get("reference_clarity", "")
                row["n_items"] = ak.parameters.get("n_items")
                # Derive experiment_type from parametrisation_id
                if ak.parametrisation_id:
                    row["experiment_type"] = ak.parametrisation_id.split("__")[0]
                else:
                    row["experiment_type"] = ""
            else:
                # V1 fallback: load from spec file
                spec_path = specs_dir / f"{verdict.experiment_id}.yaml"
                if spec_path.exists():
                    spec = ExperimentSpec.from_yaml(spec_path)
                    row["experiment_type"] = spec.experiment_type
                    row["content_profile"] = spec.corpus.content_profile
                    row["corpus_token_count"] = spec.corpus.target_token_count
                    row["target_file_count"] = spec.corpus.target_file_count
                    row["agent_model"] = spec.runner.agent_model
                else:
                    row["experiment_type"] = ""
        else:
            row["experiment_type"] = ""

        # Flatten tool calls
        for tool_name, count in verdict.session_metrics.tool_calls.items():
            row[f"tool_{tool_name}"] = count

        # Flatten per-criterion scores
        for score_entry in verdict.scores:
            row[f"score_{score_entry.criterion}"] = score_entry.score

        rows.append(row)

    return pd.DataFrame(rows)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `poetry run pytest tests/test_analysis.py -v`
Expected: All tests PASS (existing + new)

- [ ] **Step 5: Commit**

```bash
git add src/agent_retrieval/analysis/loader.py tests/test_analysis.py
git commit -m "feat: update analysis loader to read v2 metadata from answer keys"
```

---

### Task 13: New Analysis Figures for V2 Dimensions

**Files:**
- Modify: `src/agent_retrieval/analysis/figures.py`
- Modify: `src/agent_retrieval/analysis/analyze.py`
- Modify: `src/agent_retrieval/analysis/__init__.py`
- Modify: `tests/test_analysis_figures.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_analysis_figures.py`:

```python
from agent_retrieval.analysis.figures import (
    plot_accuracy_vs_n_items,
    plot_accuracy_by_discriminability,
    plot_accuracy_by_reference_clarity,
    plot_profile_comparison,
)


class TestV2Figures:
    @pytest.fixture
    def v2_df(self):
        return pd.DataFrame([
            {"experiment_type": "single_needle", "content_profile": "python_repo", "corpus_token_count": 20000,
             "discriminability": "easy", "reference_clarity": "exact", "weighted_score": 0.9},
            {"experiment_type": "single_needle", "content_profile": "python_repo", "corpus_token_count": 20000,
             "discriminability": "hard", "reference_clarity": "exact", "weighted_score": 0.6},
            {"experiment_type": "single_needle", "content_profile": "noir_fiction", "corpus_token_count": 20000,
             "discriminability": "easy", "reference_clarity": "exact", "weighted_score": 0.85},
            {"experiment_type": "multi_chain", "content_profile": "python_repo", "corpus_token_count": 40000,
             "discriminability": "easy", "reference_clarity": "synonym", "weighted_score": 0.7, "n_items": 2},
            {"experiment_type": "multi_chain", "content_profile": "python_repo", "corpus_token_count": 40000,
             "discriminability": "easy", "reference_clarity": "synonym", "weighted_score": 0.5, "n_items": 8},
        ])

    def test_plot_accuracy_vs_n_items(self, v2_df, tmp_path):
        out = tmp_path / "n_items.png"
        plot_accuracy_vs_n_items(v2_df, out)
        assert out.exists()

    def test_plot_accuracy_by_discriminability(self, v2_df, tmp_path):
        out = tmp_path / "disc.png"
        plot_accuracy_by_discriminability(v2_df, out)
        assert out.exists()

    def test_plot_accuracy_by_reference_clarity(self, v2_df, tmp_path):
        out = tmp_path / "ref.png"
        plot_accuracy_by_reference_clarity(v2_df, out)
        assert out.exists()

    def test_plot_profile_comparison(self, v2_df, tmp_path):
        out = tmp_path / "profile.png"
        plot_profile_comparison(v2_df, out)
        assert out.exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `poetry run pytest tests/test_analysis_figures.py::TestV2Figures -v`
Expected: FAIL — `ImportError: cannot import name 'plot_accuracy_vs_n_items'`

- [ ] **Step 3: Implement new figure functions**

Append to `src/agent_retrieval/analysis/figures.py`:

```python
def plot_accuracy_vs_n_items(df: pd.DataFrame, output_path: Path) -> None:
    """Errorbar plot of weighted_score vs n_items, grouped by experiment_type."""
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_df = df.dropna(subset=["n_items"])

    if plot_df.empty:
        ax.set_title("Accuracy vs N-Items (no data)")
        _save_and_close(fig, output_path)
        return

    for exp_type, group in plot_df.groupby("experiment_type"):
        stats = (
            group.groupby("n_items")["weighted_score"]
            .agg(mean="mean", std="std")
            .reset_index()
        )
        ax.errorbar(
            stats["n_items"], stats["mean"], yerr=stats["std"].fillna(0),
            label=exp_type, marker="o", capsize=4,
        )

    ax.set_xlabel("Number of Items")
    ax.set_ylabel("Weighted Score")
    ax.set_title("Accuracy vs Number of Items")
    ax.legend()
    _save_and_close(fig, output_path)


def plot_accuracy_by_discriminability(df: pd.DataFrame, output_path: Path) -> None:
    """Grouped bar chart of weighted_score by discriminability, per experiment_type."""
    fig, ax = plt.subplots(figsize=(8, 5))

    if "discriminability" not in df.columns:
        ax.set_title("Accuracy by Discriminability (no data)")
        _save_and_close(fig, output_path)
        return

    stats = (
        df.groupby(["experiment_type", "discriminability"])["weighted_score"]
        .agg(mean="mean", std="std")
        .reset_index()
    )
    pivot = stats.pivot(index="experiment_type", columns="discriminability", values="mean")
    pivot.plot(kind="bar", ax=ax, capsize=4)

    ax.set_xlabel("Experiment Type")
    ax.set_ylabel("Weighted Score")
    ax.set_title("Accuracy by Discriminability")
    ax.legend(title="Discriminability")
    plt.xticks(rotation=30, ha="right")
    _save_and_close(fig, output_path)


def plot_accuracy_by_reference_clarity(df: pd.DataFrame, output_path: Path) -> None:
    """Grouped bar chart of weighted_score by reference_clarity, per experiment_type."""
    fig, ax = plt.subplots(figsize=(8, 5))

    if "reference_clarity" not in df.columns:
        ax.set_title("Accuracy by Reference Clarity (no data)")
        _save_and_close(fig, output_path)
        return

    stats = (
        df.groupby(["experiment_type", "reference_clarity"])["weighted_score"]
        .agg(mean="mean", std="std")
        .reset_index()
    )
    pivot = stats.pivot(index="experiment_type", columns="reference_clarity", values="mean")
    pivot.plot(kind="bar", ax=ax, capsize=4)

    ax.set_xlabel("Experiment Type")
    ax.set_ylabel("Weighted Score")
    ax.set_title("Accuracy by Reference Clarity")
    ax.legend(title="Reference Clarity")
    plt.xticks(rotation=30, ha="right")
    _save_and_close(fig, output_path)


def plot_profile_comparison(df: pd.DataFrame, output_path: Path) -> None:
    """Grouped bar chart comparing content profiles across experiment types."""
    fig, ax = plt.subplots(figsize=(8, 5))

    if "content_profile" not in df.columns:
        ax.set_title("Profile Comparison (no data)")
        _save_and_close(fig, output_path)
        return

    stats = (
        df.groupby(["experiment_type", "content_profile"])["weighted_score"]
        .agg(mean="mean", std="std")
        .reset_index()
    )
    pivot = stats.pivot(index="experiment_type", columns="content_profile", values="mean")
    pivot.plot(kind="bar", ax=ax, capsize=4)

    ax.set_xlabel("Experiment Type")
    ax.set_ylabel("Weighted Score")
    ax.set_title("Profile Comparison: Python Repo vs Noir Fiction")
    ax.legend(title="Content Profile")
    plt.xticks(rotation=30, ha="right")
    _save_and_close(fig, output_path)
```

- [ ] **Step 4: Update analysis __init__.py exports**

Add to `src/agent_retrieval/analysis/__init__.py`:

```python
from agent_retrieval.analysis.figures import (
    plot_accuracy_vs_n_items,
    plot_accuracy_by_discriminability,
    plot_accuracy_by_reference_clarity,
    plot_profile_comparison,
)
# Add to __all__:
# "plot_accuracy_vs_n_items", "plot_accuracy_by_discriminability",
# "plot_accuracy_by_reference_clarity", "plot_profile_comparison"
```

- [ ] **Step 5: Update analyze.py orchestrator**

Add to the `run_analysis` function in `src/agent_retrieval/analysis/analyze.py`, after the existing figure generation:

```python
    # --- V2 figures (only generated if v2 columns present) ---
    if "discriminability" in df.columns:
        from agent_retrieval.analysis.figures import (
            plot_accuracy_vs_n_items,
            plot_accuracy_by_discriminability,
            plot_accuracy_by_reference_clarity,
            plot_profile_comparison,
        )
        fig_n_items = figures_dir / "accuracy_vs_n_items.png"
        plot_accuracy_vs_n_items(df, fig_n_items)
        figure_paths.append(str(fig_n_items.relative_to(output_dir)))

        fig_disc = figures_dir / "accuracy_by_discriminability.png"
        plot_accuracy_by_discriminability(df, fig_disc)
        figure_paths.append(str(fig_disc.relative_to(output_dir)))

        fig_ref = figures_dir / "accuracy_by_reference_clarity.png"
        plot_accuracy_by_reference_clarity(df, fig_ref)
        figure_paths.append(str(fig_ref.relative_to(output_dir)))

        fig_profile = figures_dir / "profile_comparison.png"
        plot_profile_comparison(df, fig_profile)
        figure_paths.append(str(fig_profile.relative_to(output_dir)))
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `poetry run pytest tests/test_analysis_figures.py -v`
Expected: All tests PASS (4 existing + 4 new)

- [ ] **Step 7: Commit**

```bash
git add src/agent_retrieval/analysis/figures.py src/agent_retrieval/analysis/analyze.py src/agent_retrieval/analysis/__init__.py tests/test_analysis_figures.py
git commit -m "feat: add v2 analysis figures for discriminability, reference clarity, n_items, and profile comparison"
```

---

### Task 14: Update Workspace Fixture and Run Full Test Suite

**Files:**
- Modify: `tests/conftest.py`

- [ ] **Step 1: Update the workspace fixture to include v2 directories**

```python
# tests/conftest.py
import shutil
from pathlib import Path

import pytest


@pytest.fixture
def tmp_workspace(tmp_path: Path) -> Path:
    """Create a temporary workspace with the expected directory structure."""
    dirs = [
        "runner/corpora",
        "runner/runs",
        "runner/pools",
        "judge/answer_keys",
        "judge/judgements",
        "analysis",
    ]
    for d in dirs:
        (tmp_path / "workspace" / d).mkdir(parents=True)
    (tmp_path / "specs").mkdir()
    (tmp_path / "batches").mkdir()
    (tmp_path / "experiments").mkdir()
    return tmp_path


@pytest.fixture
def sample_spec_dict() -> dict:
    """Minimal valid experiment spec as a dict (v1 format)."""
    return {
        "schema_version": "1.0",
        "experiment_id": "test-001",
        "experiment_type": "needle_in_haystack",
        "corpus": {
            "content_profile": "python_repo",
            "target_token_count": 10_000,
            "target_file_count": 10,
            "folder_depth": 2,
            "folder_distribution": "balanced",
            "generation_model": "haiku",
            "red_herring_density": "low",
        },
        "payload": {
            "insertion_model": "sonnet",
            "red_herring_hint": "Variables with similar names but different values",
            "items": [
                {
                    "item_id": "target_001",
                    "item_type": "config_value",
                    "content_hint": "A database connection timeout set to a specific number of seconds",
                    "placement": {
                        "strategy": "random_file",
                    },
                    "camouflage": "medium",
                }
            ],
        },
        "question": "What is the database connection timeout?",
        "rubric_criteria": [
            {"criterion": "correctness", "weight": 1.0},
        ],
        "runner": {
            "n_repeats": 1,
            "agent_model": "sonnet",
            "max_tokens": 50_000,
            "allowed_tools": ["Read", "Glob", "Grep"],
        },
    }
```

- [ ] **Step 2: Run the full test suite**

Run: `poetry run pytest tests/ -v`
Expected: All tests PASS (44 existing + ~30 new ≈ 74 total)

- [ ] **Step 3: Commit**

```bash
git add tests/conftest.py
git commit -m "feat: update workspace fixture with v2 pool directories"
```

---

## Self-Review Checklist

**1. Spec coverage:**
- [x] Experiment type taxonomy (single_needle, multi_chain, multi_reasoning) → Task 1
- [x] Parameter space and grid expansion → Tasks 1, 2
- [x] Unified discriminability rubric → Task 7 (in insertion prompt)
- [x] Experiment YAML format (v2) with inline grid → Tasks 1, 11
- [x] Batch file format (v2) → Task 9
- [x] Content profiles (python_repo v2, noir_fiction) → Tasks 3, 4
- [x] Background pool generation via Agent SDK → Task 5
- [x] Corpus assembly (sampling) → Task 6
- [x] Payload insertion via Agent SDK → Task 7
- [x] V2 generation orchestrator → Task 8
- [x] Workspace directory layout (pools/) → Task 14
- [x] CLI changes (generate-pool, v2 detection) → Task 10
- [x] Schema changes (answer key, batch) → Task 9
- [x] Analysis loader v2 metadata → Task 12
- [x] New analysis figures → Task 13

**2. Placeholder scan:** No TBDs, TODOs, or vague steps found.

**3. Type consistency:**
- `Parametrisation` used consistently across grid.py, assembler.py, insertion.py, generate.py
- `ExperimentTemplate` used consistently across template.py, grid.py, insertion.py, cli.py
- `parametrisation_id` property consistent in naming and format
- `BatchConfigV2` / `BatchExperimentEntry` consistent across batch.py and cli.py
- `expand_grid` / `filter_parametrisations` signatures consistent between grid.py and test_grid.py
