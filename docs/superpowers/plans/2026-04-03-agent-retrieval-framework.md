# Agent Retrieval Experiment Framework — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a framework that generates synthetic corpora, runs Claude agents against them, judges their retrieval/reasoning performance, and produces standardized analysis.

**Architecture:** Single Python package with directory-contract isolation. Components communicate only through well-defined workspace directories. Agent sandboxing via Claude Agent SDK. Two-phase corpus generation (cheap model for background, smart model for payload insertion).

**Tech Stack:** Python 3.12+, Pydantic, claude-agent-sdk, anthropic SDK, asyncio, PyYAML, pandas, matplotlib/seaborn, Jinja2, pytest

**Spec:** `docs/superpowers/specs/2026-04-03-agent-retrieval-framework-design.md`

---

### Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `src/agent_retrieval/__init__.py`
- Create: `src/agent_retrieval/schema/__init__.py`
- Create: `src/agent_retrieval/generator/__init__.py`
- Create: `src/agent_retrieval/generator/profiles/__init__.py`
- Create: `src/agent_retrieval/runner/__init__.py`
- Create: `src/agent_retrieval/judge/__init__.py`
- Create: `src/agent_retrieval/analysis/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `.gitignore`

- [ ] **Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "agent-retrieval"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "pydantic>=2.0",
    "pyyaml>=6.0",
    "anthropic>=0.50.0",
    "claude-agent-sdk>=0.1.0",
    "pandas>=2.0",
    "matplotlib>=3.8",
    "seaborn>=0.13",
    "jinja2>=3.1",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
]
notebook = [
    "jupyterlab>=4.0",
]

[project.scripts]
agent-retrieval = "agent_retrieval.cli:main"

[tool.hatch.build.targets.wheel]
packages = ["src/agent_retrieval"]

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
```

- [ ] **Step 2: Create package structure**

Create all `__init__.py` files (empty) for:
- `src/agent_retrieval/__init__.py`
- `src/agent_retrieval/schema/__init__.py`
- `src/agent_retrieval/generator/__init__.py`
- `src/agent_retrieval/generator/profiles/__init__.py`
- `src/agent_retrieval/runner/__init__.py`
- `src/agent_retrieval/judge/__init__.py`
- `src/agent_retrieval/analysis/__init__.py`
- `tests/__init__.py`

- [ ] **Step 3: Create conftest.py with shared fixtures**

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
        "judge/answer_keys",
        "judge/judgements",
        "analysis",
    ]
    for d in dirs:
        (tmp_path / "workspace" / d).mkdir(parents=True)
    (tmp_path / "specs").mkdir()
    (tmp_path / "batches").mkdir()
    return tmp_path


@pytest.fixture
def sample_spec_dict() -> dict:
    """Minimal valid experiment spec as a dict."""
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

- [ ] **Step 4: Create .gitignore**

```
workspace/
__pycache__/
*.egg-info/
dist/
.venv/
*.pyc
.pytest_cache/
.ipynb_checkpoints/
```

- [ ] **Step 5: Install the package in dev mode and verify**

Run: `cd /Users/mehrabmodi/Documents/projects/agent_retrieval_expt && pip install -e ".[dev,notebook]"`
Expected: Successful install

- [ ] **Step 6: Run pytest to verify empty test suite**

Run: `pytest -v`
Expected: "no tests ran" or similar, no errors

- [ ] **Step 7: Initialize git and commit**

```bash
git init
git add pyproject.toml src/ tests/ .gitignore docs/
git commit -m "feat: project scaffolding with package structure"
```

---

### Task 2: Pydantic Schema Models

**Files:**
- Create: `src/agent_retrieval/schema/experiment.py`
- Create: `src/agent_retrieval/schema/answer_key.py`
- Create: `src/agent_retrieval/schema/verdict.py`
- Create: `src/agent_retrieval/schema/batch.py`
- Create: `src/agent_retrieval/schema/run_state.py`
- Test: `tests/test_schema.py`

- [ ] **Step 1: Write tests for experiment spec schema**

```python
# tests/test_schema.py
import pytest
import yaml
from agent_retrieval.schema.experiment import ExperimentSpec


class TestExperimentSpec:
    def test_valid_spec_parses(self, sample_spec_dict):
        spec = ExperimentSpec.model_validate(sample_spec_dict)
        assert spec.experiment_id == "test-001"
        assert spec.experiment_type == "needle_in_haystack"
        assert spec.corpus.target_token_count == 10_000
        assert len(spec.payload.items) == 1
        assert spec.payload.items[0].item_id == "target_001"

    def test_missing_experiment_id_raises(self, sample_spec_dict):
        del sample_spec_dict["experiment_id"]
        with pytest.raises(Exception):
            ExperimentSpec.model_validate(sample_spec_dict)

    def test_depends_on_validates(self, sample_spec_dict):
        sample_spec_dict["payload"]["items"].append({
            "item_id": "target_002",
            "depends_on": "target_001",
            "item_type": "cross_reference",
            "content_hint": "References target_001",
            "placement": {"strategy": "random_file"},
            "camouflage": "low",
        })
        spec = ExperimentSpec.model_validate(sample_spec_dict)
        assert spec.payload.items[1].depends_on == "target_001"

    def test_invalid_depends_on_raises(self, sample_spec_dict):
        sample_spec_dict["payload"]["items"].append({
            "item_id": "target_002",
            "depends_on": "nonexistent_item",
            "item_type": "cross_reference",
            "content_hint": "References nothing",
            "placement": {"strategy": "random_file"},
            "camouflage": "low",
        })
        with pytest.raises(Exception):
            ExperimentSpec.model_validate(sample_spec_dict)

    def test_from_yaml_file(self, sample_spec_dict, tmp_path):
        spec_path = tmp_path / "spec.yaml"
        spec_path.write_text(yaml.dump(sample_spec_dict))
        spec = ExperimentSpec.from_yaml(spec_path)
        assert spec.experiment_id == "test-001"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_schema.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement experiment spec models**

```python
# src/agent_retrieval/schema/experiment.py
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, model_validator


class PlacementConfig(BaseModel):
    strategy: Literal["random_file", "specific_depth", "specific_filetype"]
    depth: int | None = None
    filetype: str | None = None


class PayloadItem(BaseModel):
    item_id: str
    item_type: Literal["config_value", "function_def", "fact", "cross_reference"]
    content_hint: str
    placement: PlacementConfig
    camouflage: Literal["low", "medium", "high"]
    depends_on: str | None = None


class CorpusSpec(BaseModel):
    content_profile: str
    target_token_count: int
    target_file_count: int
    folder_depth: int
    folder_distribution: Literal["balanced", "skewed", "flat"]
    generation_model: str
    red_herring_density: Literal["none", "low", "medium", "high"]


class PayloadSpec(BaseModel):
    insertion_model: str
    red_herring_hint: str
    items: list[PayloadItem]


class RubricCriterion(BaseModel):
    criterion: str
    weight: float


class RunnerSpec(BaseModel):
    n_repeats: int
    agent_model: str
    max_tokens: int
    allowed_tools: list[str]


class ExperimentSpec(BaseModel):
    schema_version: str
    experiment_id: str
    experiment_type: str
    corpus: CorpusSpec
    payload: PayloadSpec
    question: str
    rubric_criteria: list[RubricCriterion]
    runner: RunnerSpec

    @model_validator(mode="after")
    def validate_depends_on_references(self) -> ExperimentSpec:
        item_ids = {item.item_id for item in self.payload.items}
        for item in self.payload.items:
            if item.depends_on and item.depends_on not in item_ids:
                raise ValueError(
                    f"Item '{item.item_id}' depends_on '{item.depends_on}' "
                    f"which is not defined in payload items"
                )
        return self

    @classmethod
    def from_yaml(cls, path: Path) -> ExperimentSpec:
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_schema.py::TestExperimentSpec -v`
Expected: All PASS

- [ ] **Step 5: Add tests for answer key, verdict, batch, and run state schemas**

```python
# append to tests/test_schema.py
from datetime import datetime

from agent_retrieval.schema.answer_key import AnswerKey
from agent_retrieval.schema.batch import BatchConfig
from agent_retrieval.schema.run_state import RunState
from agent_retrieval.schema.verdict import Verdict


class TestAnswerKey:
    def test_valid_answer_key(self):
        ak = AnswerKey.model_validate({
            "experiment_id": "test-001",
            "generated_at": "2026-04-03T10:00:00Z",
            "items": [
                {
                    "item_id": "target_001",
                    "inserted_text": "TIMEOUT = 42",
                    "file_path": "src/config.py",
                    "line_range": [10, 10],
                    "context_summary": "Module-level constant",
                }
            ],
            "expected_answers": {
                "question": "What is the timeout?",
                "correctness": "42 seconds",
                "completeness": "Found in src/config.py",
            },
            "rubric_criteria": [
                {"criterion": "correctness", "weight": 1.0},
            ],
        })
        assert ak.experiment_id == "test-001"
        assert ak.items[0].inserted_text == "TIMEOUT = 42"


class TestBatchConfig:
    def test_valid_batch(self):
        batch = BatchConfig.model_validate({
            "batch_name": "test-batch",
            "max_parallel": 2,
            "retry_failed": True,
            "judge_model": "opus",
            "runs": [
                {"experiment_id": "test-001", "n_repeats": 3},
            ],
        })
        assert batch.batch_name == "test-batch"
        assert batch.runs[0].n_repeats == 3

    def test_per_experiment_overrides(self):
        batch = BatchConfig.model_validate({
            "batch_name": "test-batch",
            "max_parallel": 2,
            "retry_failed": False,
            "judge_model": "opus",
            "runs": [
                {
                    "experiment_id": "test-001",
                    "n_repeats": 5,
                    "agent_model": "opus",
                    "judge_model": "sonnet",
                },
            ],
        })
        assert batch.runs[0].agent_model == "opus"
        assert batch.runs[0].judge_model == "sonnet"


class TestRunState:
    def test_valid_run_state(self):
        state = RunState.model_validate({
            "experiment_id": "test-001",
            "run_id": "abc123",
            "batch_name": "test-batch",
            "status": "pending",
            "claude_code_version": "1.0.0",
        })
        assert state.status == "pending"

    def test_status_transitions(self):
        state = RunState(
            experiment_id="test-001",
            run_id="abc123",
            batch_name="test-batch",
            status="pending",
            claude_code_version="1.0.0",
        )
        state.status = "running"
        assert state.status == "running"


class TestVerdict:
    def test_valid_verdict(self):
        v = Verdict.model_validate({
            "experiment_id": "test-001",
            "run_id": "abc123",
            "batch_name": "test-batch",
            "scores": [
                {
                    "criterion": "correctness",
                    "score": 0.85,
                    "weight": 1.0,
                    "reasoning": "Correct value, wrong path",
                },
            ],
            "weighted_score": 0.85,
            "session_metrics": {
                "total_context_tokens": 50000,
                "total_turns": 5,
                "tool_calls": {"Grep": 3, "Read": 2},
                "duration_seconds": 30.0,
            },
        })
        assert v.weighted_score == 0.85
        assert v.session_metrics.tool_calls["Grep"] == 3
```

- [ ] **Step 6: Run tests to verify they fail**

Run: `pytest tests/test_schema.py -v`
Expected: FAIL — modules not found

- [ ] **Step 7: Implement answer key model**

```python
# src/agent_retrieval/schema/answer_key.py
from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel

from agent_retrieval.schema.experiment import RubricCriterion


class AnswerKeyItem(BaseModel):
    item_id: str
    inserted_text: str
    file_path: str
    line_range: list[int]
    context_summary: str


class ExpectedAnswers(BaseModel):
    question: str
    correctness: str
    completeness: str = ""


class AnswerKey(BaseModel):
    experiment_id: str
    generated_at: str
    items: list[AnswerKeyItem]
    expected_answers: ExpectedAnswers
    rubric_criteria: list[RubricCriterion]

    @classmethod
    def from_yaml(cls, path: Path) -> AnswerKey:
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)

    def to_yaml(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, sort_keys=False)
```

- [ ] **Step 8: Implement batch config model**

```python
# src/agent_retrieval/schema/batch.py
from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel


class BatchRunConfig(BaseModel):
    experiment_id: str
    n_repeats: int
    agent_model: str | None = None
    judge_model: str | None = None


class BatchConfig(BaseModel):
    batch_name: str
    max_parallel: int
    retry_failed: bool
    judge_model: str
    runs: list[BatchRunConfig]

    @classmethod
    def from_yaml(cls, path: Path) -> BatchConfig:
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)
```

- [ ] **Step 9: Implement run state model**

```python
# src/agent_retrieval/schema/run_state.py
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel


class RunState(BaseModel):
    experiment_id: str
    run_id: str
    batch_name: str
    status: Literal["pending", "running", "completed", "failed"]
    claude_code_version: str
    started_at: str | None = None
    completed_at: str | None = None
    error_message: str | None = None

    @classmethod
    def from_yaml(cls, path: Path) -> RunState:
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)

    def to_yaml(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, sort_keys=False)
```

- [ ] **Step 10: Implement verdict model**

```python
# src/agent_retrieval/schema/verdict.py
from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel


class ScoreEntry(BaseModel):
    criterion: str
    score: float
    weight: float
    reasoning: str


class SessionMetrics(BaseModel):
    total_context_tokens: int
    total_turns: int
    tool_calls: dict[str, int]
    duration_seconds: float


class Verdict(BaseModel):
    experiment_id: str
    run_id: str
    batch_name: str
    scores: list[ScoreEntry]
    weighted_score: float
    session_metrics: SessionMetrics

    @classmethod
    def from_yaml(cls, path: Path) -> Verdict:
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)

    def to_yaml(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, sort_keys=False)
```

- [ ] **Step 11: Update schema __init__.py with exports**

```python
# src/agent_retrieval/schema/__init__.py
from agent_retrieval.schema.answer_key import AnswerKey, AnswerKeyItem, ExpectedAnswers
from agent_retrieval.schema.batch import BatchConfig, BatchRunConfig
from agent_retrieval.schema.experiment import (
    CorpusSpec,
    ExperimentSpec,
    PayloadItem,
    PayloadSpec,
    PlacementConfig,
    RubricCriterion,
    RunnerSpec,
)
from agent_retrieval.schema.run_state import RunState
from agent_retrieval.schema.verdict import ScoreEntry, SessionMetrics, Verdict

__all__ = [
    "AnswerKey", "AnswerKeyItem", "ExpectedAnswers",
    "BatchConfig", "BatchRunConfig",
    "CorpusSpec", "ExperimentSpec", "PayloadItem", "PayloadSpec",
    "PlacementConfig", "RubricCriterion", "RunnerSpec",
    "RunState",
    "ScoreEntry", "SessionMetrics", "Verdict",
]
```

- [ ] **Step 12: Run all schema tests**

Run: `pytest tests/test_schema.py -v`
Expected: All PASS

- [ ] **Step 13: Commit**

```bash
git add src/agent_retrieval/schema/ tests/test_schema.py
git commit -m "feat: pydantic schema models for specs, answer keys, verdicts, batches, run state"
```

---

### Task 3: Content Profile Base Class and Python Repo Profile

**Files:**
- Create: `src/agent_retrieval/generator/profiles/base.py`
- Create: `src/agent_retrieval/generator/profiles/python_repo.py`
- Create: `src/agent_retrieval/generator/profiles/registry.py`
- Test: `tests/test_profiles.py`

- [ ] **Step 1: Write tests for content profiles**

```python
# tests/test_profiles.py
from pathlib import Path

import pytest

from agent_retrieval.generator.profiles.base import ContentProfile, GenerationContext
from agent_retrieval.generator.profiles.python_repo import PythonRepoProfile
from agent_retrieval.generator.profiles.registry import get_profile
from agent_retrieval.schema.experiment import CorpusSpec


@pytest.fixture
def corpus_spec() -> CorpusSpec:
    return CorpusSpec(
        content_profile="python_repo",
        target_token_count=10_000,
        target_file_count=10,
        folder_depth=2,
        folder_distribution="balanced",
        generation_model="haiku",
        red_herring_density="low",
    )


class TestPythonRepoProfile:
    def test_generates_folder_structure(self, corpus_spec):
        profile = PythonRepoProfile()
        paths = profile.generate_folder_structure(corpus_spec)
        assert len(paths) == corpus_spec.target_file_count
        assert all(isinstance(p, Path) for p in paths)
        assert all(p.suffix == ".py" or p.name in ("README.md", "requirements.txt", "config.yaml") for p in paths)

    def test_folder_depth_respected(self, corpus_spec):
        profile = PythonRepoProfile()
        paths = profile.generate_folder_structure(corpus_spec)
        max_depth = max(len(p.parts) - 1 for p in paths)  # -1 for filename
        assert max_depth <= corpus_spec.folder_depth

    def test_generate_file_prompt_returns_string(self, corpus_spec):
        profile = PythonRepoProfile()
        paths = profile.generate_folder_structure(corpus_spec)
        ctx = GenerationContext(
            corpus_spec=corpus_spec,
            red_herring_hint=None,
            is_red_herring_file=False,
        )
        prompt = profile.generate_file_prompt(paths[0], ctx)
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_red_herring_prompt_differs(self, corpus_spec):
        profile = PythonRepoProfile()
        paths = profile.generate_folder_structure(corpus_spec)
        ctx_normal = GenerationContext(
            corpus_spec=corpus_spec,
            red_herring_hint=None,
            is_red_herring_file=False,
        )
        ctx_herring = GenerationContext(
            corpus_spec=corpus_spec,
            red_herring_hint="Variables with similar names",
            is_red_herring_file=True,
        )
        normal = profile.generate_file_prompt(paths[0], ctx_normal)
        herring = profile.generate_file_prompt(paths[0], ctx_herring)
        assert normal != herring


class TestProfileRegistry:
    def test_get_python_repo(self):
        profile = get_profile("python_repo")
        assert isinstance(profile, PythonRepoProfile)

    def test_unknown_profile_raises(self):
        with pytest.raises(KeyError):
            get_profile("nonexistent_profile")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_profiles.py -v`
Expected: FAIL — modules not found

- [ ] **Step 3: Implement base content profile**

```python
# src/agent_retrieval/generator/profiles/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from agent_retrieval.schema.experiment import CorpusSpec


@dataclass
class GenerationContext:
    corpus_spec: CorpusSpec
    red_herring_hint: str | None
    is_red_herring_file: bool


class ContentProfile(ABC):
    @abstractmethod
    def generate_folder_structure(self, spec: CorpusSpec) -> list[Path]:
        """Return list of relative file paths to create."""
        ...

    @abstractmethod
    def generate_file_prompt(self, path: Path, context: GenerationContext) -> str:
        """Return the LLM prompt to generate content for this file."""
        ...
```

- [ ] **Step 4: Implement Python repo profile**

```python
# src/agent_retrieval/generator/profiles/python_repo.py
from __future__ import annotations

import random
from pathlib import Path

from agent_retrieval.generator.profiles.base import ContentProfile, GenerationContext
from agent_retrieval.schema.experiment import CorpusSpec

PACKAGE_NAMES = [
    "auth", "api", "models", "services", "utils", "config",
    "handlers", "middleware", "core", "db", "cache", "tasks",
    "validators", "serializers", "exceptions", "logging_config",
]

FILE_STEMS = [
    "manager", "handler", "service", "client", "factory",
    "processor", "validator", "serializer", "helper", "adapter",
    "controller", "provider", "resolver", "transformer", "monitor",
    "scheduler", "dispatcher", "registry", "builder", "wrapper",
]


class PythonRepoProfile(ContentProfile):
    def generate_folder_structure(self, spec: CorpusSpec) -> list[Path]:
        rng = random.Random(hash(spec.content_profile + str(spec.target_file_count)))
        paths: list[Path] = []

        # Reserve a few top-level files
        top_level = ["README.md", "requirements.txt", "config.yaml"]
        for name in top_level:
            if len(paths) < spec.target_file_count:
                paths.append(Path(name))

        # Generate package directories up to folder_depth
        packages = rng.sample(PACKAGE_NAMES, min(len(PACKAGE_NAMES), spec.folder_depth * 3))
        dirs: list[Path] = []
        for i, pkg in enumerate(packages):
            if i < spec.folder_depth:
                dirs.append(Path("src") / pkg)
            else:
                parent = rng.choice(dirs[:max(1, len(dirs))])
                child = Path(parent) / pkg
                if len(child.parts) - 1 <= spec.folder_depth:
                    dirs.append(child)

        # Fill directories with .py files
        while len(paths) < spec.target_file_count:
            d = rng.choice(dirs) if dirs else Path("src")
            stem = rng.choice(FILE_STEMS)
            suffix = f"_{rng.randint(1, 99)}" if rng.random() > 0.5 else ""
            filepath = d / f"{stem}{suffix}.py"
            if filepath not in paths:
                paths.append(filepath)

        return paths[:spec.target_file_count]

    def generate_file_prompt(self, path: Path, context: GenerationContext) -> str:
        file_type = path.suffix
        dir_name = path.parent.name if path.parent != Path(".") else "root"

        base_prompt = (
            f"Generate realistic Python source code for a file at '{path}' "
            f"in a '{dir_name}' package of a medium-sized web application. "
            f"Include imports, classes or functions, and docstrings. "
            f"Make it look like production code written by a competent developer. "
            f"The file should be 50-150 lines long. "
            f"Do not include any comments about this being generated."
        )

        if file_type == ".md":
            base_prompt = (
                f"Generate a realistic README.md for a Python web application project. "
                f"Include sections for setup, usage, and configuration. 50-100 lines."
            )
        elif file_type == ".yaml":
            base_prompt = (
                f"Generate a realistic YAML configuration file for a Python web app. "
                f"Include database, cache, logging, and API settings. 30-60 lines."
            )
        elif file_type == ".txt":
            base_prompt = (
                f"Generate a realistic Python requirements.txt with 15-25 common packages "
                f"and pinned versions."
            )

        if context.is_red_herring_file and context.red_herring_hint:
            base_prompt += (
                f"\n\nIMPORTANT: This file should contain content that is thematically "
                f"similar to but distinct from the following: {context.red_herring_hint}. "
                f"Include plausible-looking values that could be confused for the real target."
            )

        return base_prompt
```

- [ ] **Step 5: Implement profile registry**

```python
# src/agent_retrieval/generator/profiles/registry.py
from __future__ import annotations

from agent_retrieval.generator.profiles.base import ContentProfile
from agent_retrieval.generator.profiles.python_repo import PythonRepoProfile

_PROFILES: dict[str, type[ContentProfile]] = {
    "python_repo": PythonRepoProfile,
}


def get_profile(name: str) -> ContentProfile:
    if name not in _PROFILES:
        raise KeyError(f"Unknown content profile: '{name}'. Available: {list(_PROFILES.keys())}")
    return _PROFILES[name]()
```

- [ ] **Step 6: Update profiles __init__.py**

```python
# src/agent_retrieval/generator/profiles/__init__.py
from agent_retrieval.generator.profiles.base import ContentProfile, GenerationContext
from agent_retrieval.generator.profiles.registry import get_profile

__all__ = ["ContentProfile", "GenerationContext", "get_profile"]
```

- [ ] **Step 7: Run tests**

Run: `pytest tests/test_profiles.py -v`
Expected: All PASS

- [ ] **Step 8: Commit**

```bash
git add src/agent_retrieval/generator/profiles/ tests/test_profiles.py
git commit -m "feat: content profile base class, python repo profile, and registry"
```

---

### Task 4: Generator — Background Corpus Generation (Phase 1)

**Files:**
- Create: `src/agent_retrieval/generator/background.py`
- Test: `tests/test_generator_background.py`

- [ ] **Step 1: Write tests for background generator**

```python
# tests/test_generator_background.py
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from agent_retrieval.generator.background import BackgroundGenerator
from agent_retrieval.schema.experiment import ExperimentSpec


@pytest.fixture
def spec(sample_spec_dict) -> ExperimentSpec:
    return ExperimentSpec.model_validate(sample_spec_dict)


class TestBackgroundGenerator:
    @pytest.mark.asyncio
    async def test_creates_folder_structure(self, spec, tmp_workspace):
        corpus_dir = tmp_workspace / "workspace" / "runner" / "corpora" / spec.experiment_id

        mock_response = AsyncMock()
        mock_response.content = [type("TB", (), {"text": "# Generated Python file\nimport os\n\ndef main():\n    pass\n"})()]

        with patch("agent_retrieval.generator.background.get_llm_client") as mock_client:
            mock_client.return_value.messages.create = AsyncMock(return_value=mock_response)
            gen = BackgroundGenerator()
            await gen.generate(spec, corpus_dir)

        assert corpus_dir.exists()
        files = list(corpus_dir.rglob("*"))
        file_count = sum(1 for f in files if f.is_file())
        assert file_count == spec.corpus.target_file_count

    @pytest.mark.asyncio
    async def test_red_herring_files_created(self, spec, tmp_workspace):
        corpus_dir = tmp_workspace / "workspace" / "runner" / "corpora" / spec.experiment_id

        mock_response = AsyncMock()
        mock_response.content = [type("TB", (), {"text": "# content\nx = 1\n"})()]

        with patch("agent_retrieval.generator.background.get_llm_client") as mock_client:
            mock_client.return_value.messages.create = AsyncMock(return_value=mock_response)
            gen = BackgroundGenerator()
            red_herring_hint = "Variables with similar names"
            await gen.generate(spec, corpus_dir, red_herring_hint=red_herring_hint)

        assert corpus_dir.exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_generator_background.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Create LLM client helper**

```python
# src/agent_retrieval/generator/llm_client.py
from __future__ import annotations

import anthropic


def get_llm_client() -> anthropic.AsyncAnthropic:
    return anthropic.AsyncAnthropic()


async def generate_text(client: anthropic.AsyncAnthropic, model: str, prompt: str) -> str:
    response = await client.messages.create(
        model=model,
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text
```

- [ ] **Step 4: Implement background generator**

```python
# src/agent_retrieval/generator/background.py
from __future__ import annotations

import random
from pathlib import Path

from agent_retrieval.generator.llm_client import generate_text, get_llm_client
from agent_retrieval.generator.profiles.base import GenerationContext
from agent_retrieval.generator.profiles.registry import get_profile
from agent_retrieval.schema.experiment import ExperimentSpec

# Fraction of files that get red herring content, by density level
RED_HERRING_FRACTIONS = {"none": 0.0, "low": 0.1, "medium": 0.25, "high": 0.5}


class BackgroundGenerator:
    async def generate(
        self,
        spec: ExperimentSpec,
        corpus_dir: Path,
        red_herring_hint: str | None = None,
    ) -> list[Path]:
        """Generate background corpus files. Returns list of created file paths."""
        profile = get_profile(spec.corpus.content_profile)
        file_paths = profile.generate_folder_structure(spec.corpus)

        # Determine which files get red herring content
        rng = random.Random(hash(spec.experiment_id))
        fraction = RED_HERRING_FRACTIONS.get(spec.corpus.red_herring_density, 0.0)
        n_herrings = int(len(file_paths) * fraction)
        herring_indices = set(rng.sample(range(len(file_paths)), min(n_herrings, len(file_paths))))

        client = get_llm_client()
        created: list[Path] = []

        for i, rel_path in enumerate(file_paths):
            is_herring = i in herring_indices
            ctx = GenerationContext(
                corpus_spec=spec.corpus,
                red_herring_hint=red_herring_hint if is_herring else None,
                is_red_herring_file=is_herring,
            )
            prompt = profile.generate_file_prompt(rel_path, ctx)
            content = await generate_text(client, spec.corpus.generation_model, prompt)

            abs_path = corpus_dir / rel_path
            abs_path.parent.mkdir(parents=True, exist_ok=True)
            abs_path.write_text(content)
            created.append(abs_path)

        return created
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_generator_background.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add src/agent_retrieval/generator/background.py src/agent_retrieval/generator/llm_client.py tests/test_generator_background.py
git commit -m "feat: background corpus generator with red herring support"
```

---

### Task 5: Generator — Payload Insertion (Phase 2) and Answer Key

**Files:**
- Create: `src/agent_retrieval/generator/payload.py`
- Create: `src/agent_retrieval/generator/generate.py` (top-level orchestrator)
- Test: `tests/test_generator_payload.py`

- [ ] **Step 1: Write tests for payload insertion**

```python
# tests/test_generator_payload.py
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from agent_retrieval.generator.payload import PayloadInserter
from agent_retrieval.schema.answer_key import AnswerKey
from agent_retrieval.schema.experiment import ExperimentSpec


@pytest.fixture
def spec(sample_spec_dict) -> ExperimentSpec:
    return ExperimentSpec.model_validate(sample_spec_dict)


@pytest.fixture
def corpus_with_files(tmp_workspace, spec) -> Path:
    """Create a corpus dir with some pre-existing files."""
    corpus_dir = tmp_workspace / "workspace" / "runner" / "corpora" / spec.experiment_id
    corpus_dir.mkdir(parents=True)
    for i in range(5):
        f = corpus_dir / "src" / f"module_{i}.py"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(f"# Module {i}\nimport os\n\ndef func_{i}():\n    pass\n")
    return corpus_dir


class TestPayloadInserter:
    @pytest.mark.asyncio
    async def test_inserts_payload_and_produces_answer_key(self, spec, corpus_with_files, tmp_workspace):
        answer_key_path = tmp_workspace / "workspace" / "judge" / "answer_keys" / f"{spec.experiment_id}.yaml"

        # Mock the LLM to return modified content + structured insertion info
        mock_response_text = json.dumps({
            "modified_content": "# Module 0\nimport os\n\nCONNECTION_TIMEOUT = 42\n\ndef func_0():\n    pass\n",
            "inserted_text": "CONNECTION_TIMEOUT = 42",
            "line_range": [3, 3],
            "context_summary": "Added as module-level constant",
        })
        mock_response = AsyncMock()
        mock_response.content = [type("TB", (), {"text": mock_response_text})()]

        with patch("agent_retrieval.generator.payload.get_llm_client") as mock_client:
            mock_client.return_value.messages.create = AsyncMock(return_value=mock_response)
            inserter = PayloadInserter()
            answer_key = await inserter.insert(spec, corpus_with_files, answer_key_path)

        assert isinstance(answer_key, AnswerKey)
        assert len(answer_key.items) == 1
        assert answer_key.items[0].item_id == "target_001"
        assert answer_key.items[0].inserted_text == "CONNECTION_TIMEOUT = 42"
        assert answer_key_path.exists()

    @pytest.mark.asyncio
    async def test_dependency_order_respected(self, sample_spec_dict, tmp_workspace):
        """Items with depends_on should be inserted after their dependency."""
        sample_spec_dict["payload"]["items"].append({
            "item_id": "target_002",
            "depends_on": "target_001",
            "item_type": "cross_reference",
            "content_hint": "Imports timeout from target_001",
            "placement": {"strategy": "random_file"},
            "camouflage": "low",
        })
        spec = ExperimentSpec.model_validate(sample_spec_dict)

        corpus_dir = tmp_workspace / "workspace" / "runner" / "corpora" / spec.experiment_id
        corpus_dir.mkdir(parents=True)
        for i in range(5):
            f = corpus_dir / "src" / f"module_{i}.py"
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(f"# Module {i}\n")

        answer_key_path = tmp_workspace / "workspace" / "judge" / "answer_keys" / f"{spec.experiment_id}.yaml"

        call_order = []

        async def mock_create(**kwargs):
            prompt = kwargs.get("messages", [{}])[0].get("content", "")
            call_order.append(prompt)
            resp = AsyncMock()
            resp.content = [type("TB", (), {"text": json.dumps({
                "modified_content": "# modified\n",
                "inserted_text": "INSERTED",
                "line_range": [1, 1],
                "context_summary": "test",
            })})()]
            return resp

        with patch("agent_retrieval.generator.payload.get_llm_client") as mock_client:
            mock_client.return_value.messages.create = AsyncMock(side_effect=mock_create)
            inserter = PayloadInserter()
            answer_key = await inserter.insert(spec, corpus_dir, answer_key_path)

        assert len(answer_key.items) == 2
        # target_001 (no deps) should be inserted before target_002 (depends on target_001)
        assert answer_key.items[0].item_id == "target_001"
        assert answer_key.items[1].item_id == "target_002"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_generator_payload.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement payload inserter**

```python
# src/agent_retrieval/generator/payload.py
from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from pathlib import Path

from agent_retrieval.generator.llm_client import get_llm_client
from agent_retrieval.schema.answer_key import (
    AnswerKey,
    AnswerKeyItem,
    ExpectedAnswers,
)
from agent_retrieval.schema.experiment import ExperimentSpec, PayloadItem


def _resolve_insertion_order(items: list[PayloadItem]) -> list[PayloadItem]:
    """Topological sort: items with no deps first, then dependents."""
    ordered: list[PayloadItem] = []
    remaining = list(items)
    resolved_ids: set[str] = set()

    while remaining:
        progress = False
        for item in list(remaining):
            if item.depends_on is None or item.depends_on in resolved_ids:
                ordered.append(item)
                resolved_ids.add(item.item_id)
                remaining.remove(item)
                progress = True
        if not progress:
            raise ValueError(f"Circular dependency in payload items: {[i.item_id for i in remaining]}")

    return ordered


def _select_target_file(
    item: PayloadItem,
    corpus_dir: Path,
    used_files: set[Path],
    rng: random.Random,
) -> Path:
    """Select a file in the corpus to insert the payload into."""
    all_files = [f for f in corpus_dir.rglob("*.py") if f not in used_files]

    if item.placement.strategy == "specific_depth" and item.placement.depth is not None:
        depth = item.placement.depth
        all_files = [f for f in all_files if len(f.relative_to(corpus_dir).parts) - 1 == depth]

    if item.placement.strategy == "specific_filetype" and item.placement.filetype:
        ext = item.placement.filetype
        all_files = [f for f in corpus_dir.rglob(f"*{ext}") if f not in used_files]

    if not all_files:
        all_files = [f for f in corpus_dir.rglob("*") if f.is_file() and f not in used_files]

    return rng.choice(all_files)


class PayloadInserter:
    async def insert(
        self,
        spec: ExperimentSpec,
        corpus_dir: Path,
        answer_key_path: Path,
    ) -> AnswerKey:
        """Insert payload items into corpus and generate answer key."""
        ordered_items = _resolve_insertion_order(spec.payload.items)
        client = get_llm_client()
        rng = random.Random(hash(spec.experiment_id + "payload"))

        answer_items: list[AnswerKeyItem] = []
        used_files: set[Path] = set()
        # Track inserted item context for dependent items
        insertion_context: dict[str, dict] = {}

        for item in ordered_items:
            target_file = _select_target_file(item, corpus_dir, used_files, rng)
            used_files.add(target_file)
            existing_content = target_file.read_text()
            rel_path = str(target_file.relative_to(corpus_dir))

            # Build prompt for insertion model
            dep_context = ""
            if item.depends_on and item.depends_on in insertion_context:
                dep = insertion_context[item.depends_on]
                dep_context = (
                    f"\nThis item depends on a previously inserted item:\n"
                    f"  Item ID: {item.depends_on}\n"
                    f"  Inserted text: {dep['inserted_text']}\n"
                    f"  File: {dep['file_path']}\n"
                    f"Ensure your insertion references or connects to that item.\n"
                )

            prompt = (
                f"You are inserting a specific piece of content into an existing source file.\n"
                f"The file is at: {rel_path}\n"
                f"Item type: {item.item_type}\n"
                f"Content hint: {item.content_hint}\n"
                f"Camouflage level: {item.camouflage} (how much it should blend in)\n"
                f"{dep_context}\n"
                f"Existing file content:\n```\n{existing_content}\n```\n\n"
                f"Modify the file to naturally include the specified content. "
                f"Return a JSON object with these fields:\n"
                f'- "modified_content": the full modified file content\n'
                f'- "inserted_text": the exact text that was inserted (just the new part)\n'
                f'- "line_range": [start_line, end_line] of the insertion\n'
                f'- "context_summary": one-sentence description of how it was inserted\n'
                f"Return ONLY valid JSON, no other text."
            )

            response = await client.messages.create(
                model=spec.payload.insertion_model,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
            )
            result = json.loads(response.content[0].text)

            # Write modified file
            target_file.write_text(result["modified_content"])

            # Record for answer key
            ak_item = AnswerKeyItem(
                item_id=item.item_id,
                inserted_text=result["inserted_text"],
                file_path=rel_path,
                line_range=result["line_range"],
                context_summary=result["context_summary"],
            )
            answer_items.append(ak_item)
            insertion_context[item.item_id] = {
                "inserted_text": result["inserted_text"],
                "file_path": rel_path,
            }

        # Build expected answers from insertion context
        expected = await self._generate_expected_answers(
            client, spec, answer_items
        )

        answer_key = AnswerKey(
            experiment_id=spec.experiment_id,
            generated_at=datetime.now(timezone.utc).isoformat(),
            items=answer_items,
            expected_answers=expected,
            rubric_criteria=spec.rubric_criteria,
        )
        answer_key.to_yaml(answer_key_path)
        return answer_key

    async def _generate_expected_answers(
        self,
        client,
        spec: ExperimentSpec,
        items: list[AnswerKeyItem],
    ) -> ExpectedAnswers:
        """Use the insertion model to generate expected answer text from what was inserted."""
        items_desc = "\n".join(
            f"- {it.item_id}: '{it.inserted_text}' in {it.file_path} ({it.context_summary})"
            for it in items
        )
        prompt = (
            f"Given the following question and the items that were inserted into a codebase, "
            f"write the ideal correct and complete answer.\n\n"
            f"Question: {spec.question}\n\n"
            f"Inserted items:\n{items_desc}\n\n"
            f"Return JSON with:\n"
            f'- "correctness": the factually correct answer\n'
            f'- "completeness": what a complete answer must include\n'
            f"Return ONLY valid JSON."
        )
        response = await client.messages.create(
            model=spec.payload.insertion_model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        result = json.loads(response.content[0].text)
        return ExpectedAnswers(
            question=spec.question,
            correctness=result["correctness"],
            completeness=result.get("completeness", ""),
        )
```

- [ ] **Step 4: Implement top-level generate orchestrator**

```python
# src/agent_retrieval/generator/generate.py
from __future__ import annotations

from pathlib import Path

from agent_retrieval.generator.background import BackgroundGenerator
from agent_retrieval.generator.payload import PayloadInserter
from agent_retrieval.schema.answer_key import AnswerKey
from agent_retrieval.schema.experiment import ExperimentSpec


async def generate_experiment(
    spec: ExperimentSpec,
    workspace_dir: Path,
    skip_existing: bool = True,
) -> AnswerKey:
    """Generate corpus and answer key for a single experiment spec."""
    corpus_dir = workspace_dir / "runner" / "corpora" / spec.experiment_id
    answer_key_path = workspace_dir / "judge" / "answer_keys" / f"{spec.experiment_id}.yaml"

    if skip_existing and corpus_dir.exists() and answer_key_path.exists():
        return AnswerKey.from_yaml(answer_key_path)

    # Phase 1: Background corpus
    bg = BackgroundGenerator()
    await bg.generate(
        spec,
        corpus_dir,
        red_herring_hint=spec.payload.red_herring_hint,
    )

    # Phase 2: Payload insertion + answer key
    inserter = PayloadInserter()
    answer_key = await inserter.insert(spec, corpus_dir, answer_key_path)

    return answer_key
```

- [ ] **Step 5: Update generator __init__.py**

```python
# src/agent_retrieval/generator/__init__.py
from agent_retrieval.generator.generate import generate_experiment

__all__ = ["generate_experiment"]
```

- [ ] **Step 6: Run tests**

Run: `pytest tests/test_generator_payload.py -v`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add src/agent_retrieval/generator/ tests/test_generator_payload.py
git commit -m "feat: payload insertion, answer key generation, and top-level generate orchestrator"
```

---

### Task 6: Experiment Runner — State Machine and Batch Orchestration

**Files:**
- Create: `src/agent_retrieval/runner/state.py`
- Create: `src/agent_retrieval/runner/session.py`
- Create: `src/agent_retrieval/runner/run.py`
- Test: `tests/test_runner.py`

- [ ] **Step 1: Write tests for runner state management**

```python
# tests/test_runner.py
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import yaml

from agent_retrieval.schema.batch import BatchConfig
from agent_retrieval.schema.experiment import ExperimentSpec
from agent_retrieval.schema.run_state import RunState
from agent_retrieval.runner.state import RunStateManager


@pytest.fixture
def batch_config() -> BatchConfig:
    return BatchConfig.model_validate({
        "batch_name": "test-batch",
        "max_parallel": 2,
        "retry_failed": True,
        "judge_model": "opus",
        "runs": [
            {"experiment_id": "test-001", "n_repeats": 2},
        ],
    })


@pytest.fixture
def spec(sample_spec_dict) -> ExperimentSpec:
    return ExperimentSpec.model_validate(sample_spec_dict)


class TestRunStateManager:
    def test_create_pending_runs(self, batch_config, tmp_workspace):
        runs_dir = tmp_workspace / "workspace" / "runner" / "runs"
        mgr = RunStateManager(runs_dir)
        run_ids = mgr.create_pending_runs(batch_config, "test-001", n_repeats=2, claude_version="1.0.0")
        assert len(run_ids) == 2
        for run_id in run_ids:
            state_path = runs_dir / "test-batch" / "test-001" / run_id / "state.yaml"
            assert state_path.exists()
            state = RunState.from_yaml(state_path)
            assert state.status == "pending"

    def test_recover_interrupted_runs(self, batch_config, tmp_workspace):
        runs_dir = tmp_workspace / "workspace" / "runner" / "runs"
        mgr = RunStateManager(runs_dir)
        run_ids = mgr.create_pending_runs(batch_config, "test-001", n_repeats=1, claude_version="1.0.0")

        # Simulate interrupted run
        state_path = runs_dir / "test-batch" / "test-001" / run_ids[0] / "state.yaml"
        state = RunState.from_yaml(state_path)
        state.status = "running"
        state.to_yaml(state_path)

        recovered = mgr.recover_interrupted(batch_config.batch_name)
        assert len(recovered) == 1
        state = RunState.from_yaml(state_path)
        assert state.status == "pending"

    def test_get_pending_runs(self, batch_config, tmp_workspace):
        runs_dir = tmp_workspace / "workspace" / "runner" / "runs"
        mgr = RunStateManager(runs_dir)
        mgr.create_pending_runs(batch_config, "test-001", n_repeats=2, claude_version="1.0.0")
        pending = mgr.get_runs_by_status(batch_config.batch_name, "test-001", "pending")
        assert len(pending) == 2

    def test_skip_completed_runs(self, batch_config, tmp_workspace):
        runs_dir = tmp_workspace / "workspace" / "runner" / "runs"
        mgr = RunStateManager(runs_dir)
        run_ids = mgr.create_pending_runs(batch_config, "test-001", n_repeats=2, claude_version="1.0.0")

        # Mark one as completed
        state_path = runs_dir / "test-batch" / "test-001" / run_ids[0] / "state.yaml"
        state = RunState.from_yaml(state_path)
        state.status = "completed"
        state.to_yaml(state_path)

        pending = mgr.get_runs_by_status(batch_config.batch_name, "test-001", "pending")
        assert len(pending) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_runner.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement run state manager**

```python
# src/agent_retrieval/runner/state.py
from __future__ import annotations

import uuid
from pathlib import Path

from agent_retrieval.schema.batch import BatchConfig
from agent_retrieval.schema.run_state import RunState


class RunStateManager:
    def __init__(self, runs_dir: Path):
        self.runs_dir = runs_dir

    def create_pending_runs(
        self,
        batch: BatchConfig,
        experiment_id: str,
        n_repeats: int,
        claude_version: str,
    ) -> list[str]:
        """Create pending run directories. Returns list of run IDs."""
        run_ids = []
        for _ in range(n_repeats):
            run_id = uuid.uuid4().hex[:12]
            state = RunState(
                experiment_id=experiment_id,
                run_id=run_id,
                batch_name=batch.batch_name,
                status="pending",
                claude_code_version=claude_version,
            )
            run_dir = self.runs_dir / batch.batch_name / experiment_id / run_id
            state.to_yaml(run_dir / "state.yaml")
            run_ids.append(run_id)
        return run_ids

    def recover_interrupted(self, batch_name: str) -> list[str]:
        """Reset any 'running' states to 'pending'. Returns recovered run IDs."""
        recovered = []
        batch_dir = self.runs_dir / batch_name
        if not batch_dir.exists():
            return recovered
        for state_path in batch_dir.rglob("state.yaml"):
            state = RunState.from_yaml(state_path)
            if state.status == "running":
                state.status = "pending"
                state.to_yaml(state_path)
                recovered.append(state.run_id)
        return recovered

    def get_runs_by_status(
        self, batch_name: str, experiment_id: str, status: str
    ) -> list[tuple[str, Path]]:
        """Return (run_id, run_dir) pairs matching the given status."""
        exp_dir = self.runs_dir / batch_name / experiment_id
        if not exp_dir.exists():
            return []
        results = []
        for run_dir in sorted(exp_dir.iterdir()):
            state_path = run_dir / "state.yaml"
            if state_path.exists():
                state = RunState.from_yaml(state_path)
                if state.status == status:
                    results.append((state.run_id, run_dir))
        return results

    def update_status(
        self, run_dir: Path, status: str, **extra_fields: str
    ) -> None:
        state = RunState.from_yaml(run_dir / "state.yaml")
        state.status = status
        for k, v in extra_fields.items():
            setattr(state, k, v)
        state.to_yaml(run_dir / "state.yaml")
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_runner.py -v`
Expected: All PASS

- [ ] **Step 5: Implement agent session wrapper**

```python
# src/agent_retrieval/runner/session.py
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    TextBlock,
    query,
)


@dataclass
class AgentResult:
    response_text: str
    session_id: str
    num_turns: int
    total_cost_usd: float | None
    usage: dict


def get_claude_version() -> str:
    result = subprocess.run(
        ["claude", "--version"], capture_output=True, text=True, timeout=10
    )
    return result.stdout.strip()


def _find_session_jsonl(session_id: str, cwd: str) -> Path | None:
    """Locate the Agent SDK's session JSONL file."""
    # The SDK stores sessions under ~/.claude/projects/<encoded-cwd>/
    claude_dir = Path.home() / ".claude" / "projects"
    if not claude_dir.exists():
        return None
    # Search for the session ID across project dirs
    for project_dir in claude_dir.iterdir():
        candidate = project_dir / f"{session_id}.jsonl"
        if candidate.exists():
            return candidate
    return None


async def run_agent_session(
    question: str,
    corpus_dir: Path,
    model: str,
    allowed_tools: list[str],
    max_tokens: int,
    run_id: str,
    run_dir: Path,
) -> AgentResult:
    """Run a single agent session and return the result."""
    system_prompt = (
        f"Answer the following question by searching the provided codebase. "
        f"Your session ID is: {run_id}\n\n"
        f"Question: {question}"
    )

    options = ClaudeAgentOptions(
        model=model,
        system_prompt=system_prompt,
        cwd=str(corpus_dir),
        allowed_tools=allowed_tools,
        permission_mode="acceptEdits",
        max_turns=50,
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

    # Copy session JSONL from SDK storage to our run directory
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

- [ ] **Step 6: Implement batch runner**

```python
# src/agent_retrieval/runner/run.py
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

from agent_retrieval.runner.session import AgentResult, get_claude_version, run_agent_session
from agent_retrieval.runner.state import RunStateManager
from agent_retrieval.schema.batch import BatchConfig
from agent_retrieval.schema.experiment import ExperimentSpec


async def run_batch(
    batch: BatchConfig,
    specs_dir: Path,
    workspace_dir: Path,
) -> None:
    """Run all experiments in a batch."""
    runs_dir = workspace_dir / "runner" / "runs"
    corpora_dir = workspace_dir / "runner" / "corpora"
    state_mgr = RunStateManager(runs_dir)

    claude_version = get_claude_version()
    print(f"Claude Code version: {claude_version}")

    # Recover any interrupted runs
    recovered = state_mgr.recover_interrupted(batch.batch_name)
    if recovered:
        print(f"Recovered {len(recovered)} interrupted runs")

    # Create pending runs for any experiments that don't have enough
    for run_config in batch.runs:
        exp_id = run_config.experiment_id
        existing = state_mgr.get_runs_by_status(batch.batch_name, exp_id, "pending")
        existing += state_mgr.get_runs_by_status(batch.batch_name, exp_id, "completed")
        n_needed = run_config.n_repeats - len(existing)
        if n_needed > 0:
            state_mgr.create_pending_runs(batch, exp_id, n_needed, claude_version)

    # Collect all pending runs
    all_pending: list[tuple[str, str, Path]] = []  # (exp_id, run_id, run_dir)
    for run_config in batch.runs:
        pending = state_mgr.get_runs_by_status(batch.batch_name, run_config.experiment_id, "pending")
        for run_id, run_dir in pending:
            all_pending.append((run_config.experiment_id, run_id, run_dir))

    if batch.retry_failed:
        for run_config in batch.runs:
            failed = state_mgr.get_runs_by_status(batch.batch_name, run_config.experiment_id, "failed")
            for run_id, run_dir in failed:
                state_mgr.update_status(run_dir, "pending")
                all_pending.append((run_config.experiment_id, run_id, run_dir))

    total = len(all_pending)
    print(f"Running {total} experiment sessions (max_parallel={batch.max_parallel})")

    semaphore = asyncio.Semaphore(batch.max_parallel)
    completed = 0

    async def run_one(exp_id: str, run_id: str, run_dir: Path) -> None:
        nonlocal completed
        async with semaphore:
            spec = ExperimentSpec.from_yaml(specs_dir / f"{exp_id}.yaml")
            corpus_dir = corpora_dir / exp_id

            # Find per-experiment overrides from batch
            run_config = next(r for r in batch.runs if r.experiment_id == exp_id)
            agent_model = run_config.agent_model or spec.runner.agent_model

            state_mgr.update_status(
                run_dir, "running",
                started_at=datetime.now(timezone.utc).isoformat(),
            )

            try:
                result = await run_agent_session(
                    question=spec.question,
                    corpus_dir=corpus_dir,
                    model=agent_model,
                    allowed_tools=spec.runner.allowed_tools,
                    max_tokens=spec.runner.max_tokens,
                    run_id=run_id,
                    run_dir=run_dir,
                )

                # Save response
                response_path = run_dir / "response.json"
                response_path.write_text(json.dumps({
                    "response_text": result.response_text,
                    "session_id": result.session_id,
                    "num_turns": result.num_turns,
                    "total_cost_usd": result.total_cost_usd,
                    "usage": result.usage,
                }, indent=2))

                state_mgr.update_status(
                    run_dir, "completed",
                    completed_at=datetime.now(timezone.utc).isoformat(),
                )
                completed += 1
                print(f"[{completed}/{total}] Completed {exp_id} run {run_id}")

            except Exception as e:
                state_mgr.update_status(
                    run_dir, "failed",
                    error_message=str(e),
                )
                completed += 1
                print(f"[{completed}/{total}] FAILED {exp_id} run {run_id}: {e}")

    tasks = [run_one(exp_id, run_id, run_dir) for exp_id, run_id, run_dir in all_pending]
    await asyncio.gather(*tasks)
    print(f"Batch '{batch.batch_name}' complete.")
```

- [ ] **Step 7: Update runner __init__.py**

```python
# src/agent_retrieval/runner/__init__.py
from agent_retrieval.runner.run import run_batch

__all__ = ["run_batch"]
```

- [ ] **Step 8: Run tests**

Run: `pytest tests/test_runner.py -v`
Expected: All PASS

- [ ] **Step 9: Commit**

```bash
git add src/agent_retrieval/runner/ tests/test_runner.py
git commit -m "feat: experiment runner with state machine, agent SDK sessions, and batch orchestration"
```

---

### Task 7: LLM Judge — Metrics Extraction

**Files:**
- Create: `src/agent_retrieval/judge/metrics.py`
- Test: `tests/test_judge_metrics.py`

- [ ] **Step 1: Write tests for JSONL metrics extraction**

```python
# tests/test_judge_metrics.py
import json
from pathlib import Path

import pytest

from agent_retrieval.judge.metrics import extract_session_metrics
from agent_retrieval.schema.verdict import SessionMetrics


@pytest.fixture
def sample_jsonl(tmp_path) -> Path:
    """Create a sample session JSONL file."""
    messages = [
        {
            "type": "assistant",
            "timestamp": "2026-04-03T10:00:00Z",
            "message": {
                "content": [
                    {"type": "tool_use", "name": "Grep", "input": {"pattern": "timeout"}},
                ],
                "usage": {"input_tokens": 5000, "output_tokens": 200},
            },
        },
        {
            "type": "tool_result",
            "timestamp": "2026-04-03T10:00:05Z",
        },
        {
            "type": "assistant",
            "timestamp": "2026-04-03T10:00:06Z",
            "message": {
                "content": [
                    {"type": "tool_use", "name": "Read", "input": {"path": "config.py"}},
                    {"type": "tool_use", "name": "Grep", "input": {"pattern": "import"}},
                ],
                "usage": {"input_tokens": 8000, "output_tokens": 300},
            },
        },
        {
            "type": "tool_result",
            "timestamp": "2026-04-03T10:00:10Z",
        },
        {
            "type": "assistant",
            "timestamp": "2026-04-03T10:00:11Z",
            "message": {
                "content": [
                    {"type": "text", "text": "The timeout is 42 seconds."},
                ],
                "usage": {"input_tokens": 10000, "output_tokens": 100},
            },
        },
        {
            "type": "result",
            "timestamp": "2026-04-03T10:00:30Z",
            "duration_ms": 30000,
            "num_turns": 3,
            "usage": {"input_tokens": 23000, "output_tokens": 600},
        },
    ]
    jsonl_path = tmp_path / "session.jsonl"
    with open(jsonl_path, "w") as f:
        for msg in messages:
            f.write(json.dumps(msg) + "\n")
    return jsonl_path


class TestExtractSessionMetrics:
    def test_extracts_token_counts(self, sample_jsonl):
        metrics = extract_session_metrics(sample_jsonl)
        assert metrics.total_context_tokens == 23000

    def test_extracts_tool_calls(self, sample_jsonl):
        metrics = extract_session_metrics(sample_jsonl)
        assert metrics.tool_calls["Grep"] == 2
        assert metrics.tool_calls["Read"] == 1

    def test_extracts_turns(self, sample_jsonl):
        metrics = extract_session_metrics(sample_jsonl)
        assert metrics.total_turns == 3

    def test_extracts_duration(self, sample_jsonl):
        metrics = extract_session_metrics(sample_jsonl)
        assert metrics.duration_seconds == 30.0

    def test_empty_jsonl(self, tmp_path):
        empty = tmp_path / "empty.jsonl"
        empty.write_text("")
        metrics = extract_session_metrics(empty)
        assert metrics.total_context_tokens == 0
        assert metrics.total_turns == 0
        assert metrics.tool_calls == {}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_judge_metrics.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement metrics extraction**

```python
# src/agent_retrieval/judge/metrics.py
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from agent_retrieval.schema.verdict import SessionMetrics


def extract_session_metrics(jsonl_path: Path) -> SessionMetrics:
    """Parse Agent SDK session JSONL and extract metrics."""
    tool_calls: Counter[str] = Counter()
    total_input_tokens = 0
    total_turns = 0
    duration_seconds = 0.0

    if not jsonl_path.exists() or jsonl_path.stat().st_size == 0:
        return SessionMetrics(
            total_context_tokens=0,
            total_turns=0,
            tool_calls={},
            duration_seconds=0.0,
        )

    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)

            if entry.get("type") == "assistant":
                message = entry.get("message", {})
                content = message.get("content", [])
                for block in content:
                    if block.get("type") == "tool_use":
                        tool_calls[block["name"]] += 1

            if entry.get("type") == "result":
                usage = entry.get("usage", {})
                total_input_tokens = usage.get("input_tokens", 0)
                total_turns = entry.get("num_turns", 0)
                duration_seconds = entry.get("duration_ms", 0) / 1000.0

    return SessionMetrics(
        total_context_tokens=total_input_tokens,
        total_turns=total_turns,
        tool_calls=dict(tool_calls),
        duration_seconds=duration_seconds,
    )
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_judge_metrics.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/agent_retrieval/judge/metrics.py tests/test_judge_metrics.py
git commit -m "feat: JSONL session metrics extraction for judge"
```

---

### Task 8: LLM Judge — Scoring and Verdict Generation

**Files:**
- Create: `src/agent_retrieval/judge/scoring.py`
- Create: `src/agent_retrieval/judge/judge.py`
- Test: `tests/test_judge_scoring.py`

- [ ] **Step 1: Write tests for LLM scoring**

```python
# tests/test_judge_scoring.py
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from agent_retrieval.judge.scoring import score_response
from agent_retrieval.schema.answer_key import AnswerKey
from agent_retrieval.schema.verdict import ScoreEntry


@pytest.fixture
def answer_key() -> AnswerKey:
    return AnswerKey.model_validate({
        "experiment_id": "test-001",
        "generated_at": "2026-04-03T10:00:00Z",
        "items": [
            {
                "item_id": "target_001",
                "inserted_text": "TIMEOUT = 42",
                "file_path": "src/config.py",
                "line_range": [10, 10],
                "context_summary": "Module-level constant",
            }
        ],
        "expected_answers": {
            "question": "What is the timeout?",
            "correctness": "42 seconds in src/config.py",
            "completeness": "Found in config.py",
        },
        "rubric_criteria": [
            {"criterion": "correctness", "weight": 1.0},
            {"criterion": "completeness", "weight": 0.5},
        ],
    })


class TestScoreResponse:
    @pytest.mark.asyncio
    async def test_returns_score_entries(self, answer_key):
        mock_response_text = json.dumps({
            "scores": [
                {"criterion": "correctness", "score": 0.9, "reasoning": "Correct value found"},
                {"criterion": "completeness", "score": 1.0, "reasoning": "All items found"},
            ]
        })
        mock_response = AsyncMock()
        mock_response.content = [type("TB", (), {"text": mock_response_text})()]

        with patch("agent_retrieval.judge.scoring.get_llm_client") as mock_client:
            mock_client.return_value.messages.create = AsyncMock(return_value=mock_response)
            scores = await score_response(
                agent_response="The timeout is 42 seconds, found in src/config.py",
                answer_key=answer_key,
                judge_model="opus",
            )

        assert len(scores) == 2
        assert scores[0].criterion == "correctness"
        assert scores[0].score == 0.9

    @pytest.mark.asyncio
    async def test_weighted_score_computation(self, answer_key):
        mock_response_text = json.dumps({
            "scores": [
                {"criterion": "correctness", "score": 0.8, "reasoning": "Mostly correct"},
                {"criterion": "completeness", "score": 1.0, "reasoning": "Complete"},
            ]
        })
        mock_response = AsyncMock()
        mock_response.content = [type("TB", (), {"text": mock_response_text})()]

        with patch("agent_retrieval.judge.scoring.get_llm_client") as mock_client:
            mock_client.return_value.messages.create = AsyncMock(return_value=mock_response)
            scores = await score_response(
                agent_response="Mostly correct answer",
                answer_key=answer_key,
                judge_model="opus",
            )

        # Weighted: (0.8*1.0 + 1.0*0.5) / (1.0 + 0.5) = 1.3/1.5 = 0.8667
        total_weight = sum(c.weight for c in answer_key.rubric_criteria)
        weighted = sum(s.score * s.weight for s in scores) / total_weight
        assert abs(weighted - 0.8667) < 0.01
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_judge_scoring.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement LLM scoring**

```python
# src/agent_retrieval/judge/scoring.py
from __future__ import annotations

import json

from agent_retrieval.generator.llm_client import get_llm_client
from agent_retrieval.schema.answer_key import AnswerKey
from agent_retrieval.schema.verdict import ScoreEntry


async def score_response(
    agent_response: str,
    answer_key: AnswerKey,
    judge_model: str,
) -> list[ScoreEntry]:
    """Use an LLM judge to score the agent's response against the answer key."""
    criteria_desc = "\n".join(
        f"- {c.criterion} (weight: {c.weight})" for c in answer_key.rubric_criteria
    )
    items_desc = "\n".join(
        f"- {it.item_id}: '{it.inserted_text}' at {it.file_path}" for it in answer_key.items
    )

    prompt = (
        f"You are a strict judge evaluating an AI agent's response to a retrieval question.\n\n"
        f"**Question:** {answer_key.expected_answers.question}\n\n"
        f"**Ground truth items inserted into the codebase:**\n{items_desc}\n\n"
        f"**Expected correct answer:** {answer_key.expected_answers.correctness}\n"
        f"**Expected complete answer:** {answer_key.expected_answers.completeness}\n\n"
        f"**Agent's response:**\n{agent_response}\n\n"
        f"**Scoring criteria:**\n{criteria_desc}\n\n"
        f"Score each criterion from 0.0 to 1.0 where:\n"
        f"- 1.0 = perfectly correct/complete\n"
        f"- 0.0 = completely wrong/missing\n"
        f"- Partial credit for partial answers\n\n"
        f"Return JSON: {{\"scores\": [{{\"criterion\": \"...\", \"score\": 0.0-1.0, "
        f"\"reasoning\": \"...\"}}]}}\n"
        f"Return ONLY valid JSON."
    )

    client = get_llm_client()
    response = await client.messages.create(
        model=judge_model,
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    )
    result = json.loads(response.content[0].text)

    weight_map = {c.criterion: c.weight for c in answer_key.rubric_criteria}
    return [
        ScoreEntry(
            criterion=s["criterion"],
            score=s["score"],
            weight=weight_map.get(s["criterion"], 1.0),
            reasoning=s["reasoning"],
        )
        for s in result["scores"]
    ]
```

- [ ] **Step 4: Implement top-level judge orchestrator**

```python
# src/agent_retrieval/judge/judge.py
from __future__ import annotations

import json
from pathlib import Path

from agent_retrieval.judge.metrics import extract_session_metrics
from agent_retrieval.judge.scoring import score_response
from agent_retrieval.schema.answer_key import AnswerKey
from agent_retrieval.schema.batch import BatchConfig
from agent_retrieval.schema.verdict import Verdict


async def judge_run(
    run_dir: Path,
    answer_key: AnswerKey,
    judge_model: str,
    batch_name: str,
    verdict_path: Path,
) -> Verdict:
    """Judge a single run. Returns the verdict."""
    response_path = run_dir / "response.json"
    session_path = run_dir / "session.jsonl"

    response_data = json.loads(response_path.read_text())
    agent_response = response_data["response_text"]

    # LLM scoring
    scores = await score_response(agent_response, answer_key, judge_model)

    # Compute weighted score
    total_weight = sum(s.weight for s in scores)
    weighted_score = sum(s.score * s.weight for s in scores) / total_weight if total_weight > 0 else 0.0

    # Session metrics from JSONL
    metrics = extract_session_metrics(session_path)

    run_id = run_dir.name

    verdict = Verdict(
        experiment_id=answer_key.experiment_id,
        run_id=run_id,
        batch_name=batch_name,
        scores=scores,
        weighted_score=round(weighted_score, 4),
        session_metrics=metrics,
    )
    verdict.to_yaml(verdict_path)
    return verdict


async def judge_batch(
    batch: BatchConfig,
    workspace_dir: Path,
    rejudge: bool = False,
) -> list[Verdict]:
    """Judge all completed runs in a batch."""
    runs_dir = workspace_dir / "runner" / "runs"
    answer_keys_dir = workspace_dir / "judge" / "answer_keys"
    judgements_dir = workspace_dir / "judge" / "judgements"

    verdicts: list[Verdict] = []

    for run_config in batch.runs:
        exp_id = run_config.experiment_id
        judge_model = run_config.judge_model or batch.judge_model
        answer_key = AnswerKey.from_yaml(answer_keys_dir / f"{exp_id}.yaml")

        exp_runs_dir = runs_dir / batch.batch_name / exp_id
        if not exp_runs_dir.exists():
            continue

        for run_dir in sorted(exp_runs_dir.iterdir()):
            if not (run_dir / "response.json").exists():
                continue

            verdict_path = judgements_dir / batch.batch_name / exp_id / f"{run_dir.name}.yaml"
            if verdict_path.exists() and not rejudge:
                verdicts.append(Verdict.from_yaml(verdict_path))
                continue

            verdict = await judge_run(
                run_dir=run_dir,
                answer_key=answer_key,
                judge_model=judge_model,
                batch_name=batch.batch_name,
                verdict_path=verdict_path,
            )
            verdicts.append(verdict)
            print(f"Judged {exp_id}/{run_dir.name}: weighted_score={verdict.weighted_score}")

    return verdicts
```

- [ ] **Step 5: Update judge __init__.py**

```python
# src/agent_retrieval/judge/__init__.py
from agent_retrieval.judge.judge import judge_batch, judge_run

__all__ = ["judge_batch", "judge_run"]
```

- [ ] **Step 6: Run tests**

Run: `pytest tests/test_judge_scoring.py tests/test_judge_metrics.py -v`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add src/agent_retrieval/judge/ tests/test_judge_scoring.py
git commit -m "feat: LLM judge with scoring, metrics extraction, and batch orchestration"
```

---

### Task 9: Analysis Module — Data Loading and Summary Tables

**Files:**
- Create: `src/agent_retrieval/analysis/loader.py`
- Create: `src/agent_retrieval/analysis/tables.py`
- Test: `tests/test_analysis.py`

- [ ] **Step 1: Write tests for data loading and tables**

```python
# tests/test_analysis.py
import yaml
from pathlib import Path

import pandas as pd
import pytest

from agent_retrieval.analysis.loader import load_batch_results
from agent_retrieval.analysis.tables import accuracy_by_type, accuracy_by_param, tool_usage_by_type


@pytest.fixture
def populated_workspace(tmp_workspace) -> Path:
    """Create workspace with sample verdicts and specs."""
    ws = tmp_workspace

    # Create two specs with different types
    for exp_id, exp_type, token_count in [
        ("needle-001", "needle_in_haystack", 100_000),
        ("needle-002", "needle_in_haystack", 500_000),
        ("chain-001", "chain_of_retrieval", 100_000),
    ]:
        spec = {
            "schema_version": "1.0",
            "experiment_id": exp_id,
            "experiment_type": exp_type,
            "corpus": {
                "content_profile": "python_repo",
                "target_token_count": token_count,
                "target_file_count": 50,
                "folder_depth": 3,
                "folder_distribution": "balanced",
                "generation_model": "haiku",
                "red_herring_density": "medium",
            },
            "payload": {
                "insertion_model": "sonnet",
                "red_herring_hint": "Similar values",
                "items": [{"item_id": "t1", "item_type": "config_value",
                           "content_hint": "a value", "placement": {"strategy": "random_file"},
                           "camouflage": "medium"}],
            },
            "question": "Find the value",
            "rubric_criteria": [{"criterion": "correctness", "weight": 1.0}],
            "runner": {"n_repeats": 2, "agent_model": "sonnet", "max_tokens": 50000,
                       "allowed_tools": ["Read", "Grep"]},
        }
        spec_path = ws / "specs" / f"{exp_id}.yaml"
        spec_path.parent.mkdir(parents=True, exist_ok=True)
        spec_path.write_text(yaml.dump(spec))

    # Create verdicts
    batch_name = "test-batch"
    for exp_id, scores in [
        ("needle-001", [0.9, 0.85]),
        ("needle-002", [0.7, 0.6]),
        ("chain-001", [0.95, 0.9]),
    ]:
        for i, score in enumerate(scores):
            verdict = {
                "experiment_id": exp_id,
                "run_id": f"run_{i}",
                "batch_name": batch_name,
                "scores": [{"criterion": "correctness", "score": score, "weight": 1.0, "reasoning": "test"}],
                "weighted_score": score,
                "session_metrics": {
                    "total_context_tokens": 50000 + i * 10000,
                    "total_turns": 5 + i,
                    "tool_calls": {"Grep": 3 + i, "Read": 2 + i},
                    "duration_seconds": 30.0 + i * 5,
                },
            }
            verdict_path = ws / "workspace" / "judge" / "judgements" / batch_name / exp_id / f"run_{i}.yaml"
            verdict_path.parent.mkdir(parents=True, exist_ok=True)
            verdict_path.write_text(yaml.dump(verdict))

    return ws


class TestLoadBatchResults:
    def test_loads_all_verdicts_into_dataframe(self, populated_workspace):
        df = load_batch_results(
            "test-batch",
            workspace_dir=populated_workspace / "workspace",
            specs_dir=populated_workspace / "specs",
        )
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 6  # 3 experiments x 2 repeats
        assert "experiment_type" in df.columns
        assert "weighted_score" in df.columns
        assert "target_token_count" in df.columns


class TestAccuracyByType:
    def test_groups_by_experiment_type(self, populated_workspace):
        df = load_batch_results(
            "test-batch",
            workspace_dir=populated_workspace / "workspace",
            specs_dir=populated_workspace / "specs",
        )
        result = accuracy_by_type(df)
        assert len(result) == 2  # needle_in_haystack, chain_of_retrieval
        assert "mean" in result.columns
        assert "std" in result.columns


class TestAccuracyByParam:
    def test_groups_by_type_and_param(self, populated_workspace):
        df = load_batch_results(
            "test-batch",
            workspace_dir=populated_workspace / "workspace",
            specs_dir=populated_workspace / "specs",
        )
        result = accuracy_by_param(df, param_column="target_token_count")
        assert len(result) > 0


class TestToolUsageByType:
    def test_returns_tool_counts(self, populated_workspace):
        df = load_batch_results(
            "test-batch",
            workspace_dir=populated_workspace / "workspace",
            specs_dir=populated_workspace / "specs",
        )
        result = tool_usage_by_type(df)
        assert "Grep" in result.columns or "Grep" in result.index.get_level_values(-1)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_analysis.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement data loader**

```python
# src/agent_retrieval/analysis/loader.py
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from agent_retrieval.schema.verdict import Verdict


def load_batch_results(
    batch_name: str,
    workspace_dir: Path,
    specs_dir: Path,
) -> pd.DataFrame:
    """Load all verdicts for a batch into a DataFrame, enriched with spec metadata."""
    judgements_dir = workspace_dir / "judge" / "judgements" / batch_name
    if not judgements_dir.exists():
        return pd.DataFrame()

    rows: list[dict] = []
    spec_cache: dict[str, dict] = {}

    for verdict_path in judgements_dir.rglob("*.yaml"):
        verdict = Verdict.from_yaml(verdict_path)

        # Load and cache spec metadata
        if verdict.experiment_id not in spec_cache:
            spec_path = specs_dir / f"{verdict.experiment_id}.yaml"
            if spec_path.exists():
                with open(spec_path) as f:
                    spec_data = yaml.safe_load(f)
                spec_cache[verdict.experiment_id] = spec_data

        spec_data = spec_cache.get(verdict.experiment_id, {})

        row = {
            "experiment_id": verdict.experiment_id,
            "run_id": verdict.run_id,
            "batch_name": verdict.batch_name,
            "weighted_score": verdict.weighted_score,
            "total_context_tokens": verdict.session_metrics.total_context_tokens,
            "total_turns": verdict.session_metrics.total_turns,
            "duration_seconds": verdict.session_metrics.duration_seconds,
            # Spec metadata
            "experiment_type": spec_data.get("experiment_type", "unknown"),
            "target_token_count": spec_data.get("corpus", {}).get("target_token_count", 0),
            "target_file_count": spec_data.get("corpus", {}).get("target_file_count", 0),
            "folder_depth": spec_data.get("corpus", {}).get("folder_depth", 0),
            "red_herring_density": spec_data.get("corpus", {}).get("red_herring_density", "none"),
            "n_payload_items": len(spec_data.get("payload", {}).get("items", [])),
        }

        # Flatten tool calls
        for tool, count in verdict.session_metrics.tool_calls.items():
            row[f"tool_{tool}"] = count

        # Flatten per-criterion scores
        for score_entry in verdict.scores:
            row[f"score_{score_entry.criterion}"] = score_entry.score

        rows.append(row)

    return pd.DataFrame(rows)
```

- [ ] **Step 4: Implement summary tables**

```python
# src/agent_retrieval/analysis/tables.py
from __future__ import annotations

import pandas as pd


def accuracy_by_type(df: pd.DataFrame) -> pd.DataFrame:
    """Mean/std weighted_score grouped by experiment_type."""
    return df.groupby("experiment_type")["weighted_score"].agg(["mean", "std", "count"]).round(4)


def accuracy_by_param(df: pd.DataFrame, param_column: str) -> pd.DataFrame:
    """Mean/std weighted_score grouped by experiment_type and a parameter column."""
    return (
        df.groupby(["experiment_type", param_column])["weighted_score"]
        .agg(["mean", "std", "count"])
        .round(4)
    )


def tool_usage_by_type(df: pd.DataFrame) -> pd.DataFrame:
    """Tool call counts aggregated by experiment_type."""
    tool_cols = [c for c in df.columns if c.startswith("tool_")]
    if not tool_cols:
        return pd.DataFrame()
    result = df.groupby("experiment_type")[tool_cols].agg(["mean", "std"]).round(2)
    # Flatten multi-level columns
    result.columns = [f"{tool.replace('tool_', '')}_{stat}" for tool, stat in result.columns]
    return result
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_analysis.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add src/agent_retrieval/analysis/loader.py src/agent_retrieval/analysis/tables.py tests/test_analysis.py
git commit -m "feat: analysis data loader and summary table functions"
```

---

### Task 10: Analysis Module — Figures and Report

**Files:**
- Create: `src/agent_retrieval/analysis/figures.py`
- Create: `src/agent_retrieval/analysis/report.py`
- Create: `src/agent_retrieval/analysis/analyze.py`
- Create: `src/agent_retrieval/analysis/templates/report.html.j2`
- Test: `tests/test_analysis_figures.py`

- [ ] **Step 1: Write tests for figure generation**

```python
# tests/test_analysis_figures.py
from pathlib import Path

import pandas as pd
import pytest

from agent_retrieval.analysis.figures import (
    plot_accuracy_vs_corpus_size,
    plot_context_usage,
    plot_tool_distribution,
    plot_cross_type_comparison,
)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame([
        {"experiment_type": "needle_in_haystack", "target_token_count": 100_000,
         "weighted_score": 0.9, "total_context_tokens": 50_000, "tool_Grep": 5, "tool_Read": 3},
        {"experiment_type": "needle_in_haystack", "target_token_count": 100_000,
         "weighted_score": 0.85, "total_context_tokens": 55_000, "tool_Grep": 6, "tool_Read": 4},
        {"experiment_type": "needle_in_haystack", "target_token_count": 500_000,
         "weighted_score": 0.7, "total_context_tokens": 120_000, "tool_Grep": 12, "tool_Read": 8},
        {"experiment_type": "chain_of_retrieval", "target_token_count": 100_000,
         "weighted_score": 0.95, "total_context_tokens": 60_000, "tool_Grep": 8, "tool_Read": 5},
    ])


class TestFigures:
    def test_accuracy_vs_corpus_size_creates_file(self, sample_df, tmp_path):
        out = tmp_path / "accuracy_vs_corpus_size.png"
        plot_accuracy_vs_corpus_size(sample_df, out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_context_usage_creates_file(self, sample_df, tmp_path):
        out = tmp_path / "context_usage.png"
        plot_context_usage(sample_df, out)
        assert out.exists()

    def test_tool_distribution_creates_file(self, sample_df, tmp_path):
        out = tmp_path / "tool_dist.png"
        plot_tool_distribution(sample_df, out)
        assert out.exists()

    def test_cross_type_creates_file(self, sample_df, tmp_path):
        out = tmp_path / "cross_type.png"
        plot_cross_type_comparison(sample_df, out)
        assert out.exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_analysis_figures.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement figure generation**

```python
# src/agent_retrieval/analysis/figures.py
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_accuracy_vs_corpus_size(df: pd.DataFrame, output_path: Path) -> None:
    """Scatter/line: accuracy vs corpus token count, colored by experiment type."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for exp_type, group in df.groupby("experiment_type"):
        agg = group.groupby("target_token_count")["weighted_score"].agg(["mean", "std"]).reset_index()
        ax.errorbar(
            agg["target_token_count"], agg["mean"], yerr=agg["std"],
            label=exp_type, marker="o", capsize=4,
        )
    ax.set_xlabel("Corpus Size (tokens)")
    ax.set_ylabel("Weighted Accuracy Score")
    ax.set_title("Accuracy vs Corpus Size")
    ax.legend()
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_context_usage(df: pd.DataFrame, output_path: Path) -> None:
    """Context tokens consumed vs corpus size."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for exp_type, group in df.groupby("experiment_type"):
        agg = group.groupby("target_token_count")["total_context_tokens"].agg(["mean", "std"]).reset_index()
        ax.errorbar(
            agg["target_token_count"], agg["mean"], yerr=agg["std"],
            label=exp_type, marker="s", capsize=4,
        )
    ax.set_xlabel("Corpus Size (tokens)")
    ax.set_ylabel("Context Tokens Consumed")
    ax.set_title("Context Usage vs Corpus Size")
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_tool_distribution(df: pd.DataFrame, output_path: Path) -> None:
    """Stacked bar: tool call mix per experiment type."""
    tool_cols = [c for c in df.columns if c.startswith("tool_")]
    if not tool_cols:
        return
    agg = df.groupby("experiment_type")[tool_cols].mean()
    agg.columns = [c.replace("tool_", "") for c in agg.columns]
    fig, ax = plt.subplots(figsize=(10, 6))
    agg.plot(kind="bar", stacked=True, ax=ax)
    ax.set_ylabel("Mean Tool Calls per Run")
    ax.set_title("Tool Usage Distribution by Experiment Type")
    ax.legend(title="Tool")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_cross_type_comparison(df: pd.DataFrame, output_path: Path) -> None:
    """Overlay metrics across experiment types on shared dimensions."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Accuracy comparison
    sns.boxplot(data=df, x="experiment_type", y="weighted_score", ax=axes[0])
    axes[0].set_title("Accuracy by Experiment Type")
    axes[0].set_ylim(0, 1.05)
    axes[0].tick_params(axis="x", rotation=30)

    # Context usage comparison
    sns.boxplot(data=df, x="experiment_type", y="total_context_tokens", ax=axes[1])
    axes[1].set_title("Context Usage by Experiment Type")
    axes[1].tick_params(axis="x", rotation=30)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
```

- [ ] **Step 4: Create HTML report template**

```html
{# src/agent_retrieval/analysis/templates/report.html.j2 #}
<!DOCTYPE html>
<html>
<head>
    <title>Experiment Analysis: {{ batch_name }}</title>
    <style>
        body { font-family: sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }
        h1 { border-bottom: 2px solid #333; padding-bottom: 10px; }
        h2 { color: #555; margin-top: 40px; }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 8px 12px; text-align: left; }
        th { background: #f5f5f5; }
        img { max-width: 100%; margin: 20px 0; border: 1px solid #eee; }
        .metric { display: inline-block; margin: 10px 20px; text-align: center; }
        .metric .value { font-size: 2em; font-weight: bold; }
        .metric .label { color: #777; }
    </style>
</head>
<body>
    <h1>Experiment Analysis: {{ batch_name }}</h1>

    <div>
        <div class="metric">
            <div class="value">{{ n_experiments }}</div>
            <div class="label">Experiments</div>
        </div>
        <div class="metric">
            <div class="value">{{ n_runs }}</div>
            <div class="label">Total Runs</div>
        </div>
        <div class="metric">
            <div class="value">{{ "%.3f"|format(mean_accuracy) }}</div>
            <div class="label">Mean Accuracy</div>
        </div>
    </div>

    <h2>Accuracy by Experiment Type</h2>
    {{ accuracy_by_type_html }}

    <h2>Accuracy by Parameter</h2>
    {{ accuracy_by_param_html }}

    <h2>Tool Usage by Type</h2>
    {{ tool_usage_html }}

    <h2>Figures</h2>
    {% for fig_name, fig_path in figures %}
    <h3>{{ fig_name }}</h3>
    <img src="{{ fig_path }}" alt="{{ fig_name }}">
    {% endfor %}
</body>
</html>
```

- [ ] **Step 5: Implement report generator**

```python
# src/agent_retrieval/analysis/report.py
from __future__ import annotations

from pathlib import Path

import pandas as pd
from jinja2 import Environment, FileSystemLoader

from agent_retrieval.analysis.tables import accuracy_by_type, accuracy_by_param, tool_usage_by_type


def generate_report(
    df: pd.DataFrame,
    batch_name: str,
    output_dir: Path,
) -> Path:
    """Generate an HTML report from the analysis results."""
    template_dir = Path(__file__).parent / "templates"
    env = Environment(loader=FileSystemLoader(str(template_dir)))
    template = env.get_template("report.html.j2")

    figures_dir = output_dir / "figures"
    figures = []
    for fig_file in sorted(figures_dir.glob("*.png")):
        name = fig_file.stem.replace("_", " ").title()
        figures.append((name, f"figures/{fig_file.name}"))

    acc_type = accuracy_by_type(df)
    acc_param = accuracy_by_param(df, "target_token_count")
    tool_usage = tool_usage_by_type(df)

    html = template.render(
        batch_name=batch_name,
        n_experiments=df["experiment_id"].nunique(),
        n_runs=len(df),
        mean_accuracy=df["weighted_score"].mean() if len(df) > 0 else 0,
        accuracy_by_type_html=acc_type.to_html() if len(acc_type) > 0 else "<p>No data</p>",
        accuracy_by_param_html=acc_param.to_html() if len(acc_param) > 0 else "<p>No data</p>",
        tool_usage_html=tool_usage.to_html() if len(tool_usage) > 0 else "<p>No data</p>",
        figures=figures,
    )

    report_path = output_dir / "report.html"
    report_path.write_text(html)
    return report_path
```

- [ ] **Step 6: Implement top-level analyze orchestrator**

```python
# src/agent_retrieval/analysis/analyze.py
from __future__ import annotations

from pathlib import Path

from agent_retrieval.analysis.figures import (
    plot_accuracy_vs_corpus_size,
    plot_context_usage,
    plot_cross_type_comparison,
    plot_tool_distribution,
)
from agent_retrieval.analysis.loader import load_batch_results
from agent_retrieval.analysis.report import generate_report
from agent_retrieval.analysis.tables import accuracy_by_type, accuracy_by_param, tool_usage_by_type


def run_analysis(
    batch_name: str,
    workspace_dir: Path,
    specs_dir: Path,
) -> Path:
    """Run full analysis pipeline for a batch. Returns output directory."""
    df = load_batch_results(batch_name, workspace_dir, specs_dir)
    output_dir = workspace_dir / "analysis" / batch_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save summary CSV
    df.to_csv(output_dir / "summary.csv", index=False)

    # Save tables
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(exist_ok=True)
    accuracy_by_type(df).to_csv(tables_dir / "accuracy_by_type.csv")
    accuracy_by_param(df, "target_token_count").to_csv(tables_dir / "accuracy_by_param.csv")
    tool_usage_by_type(df).to_csv(tables_dir / "tool_usage_by_type.csv")

    # Generate figures
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    plot_accuracy_vs_corpus_size(df, figures_dir / "accuracy_vs_corpus_size.png")
    plot_context_usage(df, figures_dir / "context_usage_vs_corpus_size.png")
    plot_tool_distribution(df, figures_dir / "tool_distribution_by_type.png")
    plot_cross_type_comparison(df, figures_dir / "cross_type_comparison.png")

    # Generate HTML report
    generate_report(df, batch_name, output_dir)

    print(f"Analysis written to {output_dir}")
    return output_dir
```

- [ ] **Step 7: Update analysis __init__.py**

```python
# src/agent_retrieval/analysis/__init__.py
from agent_retrieval.analysis.analyze import run_analysis
from agent_retrieval.analysis.figures import (
    plot_accuracy_vs_corpus_size,
    plot_context_usage,
    plot_cross_type_comparison,
    plot_tool_distribution,
)
from agent_retrieval.analysis.loader import load_batch_results
from agent_retrieval.analysis.tables import accuracy_by_param, accuracy_by_type, tool_usage_by_type

__all__ = [
    "run_analysis",
    "load_batch_results",
    "accuracy_by_type",
    "accuracy_by_param",
    "tool_usage_by_type",
    "plot_accuracy_vs_corpus_size",
    "plot_context_usage",
    "plot_cross_type_comparison",
    "plot_tool_distribution",
]
```

- [ ] **Step 8: Run tests**

Run: `pytest tests/test_analysis.py tests/test_analysis_figures.py -v`
Expected: All PASS

- [ ] **Step 9: Commit**

```bash
git add src/agent_retrieval/analysis/ tests/test_analysis_figures.py
git commit -m "feat: analysis module with figures, tables, and HTML report generation"
```

---

### Task 11: CLI Entrypoints

**Files:**
- Create: `src/agent_retrieval/cli.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write tests for CLI argument parsing**

```python
# tests/test_cli.py
from unittest.mock import AsyncMock, patch

import pytest

from agent_retrieval.cli import parse_args


class TestCLI:
    def test_generate_spec_file(self):
        args = parse_args(["generate", "specs/needle-001.yaml"])
        assert args.command == "generate"
        assert args.config_path == "specs/needle-001.yaml"

    def test_generate_batch_file(self):
        args = parse_args(["generate", "batches/test.yaml"])
        assert args.command == "generate"
        assert args.config_path == "batches/test.yaml"

    def test_run_batch(self):
        args = parse_args(["run", "batches/test.yaml"])
        assert args.command == "run"
        assert args.config_path == "batches/test.yaml"

    def test_judge_batch(self):
        args = parse_args(["judge", "batches/test.yaml"])
        assert args.command == "judge"
        assert args.config_path == "batches/test.yaml"

    def test_judge_rejudge_flag(self):
        args = parse_args(["judge", "batches/test.yaml", "--rejudge"])
        assert args.rejudge is True

    def test_analyze_batch(self):
        args = parse_args(["analyze", "batches/test.yaml"])
        assert args.command == "analyze"
        assert args.config_path == "batches/test.yaml"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_cli.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement CLI**

```python
# src/agent_retrieval/cli.py
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from agent_retrieval.schema.batch import BatchConfig
from agent_retrieval.schema.experiment import ExperimentSpec


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="agent-retrieval",
        description="Agent Retrieval Experiment Framework",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="Generate corpus and answer key")
    gen.add_argument("config_path", help="Path to spec YAML or batch YAML")
    gen.add_argument("--workspace", default="workspace", help="Workspace directory")

    run = sub.add_parser("run", help="Run experiments in a batch")
    run.add_argument("config_path", help="Path to batch YAML")
    run.add_argument("--workspace", default="workspace", help="Workspace directory")
    run.add_argument("--specs-dir", default="specs", help="Specs directory")

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
    """Heuristic: batch files live in batches/ or contain batch_name."""
    if "batch" in path.stem.lower() or path.parent.name == "batches":
        return True
    import yaml
    with open(path) as f:
        data = yaml.safe_load(f)
    return "batch_name" in data


async def _generate(args: argparse.Namespace) -> None:
    from agent_retrieval.generator import generate_experiment

    config_path = Path(args.config_path)
    workspace_dir = Path(args.workspace)

    if _is_batch_file(config_path):
        batch = BatchConfig.from_yaml(config_path)
        specs_dir = config_path.parent.parent / "specs"
        for run_config in batch.runs:
            spec = ExperimentSpec.from_yaml(specs_dir / f"{run_config.experiment_id}.yaml")
            print(f"Generating corpus for {spec.experiment_id}...")
            await generate_experiment(spec, workspace_dir)
            print(f"  Done: {spec.experiment_id}")
    else:
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
    if args.command == "generate":
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

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_cli.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/agent_retrieval/cli.py tests/test_cli.py
git commit -m "feat: CLI entrypoints for generate, run, judge, and analyze commands"
```

---

### Task 12: Jupyter Notebook Template

**Files:**
- Create: `notebooks/analysis_template.ipynb`

- [ ] **Step 1: Create the notebook template**

```python
# Use NotebookEdit or write JSON directly
# The notebook should contain these cells:

# Cell 1 (markdown):
# # Agent Retrieval Experiment Analysis
# Configure the batch name below and run all cells.

# Cell 2 (code):
BATCH_NAME = "scaling-test-v1"
WORKSPACE_DIR = "../workspace"
SPECS_DIR = "../specs"

# Cell 3 (code):
from pathlib import Path
from agent_retrieval.analysis import load_batch_results
results = load_batch_results(BATCH_NAME, Path(WORKSPACE_DIR), Path(SPECS_DIR))
print(f"Loaded {len(results)} runs across {results['experiment_id'].nunique()} experiments")
results.head()

# Cell 4 (markdown):
# ## Accuracy by Experiment Type

# Cell 5 (code):
from agent_retrieval.analysis import accuracy_by_type
accuracy_by_type(results)

# Cell 6 (markdown):
# ## Accuracy by Parameter

# Cell 7 (code):
from agent_retrieval.analysis import accuracy_by_param
accuracy_by_param(results, "target_token_count")

# Cell 8 (markdown):
# ## Tool Usage

# Cell 9 (code):
from agent_retrieval.analysis import tool_usage_by_type
tool_usage_by_type(results)

# Cell 10 (markdown):
# ## Figures

# Cell 11 (code):
%matplotlib inline
from agent_retrieval.analysis import (
    plot_accuracy_vs_corpus_size,
    plot_context_usage,
    plot_tool_distribution,
    plot_cross_type_comparison,
)
from pathlib import Path
import matplotlib.pyplot as plt

# Cell 12 (code):
from IPython.display import display
import tempfile
for plot_fn, title in [
    (plot_accuracy_vs_corpus_size, "Accuracy vs Corpus Size"),
    (plot_context_usage, "Context Usage vs Corpus Size"),
    (plot_tool_distribution, "Tool Distribution by Type"),
    (plot_cross_type_comparison, "Cross-Type Comparison"),
]:
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        plot_fn(results, Path(f.name))
        from IPython.display import Image
        display(Image(filename=f.name))

# Cell 13 (markdown):
# ## Custom Analysis
# Use the `results` DataFrame for ad-hoc exploration.

# Cell 14 (code):
# Your analysis here
results.describe()
```

Write this as a proper `.ipynb` JSON file.

- [ ] **Step 2: Verify notebook is valid JSON**

Run: `python -c "import json; json.load(open('notebooks/analysis_template.ipynb'))"`
Expected: No error

- [ ] **Step 3: Commit**

```bash
git add notebooks/
git commit -m "feat: Jupyter notebook template for interactive analysis"
```

---

### Task 13: Sample Experiment Spec and Batch File

**Files:**
- Create: `specs/needle-in-haystack-small.yaml`
- Create: `batches/smoke-test.yaml`

- [ ] **Step 1: Create a sample experiment spec**

```yaml
# specs/needle-in-haystack-small.yaml
schema_version: "1.0"
experiment_id: "needle-in-haystack-small"
experiment_type: "needle_in_haystack"

corpus:
  content_profile: "python_repo"
  target_token_count: 20_000
  target_file_count: 15
  folder_depth: 2
  folder_distribution: "balanced"
  generation_model: "claude-haiku-4-5-20251001"
  red_herring_density: "low"

payload:
  insertion_model: "claude-sonnet-4-6"
  red_herring_hint: "Configuration values with similar names like max_retries, pool_size, but different values"
  items:
    - item_id: "target_001"
      item_type: "config_value"
      content_hint: "A database connection timeout set to a specific number of seconds"
      placement:
        strategy: "random_file"
      camouflage: "medium"

question: "What is the database connection timeout value configured in this codebase?"

rubric_criteria:
  - criterion: "correctness"
    weight: 1.0
  - criterion: "completeness"
    weight: 0.5

runner:
  n_repeats: 3
  agent_model: "claude-sonnet-4-6"
  max_tokens: 100_000
  allowed_tools: ["Read", "Glob", "Grep", "Bash"]
```

- [ ] **Step 2: Create a sample batch file**

```yaml
# batches/smoke-test.yaml
batch_name: "smoke-test"
max_parallel: 2
retry_failed: true
judge_model: "claude-sonnet-4-6"

runs:
  - experiment_id: "needle-in-haystack-small"
    n_repeats: 3
```

- [ ] **Step 3: Validate both files load correctly**

Run: `python -c "from agent_retrieval.schema import ExperimentSpec, BatchConfig; ExperimentSpec.from_yaml('specs/needle-in-haystack-small.yaml'); BatchConfig.from_yaml('batches/smoke-test.yaml'); print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add specs/ batches/
git commit -m "feat: sample experiment spec and smoke-test batch file"
```

---

### Task 14: Full Test Suite Verification

- [ ] **Step 1: Run the entire test suite**

Run: `pytest tests/ -v --tb=short`
Expected: All tests PASS

- [ ] **Step 2: Verify CLI help works**

Run: `agent-retrieval --help`
Expected: Shows usage with generate/run/judge/analyze subcommands

Run: `agent-retrieval generate --help`
Expected: Shows generate-specific options

- [ ] **Step 3: Final commit with any fixes**

```bash
git add -A
git commit -m "chore: final test suite verification and fixes"
```
