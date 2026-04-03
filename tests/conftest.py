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
