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


import json
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from agent_retrieval.runner.run import run_batch
from agent_retrieval.runner.session import AgentResult
from agent_retrieval.schema.batch import BatchConfig


class TestRunnerPureReasoning:
    @pytest.mark.asyncio
    async def test_pure_reasoning_uses_run_dir_as_cwd_and_passes_question(
        self, tmp_workspace
    ):
        """pure_reasoning runs send the inlined question prompt and use
        run_dir as cwd (no populated corpus required)."""
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
            "max_parallel": 1,
            "retry_failed": False,
            "agent_model": "claude-haiku-4-5-20251001",
            "effort_mode": "low",
            "n_repeats": 1,
            "max_turns": 5,
            "allowed_tools": [],
            "experiments": [{
                "experiment_type": "pure_reasoning",
                "grid": {
                    "content_profile": ["python_repo"],
                    "n_items": [2],
                },
            }],
        })

        captured: dict = {}

        async def fake_run_session(**kwargs):
            captured.update(kwargs)
            return AgentResult(
                response_text="window = [100, 500]",
                session_id="x", num_turns=1, total_cost_usd=0.0, usage={},
            )

        # Mock get_claude_version to avoid the subprocess call.
        with patch("agent_retrieval.runner.run.get_claude_version", return_value="claude 1.0"), \
             patch(
                 "agent_retrieval.runner.run.run_agent_session",
                 side_effect=fake_run_session,
             ):
            await run_batch(batch, experiments_dir=tmp_workspace / "experiments",
                            workspace_dir=workspace)

        # The session should have been invoked with the inlined question.
        assert "Inlined facts" in captured["question"]
        # corpus_dir for pure_reasoning is the run_dir (not the corpora_dir).
        # run_dir lives under workspace/runner/runs/<batch_run_name>/<pid>/<run_id>
        assert "runner/runs" in str(captured["corpus_dir"])
        assert "runner/corpora" not in str(captured["corpus_dir"])

    @pytest.mark.asyncio
    async def test_pure_reasoning_l2_uses_run_dir_as_cwd(self, tmp_workspace):
        """pure_reasoning_l2 runs use run_dir as cwd (no populated corpus required)."""
        workspace = tmp_workspace / "workspace"
        ak_path = workspace / "judge" / "answer_keys" / "pure_reasoning_l2__python_repo__n4.yaml"
        ak_path.write_text(yaml.dump({
            "parametrisation_id": "pure_reasoning_l2__python_repo__n4",
            "experiment_type": "pure_reasoning_l2",
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
            "batch_name": "test_pure_l2",
            "max_parallel": 1,
            "retry_failed": False,
            "agent_model": "claude-haiku-4-5-20251001",
            "effort_mode": "low",
            "n_repeats": 1,
            "max_turns": 5,
            "allowed_tools": [],
            "experiments": [{
                "experiment_type": "pure_reasoning_l2",
                "grid": {
                    "content_profile": ["python_repo"],
                    "n_items": [4],
                },
            }],
        })

        captured: dict = {}

        async def fake_run_session(**kwargs):
            captured.update(kwargs)
            return AgentResult(
                response_text="window = [100, 500]",
                session_id="x", num_turns=1, total_cost_usd=0.0, usage={},
            )

        with patch("agent_retrieval.runner.run.get_claude_version", return_value="claude 1.0"), \
             patch(
                 "agent_retrieval.runner.run.run_agent_session",
                 side_effect=fake_run_session,
             ):
            await run_batch(batch, experiments_dir=tmp_workspace / "experiments",
                            workspace_dir=workspace)

        assert "Inlined facts" in captured["question"]
        assert "runner/runs" in str(captured["corpus_dir"])
        assert "runner/corpora" not in str(captured["corpus_dir"])

    @pytest.mark.asyncio
    async def test_pure_reasoning_l3_uses_run_dir_as_cwd(self, tmp_workspace):
        """pure_reasoning_l3 runs use run_dir as cwd (no populated corpus required)."""
        workspace = tmp_workspace / "workspace"
        ak_path = workspace / "judge" / "answer_keys" / "pure_reasoning_l3__python_repo__n4.yaml"
        ak_path.write_text(yaml.dump({
            "parametrisation_id": "pure_reasoning_l3__python_repo__n4",
            "experiment_type": "pure_reasoning_l3",
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
            "batch_name": "test_pure_l3",
            "max_parallel": 1,
            "retry_failed": False,
            "agent_model": "claude-haiku-4-5-20251001",
            "effort_mode": "low",
            "n_repeats": 1,
            "max_turns": 5,
            "allowed_tools": [],
            "experiments": [{
                "experiment_type": "pure_reasoning_l3",
                "grid": {
                    "content_profile": ["python_repo"],
                    "n_items": [4],
                },
            }],
        })

        captured: dict = {}

        async def fake_run_session(**kwargs):
            captured.update(kwargs)
            return AgentResult(
                response_text="window = [100, 500]",
                session_id="x", num_turns=1, total_cost_usd=0.0, usage={},
            )

        with patch("agent_retrieval.runner.run.get_claude_version", return_value="claude 1.0"), \
             patch(
                 "agent_retrieval.runner.run.run_agent_session",
                 side_effect=fake_run_session,
             ):
            await run_batch(batch, experiments_dir=tmp_workspace / "experiments",
                            workspace_dir=workspace)

        assert "Inlined facts" in captured["question"]
        assert "runner/runs" in str(captured["corpus_dir"])
        assert "runner/corpora" not in str(captured["corpus_dir"])
