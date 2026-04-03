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
        state_path = runs_dir / "test-batch" / "test-001" / run_ids[0] / "state.yaml"
        state = RunState.from_yaml(state_path)
        state.status = "completed"
        state.to_yaml(state_path)
        pending = mgr.get_runs_by_status(batch_config.batch_name, "test-001", "pending")
        assert len(pending) == 1
