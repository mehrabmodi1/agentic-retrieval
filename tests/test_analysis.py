import yaml
from pathlib import Path
import pandas as pd
import pytest
from agent_retrieval.analysis.loader import load_batch_results
from agent_retrieval.analysis.tables import accuracy_by_type, accuracy_by_param, tool_usage_by_type


@pytest.fixture
def populated_workspace(tmp_workspace) -> Path:
    ws = tmp_workspace
    for exp_id, exp_type, token_count in [
        ("needle-001", "needle_in_haystack", 100_000),
        ("needle-002", "needle_in_haystack", 500_000),
        ("chain-001", "chain_of_retrieval", 100_000),
    ]:
        spec = {
            "schema_version": "1.0", "experiment_id": exp_id, "experiment_type": exp_type,
            "corpus": {"content_profile": "python_repo", "target_token_count": token_count,
                       "target_file_count": 50, "folder_depth": 3, "folder_distribution": "balanced",
                       "generation_model": "haiku", "red_herring_density": "medium"},
            "payload": {"insertion_model": "sonnet", "red_herring_hint": "Similar values",
                        "items": [{"item_id": "t1", "item_type": "config_value",
                                   "content_hint": "a value", "placement": {"strategy": "random_file"},
                                   "camouflage": "medium"}]},
            "question": "Find the value",
            "rubric_criteria": [{"criterion": "correctness", "weight": 1.0}],
            "runner": {"n_repeats": 2, "agent_model": "sonnet", "max_tokens": 50000,
                       "allowed_tools": ["Read", "Grep"]},
        }
        spec_path = ws / "specs" / f"{exp_id}.yaml"
        spec_path.parent.mkdir(parents=True, exist_ok=True)
        spec_path.write_text(yaml.dump(spec))

    batch_name = "test-batch"
    for exp_id, scores in [("needle-001", [0.9, 0.85]), ("needle-002", [0.7, 0.6]), ("chain-001", [0.95, 0.9])]:
        for i, score in enumerate(scores):
            verdict = {
                "parametrisation_id": exp_id, "run_id": f"run_{i}", "batch_name": batch_name,
                "scores": [{"criterion": "correctness", "score": score, "weight": 1.0, "reasoning": "test"}],
                "weighted_score": score,
                "session_metrics": {"total_context_tokens": 50000 + i * 10000, "total_turns": 5 + i,
                                    "tool_calls": {"Grep": 3 + i, "Read": 2 + i}, "duration_seconds": 30.0 + i * 5},
            }
            verdict_path = ws / "workspace" / "judge" / "judgements" / batch_name / exp_id / f"run_{i}.yaml"
            verdict_path.parent.mkdir(parents=True, exist_ok=True)
            verdict_path.write_text(yaml.dump(verdict))
    return ws


class TestLoadBatchResults:
    def test_loads_all_verdicts_into_dataframe(self, populated_workspace):
        df = load_batch_results("test-batch",
            workspace_dir=populated_workspace / "workspace", specs_dir=populated_workspace / "specs")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 6
        assert "experiment_type" in df.columns
        assert "weighted_score" in df.columns
        assert "target_token_count" in df.columns


class TestAccuracyByType:
    def test_groups_by_experiment_type(self, populated_workspace):
        df = load_batch_results("test-batch",
            workspace_dir=populated_workspace / "workspace", specs_dir=populated_workspace / "specs")
        result = accuracy_by_type(df)
        assert len(result) == 2
        assert "mean" in result.columns
        assert "std" in result.columns


class TestAccuracyByParam:
    def test_groups_by_type_and_param(self, populated_workspace):
        df = load_batch_results("test-batch",
            workspace_dir=populated_workspace / "workspace", specs_dir=populated_workspace / "specs")
        result = accuracy_by_param(df, param_column="target_token_count")
        assert len(result) > 0


class TestToolUsageByType:
    def test_returns_tool_counts(self, populated_workspace):
        df = load_batch_results("test-batch",
            workspace_dir=populated_workspace / "workspace", specs_dir=populated_workspace / "specs")
        result = tool_usage_by_type(df)
        assert "Grep" in result.columns or "Grep" in str(result.columns)


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
            "parametrisation_id": "single_needle__python_repo__20k__easy__exact",
            "run_id": "run001",
            "batch_name": batch_name,
            "scores": [{"criterion": "correctness", "score": 0.9, "weight": 1.0, "reasoning": "Good"}],
            "weighted_score": 0.9,
            "session_metrics": {"total_context_tokens": 5000, "total_turns": 3, "tool_calls": {"Grep": 2}, "duration_seconds": 10.0},
        }))

        df = load_batch_results(batch_name, workspace_dir=workspace, specs_dir=tmp_path / "specs")
        assert len(df) == 1
        assert df.iloc[0]["content_profile"] == "python_repo"
        assert df.iloc[0]["corpus_token_count"] == 20000
        assert df.iloc[0]["discriminability"] == "easy"
        assert df.iloc[0]["reference_clarity"] == "exact"
        assert df.iloc[0]["experiment_type"] == "single_needle"
