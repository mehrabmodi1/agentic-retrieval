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
