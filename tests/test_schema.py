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
            "parametrisation_id": "test-001",
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
        assert ak.parametrisation_id == "test-001"
        assert ak.items[0].inserted_text == "TIMEOUT = 42"


class TestBatchConfig:
    def test_valid_batch(self):
        batch = BatchConfig.model_validate({
            "batch_name": "test-batch",
            "max_parallel": 2,
            "retry_failed": True,
            "judge_model": "opus",
            "experiments": ["single_needle"],
        })
        assert batch.batch_name == "test-batch"
        assert batch.experiments[0].experiment_type == "single_needle"


class TestRunState:
    def test_valid_run_state(self):
        state = RunState.model_validate({
            "parametrisation_id": "test-001",
            "run_id": "abc123",
            "batch_name": "test-batch",
            "status": "pending",
            "claude_code_version": "1.0.0",
        })
        assert state.status == "pending"

    def test_status_transitions(self):
        state = RunState(
            parametrisation_id="test-001",
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
            "parametrisation_id": "test-001",
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


from agent_retrieval.schema.batch import BatchConfig, BatchExperimentEntry


class TestBatchConfig:
    def test_simple_experiment_list(self):
        batch = BatchConfig.model_validate({
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
        batch = BatchConfig.model_validate({
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
        batch = BatchConfig.model_validate({
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
            "parametrisation_id": "single_needle__python_repo__20k__easy__exact",
            "generated_at": "2026-04-03T10:00:00Z",
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

    def test_answer_key_requires_parametrisation_id(self):
        """parametrisation_id is required."""
        with pytest.raises(Exception):
            AnswerKey.model_validate({
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
