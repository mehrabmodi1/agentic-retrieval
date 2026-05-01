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
    d["grid"] = single_needle_dict["grid"].copy()
    d["grid"]["n_items"] = [2, 8, 16]
    return d


class TestExperimentTemplate:
    def test_valid_single_needle(self, single_needle_dict):
        tmpl = ExperimentTemplate.model_validate(single_needle_dict)
        assert tmpl.experiment_type == "single_needle"
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
