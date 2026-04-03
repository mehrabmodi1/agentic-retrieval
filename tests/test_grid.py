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
    d["grid"] = d["grid"].copy()
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
