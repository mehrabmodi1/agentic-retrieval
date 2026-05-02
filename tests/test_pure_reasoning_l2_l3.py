from pathlib import Path

import pytest

from agent_retrieval.generator.pure_reasoning_gen import generate_pure_reasoning_cell
from agent_retrieval.schema.answer_key import AnswerKey
from agent_retrieval.schema.template import ExperimentTemplate, Parametrisation


def _l2_template():
    """L2 template with 8 lower + 8 upper items, mixed variants."""
    items = []
    for i in range(8):
        items.append({
            "text": f"L_item_{i}_v{i % 3}",
            "bound_direction": "lower",
            "bound_value": str(100 + i * 10),
            "context_summary": f"lower {i}",
            "variant_id": i % 3,
            "variant_text": f"L_item_{i}_v{i % 3}",
        })
    for i in range(8):
        items.append({
            "text": f"U_item_{i}_v{i % 3}",
            "bound_direction": "upper",
            "bound_value": str(500 + i * 10),
            "context_summary": f"upper {i}",
            "variant_id": i % 3,
            "variant_text": f"U_item_{i}_v{i % 3}",
        })
    return ExperimentTemplate.model_validate({
        "experiment_type": "pure_reasoning_l2",
        "payload": {"item_type": "fact"},
        "question_examples": {
            "python_repo": {
                "hard_contextual": {
                    "question": (
                        "L2 prompt. Normalise to a common scale (seconds) before reasoning.\n"
                        "{n} facts:\n{facts_block}"
                    ),
                    "answer": "ok",
                },
            },
        },
        "rubric_criteria": [{"criterion": "endpoint_correctness", "weight": 1.0}],
        "grid": {"content_profile": ["python_repo"], "n_items": [4]},
        "fixed_pool": {"python_repo": items},
    })


def _l3_template_and_world_state():
    """L3 template with 4 quadrants of 4 items each."""
    items = []
    for live, direction, gate_var in [
        (True, "lower", "phase_rolling"),
        (True, "upper", "phase_rolling"),
        (False, "lower", "feature_x_off"),
        (False, "upper", "feature_x_off"),
    ]:
        for i in range(4):
            items.append({
                "text": f"if {gate_var} then {direction}_{i}",
                "bound_direction": direction,
                "bound_value": str((100 if direction == "lower" else 500) + i * 10),
                "context_summary": f"{direction} {i} live={live}",
                "variant_id": i % 3,
                "variant_text": f"{direction}_{i}",
                "gate_clause": f"if {gate_var} then",
                "gate_world_var": gate_var,
                "live": live,
            })
    template = ExperimentTemplate.model_validate({
        "experiment_type": "pure_reasoning_l3",
        "payload": {"item_type": "fact"},
        "question_examples": {
            "python_repo": {
                "hard_contextual": {
                    "question": (
                        "L3 prompt. World state:\n{world_state_block}\n"
                        "{n} facts:\n{facts_block}"
                    ),
                    "answer": "ok",
                },
            },
        },
        "rubric_criteria": [{"criterion": "endpoint_correctness", "weight": 1.0}],
        "grid": {"content_profile": ["python_repo"], "n_items": [4]},
        "fixed_pool": {"python_repo": items},
    })
    world_state = {
        "python_repo": {"phase_rolling": True, "feature_x_off": False}
    }
    return template, world_state


class TestL2Generation:
    def test_l2_writes_ak_with_unit_normalisation_in_question(self, tmp_path):
        template = _l2_template()
        param = Parametrisation(
            experiment_type="pure_reasoning_l2",
            content_profile="python_repo",
            n_items=4,
        )
        ak_path = tmp_path / "ak.yaml"
        generate_pure_reasoning_cell(
            template=template, parametrisation=param, answer_key_path=ak_path,
        )
        ak = AnswerKey.from_yaml(ak_path)
        assert ak.parametrisation_id == "pure_reasoning_l2__python_repo__n4"
        assert "Normalise to a common scale" in ak.expected_answers.question
        assert len(ak.items) == 4

    def test_l2_ak_items_carry_variant_metadata(self, tmp_path):
        """variant_id/variant_text must round-trip through the written YAML file."""
        import yaml
        template = _l2_template()
        param = Parametrisation(
            experiment_type="pure_reasoning_l2",
            content_profile="python_repo",
            n_items=4,
        )
        ak_path = tmp_path / "ak.yaml"
        generate_pure_reasoning_cell(
            template=template, parametrisation=param, answer_key_path=ak_path,
        )
        raw = yaml.safe_load(ak_path.read_text())
        for it in raw["items"]:
            assert "variant_id" in it
            assert "variant_text" in it


class TestL3Generation:
    def test_l3_n2_picks_only_live_items(self, tmp_path):
        import yaml
        template, world_state = _l3_template_and_world_state()
        param = Parametrisation(
            experiment_type="pure_reasoning_l3",
            content_profile="python_repo",
            n_items=2,
        )
        ak_path = tmp_path / "ak.yaml"
        generate_pure_reasoning_cell(
            template=template, parametrisation=param,
            answer_key_path=ak_path, world_state=world_state,
        )
        raw = yaml.safe_load(ak_path.read_text())
        for it in raw["items"]:
            assert it["live"] is True

    def test_l3_n4_covers_all_quadrants(self, tmp_path):
        import yaml
        template, world_state = _l3_template_and_world_state()
        param = Parametrisation(
            experiment_type="pure_reasoning_l3",
            content_profile="python_repo",
            n_items=4,
        )
        ak_path = tmp_path / "ak.yaml"
        generate_pure_reasoning_cell(
            template=template, parametrisation=param,
            answer_key_path=ak_path, world_state=world_state,
        )
        raw = yaml.safe_load(ak_path.read_text())
        quadrants = {(it["live"], it["bound_direction"]) for it in raw["items"]}
        assert quadrants == {(True, "lower"), (True, "upper"), (False, "lower"), (False, "upper")}

    def test_l3_question_includes_world_state_block(self, tmp_path):
        template, world_state = _l3_template_and_world_state()
        param = Parametrisation(
            experiment_type="pure_reasoning_l3",
            content_profile="python_repo",
            n_items=4,
        )
        ak_path = tmp_path / "ak.yaml"
        generate_pure_reasoning_cell(
            template=template, parametrisation=param,
            answer_key_path=ak_path, world_state=world_state,
        )
        ak = AnswerKey.from_yaml(ak_path)
        # World-state block must be substituted (placeholder absent) and contain
        # both world variables in the bullet format produced by _format_world_state_block.
        assert "{world_state_block}" not in ak.expected_answers.question
        assert "- phase_rolling = True" in ak.expected_answers.question
        assert "- feature_x_off = False" in ak.expected_answers.question

    def test_l3_correctness_mentions_live_items_only(self, tmp_path):
        template, world_state = _l3_template_and_world_state()
        param = Parametrisation(
            experiment_type="pure_reasoning_l3",
            content_profile="python_repo",
            n_items=4,
        )
        ak_path = tmp_path / "ak.yaml"
        generate_pure_reasoning_cell(
            template=template, parametrisation=param,
            answer_key_path=ak_path, world_state=world_state,
        )
        ak = AnswerKey.from_yaml(ak_path)
        assert "live" in ak.expected_answers.correctness.lower() or \
            "applicable" in ak.expected_answers.correctness.lower() or \
            "precondition" in ak.expected_answers.correctness.lower()


class TestL1StillWorks:
    """Make sure L1 (the existing pure_reasoning) still generates AKs identically."""
    def test_l1_unchanged(self, tmp_path):
        template = ExperimentTemplate.model_validate({
            "experiment_type": "pure_reasoning",
            "payload": {"item_type": "fact"},
            "question_examples": {
                "python_repo": {
                    "hard_contextual": {
                        "question": "L1: {n} facts:\n{facts_block}",
                        "answer": "ok",
                    },
                },
            },
            "rubric_criteria": [{"criterion": "endpoint_correctness", "weight": 1.0}],
            "grid": {"content_profile": ["python_repo"], "n_items": [2]},
            "fixed_pool": {
                "python_repo": [
                    {"text": "lower fact", "bound_direction": "lower", "bound_value": "100", "context_summary": "x"},
                    {"text": "upper fact", "bound_direction": "upper", "bound_value": "500", "context_summary": "y"},
                ],
            },
        })
        param = Parametrisation(
            experiment_type="pure_reasoning",
            content_profile="python_repo",
            n_items=2,
        )
        ak_path = tmp_path / "ak.yaml"
        generate_pure_reasoning_cell(
            template=template, parametrisation=param, answer_key_path=ak_path,
        )
        ak = AnswerKey.from_yaml(ak_path)
        assert ak.parametrisation_id == "pure_reasoning__python_repo__n2"
        assert len(ak.items) == 2
