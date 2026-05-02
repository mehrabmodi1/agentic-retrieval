"""One-shot script to back-fill the hallucination-penalty wording into the
already-generated multi_retrieval and pure_reasoning answer keys.

Updates ONLY expected_answers.correctness and expected_answers.completeness.
All other fields are preserved exactly.
"""
from __future__ import annotations

from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Project root relative to this script
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
AK_DIR = PROJECT_ROOT / "workspace" / "judge" / "answer_keys"


# ---------------------------------------------------------------------------
# New wording builders
# ---------------------------------------------------------------------------

def _multi_retrieval_correctness(n: int) -> str:
    return (
        f"Each of the {n} items must be reported with both the verbatim "
        f"inserted_text and the corresponding value, matching the "
        f"answer-key items. Items in the agent's response whose verbatim "
        f"text does not appear in the answer-key item list are FALSE "
        f"POSITIVES; if the agent also cites a specific file_path or line "
        f"number for a false-positive item, treat it as a HALLUCINATION "
        f"(the agent claimed verbatim content that does not exist) and "
        f"penalize it under the precision criterion as strictly as a missed "
        f"item is penalized under recall."
    )


def _multi_retrieval_completeness(n: int) -> str:
    return (
        f"All {n} items must be reported. Recall is the fraction of "
        f"answer-key items correctly reported (verbatim text + value "
        f"match). Hallucinated items in the response do not count toward "
        f"recall."
    )


def _pure_reasoning_correctness(
    n: int,
    max_lower_text: str,
    max_lower_val: str,
    min_upper_text: str,
    min_upper_val: str,
) -> str:
    return (
        f"The answer must derive the narrowest valid window. The lower "
        f"endpoint is established by the fact whose bound_direction is "
        f"'lower' and bound_value is largest: {max_lower_text!r} "
        f"(value: {max_lower_val}). The upper endpoint is established "
        f"by the fact whose bound_direction is 'upper' and bound_value "
        f"is smallest: {min_upper_text!r} (value: {min_upper_val}). "
        f"The agent must cite both facts. "
        f"The agent's cited evidence must come from the {n} facts provided "
        f"in the question. Any cited fact that is not verbatim one of the "
        f"{n} provided facts is a HALLUCINATION and must be penalized under "
        f"the endpoint_correctness criterion."
    )


def _pure_reasoning_completeness(n: int) -> str:
    return (
        f"The agent must classify each of the {n} facts as a lower or "
        f"upper bound consistent with the answer-key bound_direction "
        f"field, OR justify any deviation in the reasoning. "
        f"Classifications attached to facts not actually present in the "
        f"question prompt are hallucinations and reduce "
        f"classification_accuracy."
    )


# ---------------------------------------------------------------------------
# Helpers for pure_reasoning endpoint re-derivation
# ---------------------------------------------------------------------------

def _to_comparable(value: str) -> float:
    """Mirror of pure_reasoning_gen._to_comparable — no import needed."""
    s = value.strip()
    try:
        return float(s)
    except ValueError:
        pass
    s_upper = s.upper()
    is_pm = "PM" in s_upper
    is_am = "AM" in s_upper
    if not (is_pm or is_am):
        raise ValueError(f"unrecognised bound_value: {value!r}")
    s_clean = s_upper.replace("PM", "").replace("AM", "").strip()
    if ":" in s_clean:
        h_str, m_str = s_clean.split(":", 1)
        h = int(h_str)
        m = int(m_str)
    else:
        h = int(s_clean)
        m = 0
    if is_pm and h != 12:
        h += 12
    elif is_am and h == 12:
        h = 0
    return float(h * 60 + m)


def _compute_endpoints(items: list[dict]) -> tuple[str, str, str, str]:
    """Return (max_lower_val, max_lower_text, min_upper_val, min_upper_text)."""
    lowers = [it for it in items if it.get("bound_direction") == "lower"]
    uppers = [it for it in items if it.get("bound_direction") == "upper"]
    if not lowers or not uppers:
        return "", "", "", ""
    max_lower = max(lowers, key=lambda it: _to_comparable(it["value"]))
    min_upper = min(uppers, key=lambda it: _to_comparable(it["value"]))
    return (
        max_lower["value"],
        max_lower["inserted_text"],
        min_upper["value"],
        min_upper["inserted_text"],
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def refresh_multi_retrieval(path: Path) -> bool:
    text = path.read_text()
    ak = yaml.safe_load(text)
    n = len(ak["items"])
    new_correctness = _multi_retrieval_correctness(n)
    new_completeness = _multi_retrieval_completeness(n)
    if (
        ak["expected_answers"].get("correctness") == new_correctness
        and ak["expected_answers"].get("completeness") == new_completeness
    ):
        return False  # already up-to-date (idempotent)
    ak["expected_answers"]["correctness"] = new_correctness
    ak["expected_answers"]["completeness"] = new_completeness
    with open(path, "w") as f:
        yaml.dump(ak, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    return True


def refresh_pure_reasoning(path: Path) -> bool:
    text = path.read_text()
    ak = yaml.safe_load(text)
    n = len(ak["items"])
    max_lower_val, max_lower_text, min_upper_val, min_upper_text = _compute_endpoints(ak["items"])
    new_correctness = _pure_reasoning_correctness(
        n, max_lower_text, max_lower_val, min_upper_text, min_upper_val
    )
    new_completeness = _pure_reasoning_completeness(n)
    if (
        ak["expected_answers"].get("correctness") == new_correctness
        and ak["expected_answers"].get("completeness") == new_completeness
    ):
        return False  # already up-to-date (idempotent)
    ak["expected_answers"]["correctness"] = new_correctness
    ak["expected_answers"]["completeness"] = new_completeness
    with open(path, "w") as f:
        yaml.dump(ak, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    return True


def main() -> None:
    updated = []

    for path in sorted(AK_DIR.glob("multi_retrieval__*.yaml")):
        if refresh_multi_retrieval(path):
            updated.append(path.name)
            print(f"  updated: {path.name}")

    for path in sorted(AK_DIR.glob("pure_reasoning__*.yaml")):
        if refresh_pure_reasoning(path):
            updated.append(path.name)
            print(f"  updated: {path.name}")

    print(f"\nTotal files updated: {len(updated)}")


if __name__ == "__main__":
    main()
