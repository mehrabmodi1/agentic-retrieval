from __future__ import annotations

import random
from pathlib import Path

from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, query

from agent_retrieval.schema.template import ExperimentTemplate, Parametrisation


DISCRIMINABILITY_RUBRIC = """## Discriminability Rubric

| Level | Retrieval method | Cognitive demand |
|---|---|---|
| easy | Findable by exact string search on terms in the question | Locate and report |
| hard | Embedded in surrounding context; requires reading and comprehending surrounding content to identify | Comprehend context, extract implicit information |
"""

REFERENCE_CLARITY_RUBRIC = """## Reference Clarity

| Level | Definition |
|---|---|
| exact | Question uses the same identifier/name as the needle |
| synonym | Question uses a different term for the same concept |
| contextual | Question describes the concept indirectly; requires domain understanding to connect |
"""

ANSWER_KEY_SCHEMA = """parametrisation_id: "{parametrisation_id}"
experiment_type: "{experiment_type}"
generated_at: "<ISO timestamp>"
parameters:
  content_profile: "{content_profile}"
  corpus_token_count: {corpus_token_count}
  discriminability: "{discriminability}"
  reference_clarity: "{reference_clarity}"
items:
  - item_id: "target_001"
    inserted_text: "<the exact text you inserted>"
    file_path: "<relative path to the file>"
    line_range: [<start_line>, <end_line>]
    context_summary: "<one sentence describing where/how it was inserted>"
expected_answers:
  question: "<the question you generated>"
  correctness: "<what a correct answer must include>"
  completeness: "<what a complete answer covers>"
rubric_criteria:
{rubric_criteria_yaml}"""


def _format_examples(
    template: ExperimentTemplate,
    parametrisation: Parametrisation,
) -> str:
    profile = parametrisation.content_profile
    disc = parametrisation.discriminability
    ref = parametrisation.reference_clarity

    profile_examples = template.question_examples.get(profile, {})
    key = f"{disc}_{ref}"
    example = profile_examples.get(key)

    if example is None:
        # Fall back to any example for this profile
        for k, v in profile_examples.items():
            example = v
            break

    if example is None:
        return "No examples available."

    lines = [f"**Question:** {example.question}", f"**Answer:** {example.answer}"]
    if example.needle:
        lines.insert(1, f"**Needle:** {example.needle}")
    if example.chain:
        chain_str = "\n".join(
            f"  - [{c['file_context']}] {c['needle']}" for c in example.chain
        )
        lines.insert(1, f"**Chain:**\n{chain_str}")
    if example.items:
        items_str = "\n".join(
            f"  - [{it['file_context']}] {it['needle']}" for it in example.items
        )
        lines.insert(1, f"**Items:**\n{items_str}")

    return "\n".join(lines)


def _select_target_files(
    corpus_dir: Path,
    parametrisation: Parametrisation,
    n_files: int,
) -> list[Path]:
    """Deterministically select target files from corpus using RNG."""
    all_files = sorted(f for f in corpus_dir.rglob("*.md") if f.is_file())
    if not all_files:
        return []
    rng = random.Random(hash(parametrisation.parametrisation_id) ^ 0xBEEF)
    return rng.sample(all_files, min(n_files, len(all_files)))


def _read_target_files(files: list[Path], corpus_dir: Path) -> str:
    """Read files and format as inline content for the prompt."""
    sections = []
    for f in files:
        rel = f.relative_to(corpus_dir)
        content = f.read_text()
        sections.append(f"### File: {rel}\n```\n{content}\n```")
    return "\n\n".join(sections)


def build_insertion_prompt(
    template: ExperimentTemplate,
    parametrisation: Parametrisation,
    answer_key_path: Path,
    target_files_content: str = "",
) -> str:
    n_items = parametrisation.n_items or 1
    exp_type = parametrisation.experiment_type

    type_instructions = {
        "single_needle": (
            "Insert exactly 1 needle into one of the target files below. "
            "Generate a question that asks about the inserted information."
        ),
        "multi_chain": (
            f"Insert a chain of {n_items} linked needles across different target files below. "
            f"Each needle must reference or point to the next one in the chain. The question "
            f"must provide the entry point (first needle's location) and ask what the chain "
            f"ultimately resolves to. Previous links can be forgotten after following — only "
            f"the final value matters."
        ),
        "multi_reasoning": (
            f"Insert {n_items} independent needles across different target files below. "
            f"Each needle contains a distinct piece of information. The question must require "
            f"finding ALL items and combining/reasoning about them to produce the answer."
        ),
    }

    examples_str = _format_examples(template, parametrisation)

    rubric_criteria_yaml = "\n".join(
        f'  - criterion: "{c.criterion}"\n    weight: {c.weight}'
        for c in template.rubric_criteria
    )

    answer_schema = ANSWER_KEY_SCHEMA.format(
        parametrisation_id=parametrisation.parametrisation_id,
        experiment_type=exp_type,
        content_profile=parametrisation.content_profile,
        corpus_token_count=parametrisation.corpus_token_count,
        discriminability=parametrisation.discriminability,
        reference_clarity=parametrisation.reference_clarity,
        rubric_criteria_yaml=rubric_criteria_yaml,
    )

    return (
        f"You are inserting needle(s) into a corpus for a retrieval experiment.\n\n"
        f"Experiment type: {exp_type}\n"
        f"Discriminability: {parametrisation.discriminability}\n"
        f"Reference clarity: {parametrisation.reference_clarity}\n"
        f"Number of items: {n_items}\n\n"
        f"{DISCRIMINABILITY_RUBRIC}\n"
        f"{REFERENCE_CLARITY_RUBRIC}\n"
        f"## Example\n{examples_str}\n\n"
        f"## Type-specific instructions\n{type_instructions[exp_type]}\n\n"
        f"## Target files (pre-selected — use these)\n{target_files_content}\n\n"
        f"## Instructions\n"
        f"The target files above have been pre-selected and pre-read for you.\n"
        f"Do NOT browse or read any other files. Work only with what is provided above.\n\n"
        f"1. Choose which of the target files above to insert needle(s) into\n"
        f"2. Use the Edit tool to insert each needle so it reads naturally within the file\n"
        f"3. Write the answer key YAML to: {answer_key_path}\n\n"
        f"The answer key MUST follow this exact structure:\n"
        f"```yaml\n{answer_schema}\n```\n\n"
        f"For multiple items, add additional entries under 'items:' with sequential "
        f"item_ids (target_001, target_002, etc.).\n"
    )


async def insert_payloads(
    template: ExperimentTemplate,
    parametrisation: Parametrisation,
    corpus_dir: Path,
    answer_key_path: Path,
) -> None:
    if answer_key_path.exists():
        return

    answer_key_path.parent.mkdir(parents=True, exist_ok=True)

    n_items = parametrisation.n_items or 1
    # Select more files than needed so the agent has choices
    n_target_files = max(n_items * 2, 4)
    target_files = _select_target_files(corpus_dir, parametrisation, n_target_files)
    target_files_content = _read_target_files(target_files, corpus_dir)

    system_prompt = build_insertion_prompt(
        template, parametrisation, answer_key_path, target_files_content,
    )

    options = ClaudeAgentOptions(
        model="claude-sonnet-4-6",
        system_prompt=system_prompt,
        cwd=str(corpus_dir),
        allowed_tools=["Edit", "Write"],
        permission_mode="acceptEdits",
        max_turns=10,
    )

    prompt = (
        "Insert the needle(s) into the target files provided above, "
        "then write the answer key. Do not read or browse any files."
    )

    async for message in query(prompt=prompt, options=options):
        if isinstance(message, ResultMessage):
            break
