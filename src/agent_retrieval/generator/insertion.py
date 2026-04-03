from __future__ import annotations

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


def build_insertion_prompt(
    template: ExperimentTemplate,
    parametrisation: Parametrisation,
    answer_key_path: Path,
) -> str:
    n_items = parametrisation.n_items or 1
    exp_type = parametrisation.experiment_type

    type_instructions = {
        "single_needle": (
            "Insert exactly 1 needle into the corpus. "
            "Generate a question that asks about the inserted information."
        ),
        "multi_chain": (
            f"Insert a chain of {n_items} linked needles. Each needle must reference or point to "
            f"the next one in the chain. The question must provide the entry point (first needle's "
            f"location) and ask what the chain ultimately resolves to. Each link should be in a "
            f"different file. Previous links can be forgotten after following — only the final "
            f"value matters."
        ),
        "multi_reasoning": (
            f"Insert {n_items} independent needles, each containing a distinct piece of information. "
            f"The question must require finding ALL items and combining/reasoning about them to "
            f"produce the answer. Items should be in different files."
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
        f"## Instructions\n"
        f"1. Browse the corpus to understand its structure and content style\n"
        f"2. Choose {n_items} file(s) to insert needles into\n"
        f"3. Read each target file to understand the surrounding context\n"
        f"4. Insert each needle so it reads naturally within the file\n"
        f"5. Generate a question and expected answer\n"
        f"6. Write the answer key as valid YAML to: {answer_key_path}\n\n"
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

    system_prompt = build_insertion_prompt(template, parametrisation, answer_key_path)

    options = ClaudeAgentOptions(
        model="claude-sonnet-4-6",
        system_prompt=system_prompt,
        cwd=str(corpus_dir),
        allowed_tools=["Read", "Edit", "Write", "Glob", "Grep"],
        permission_mode="acceptEdits",
        max_turns=100,
    )

    prompt = (
        "Begin by browsing the corpus to understand its structure. "
        "Then insert the needle(s) and write the answer key."
    )

    async for message in query(prompt=prompt, options=options):
        if isinstance(message, ResultMessage):
            break
