from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from pathlib import Path

from agent_retrieval.generator.llm_client import get_llm_client
from agent_retrieval.schema.answer_key import AnswerKey, AnswerKeyItem, ExpectedAnswers
from agent_retrieval.schema.experiment import ExperimentSpec, PayloadItem


def _resolve_insertion_order(items: list[PayloadItem]) -> list[PayloadItem]:
    """Topological sort: items with no deps first, then dependents."""
    id_to_item = {item.item_id: item for item in items}
    result: list[PayloadItem] = []
    visited: set[str] = set()

    def visit(item: PayloadItem) -> None:
        if item.item_id in visited:
            return
        if item.depends_on and item.depends_on not in visited:
            visit(id_to_item[item.depends_on])
        visited.add(item.item_id)
        result.append(item)

    for item in items:
        visit(item)

    return result


def _select_target_file(
    item: PayloadItem,
    corpus_dir: Path,
    used_files: set[Path],
    rng: random.Random,
) -> Path:
    """Select a file from corpus_dir based on the item's placement strategy."""
    all_files = [f for f in corpus_dir.rglob("*") if f.is_file()]
    strategy = item.placement.strategy

    if strategy == "specific_filetype" and item.placement.filetype:
        ext = item.placement.filetype
        candidates = [f for f in all_files if f.suffix == ext and f not in used_files]
        if not candidates:
            candidates = [f for f in all_files if f.suffix == ext]
    elif strategy == "specific_depth" and item.placement.depth is not None:
        target_depth = item.placement.depth
        candidates = [
            f for f in all_files
            if len(f.relative_to(corpus_dir).parts) == target_depth and f not in used_files
        ]
        if not candidates:
            candidates = [
                f for f in all_files
                if len(f.relative_to(corpus_dir).parts) == target_depth
            ]
    else:
        candidates = [f for f in all_files if f not in used_files]

    if not candidates:
        candidates = all_files

    return rng.choice(candidates)


class PayloadInserter:
    async def insert(
        self,
        spec: ExperimentSpec,
        corpus_dir: Path,
        answer_key_path: Path,
    ) -> AnswerKey:
        client = get_llm_client()
        ordered_items = _resolve_insertion_order(spec.payload.items)
        rng = random.Random(hash(spec.experiment_id))

        used_files: set[Path] = set()
        answer_key_items: list[AnswerKeyItem] = []
        insertion_results: list[dict] = []

        for item in ordered_items:
            target_file = _select_target_file(item, corpus_dir, used_files, rng)
            used_files.add(target_file)
            existing_content = target_file.read_text()

            dep_context = ""
            if item.depends_on:
                dep_item = next(
                    (r for r in insertion_results if r["item_id"] == item.depends_on), None
                )
                if dep_item:
                    dep_context = (
                        f"\nDependency context: This item depends on '{item.depends_on}'. "
                        f"The dependent item inserted: {dep_item['inserted_text']} "
                        f"in file {dep_item['file_path']}."
                    )

            prompt = (
                f"You are inserting a payload item into a source file for a retrieval benchmark.\n\n"
                f"Item ID: {item.item_id}\n"
                f"Item type: {item.item_type}\n"
                f"Content hint: {item.content_hint}\n"
                f"Camouflage level: {item.camouflage}\n"
                f"{dep_context}\n\n"
                f"Existing file content ({target_file.name}):\n"
                f"```\n{existing_content}\n```\n\n"
                f"Insert the payload naturally into the file. "
                f"For 'low' camouflage the insertion should be obvious; "
                f"for 'medium' it should blend in somewhat; "
                f"for 'high' it should be very subtle.\n\n"
                f"Return ONLY a JSON object with these keys:\n"
                f"- modified_content: the full file content with insertion\n"
                f"- inserted_text: just the inserted snippet\n"
                f"- line_range: [start_line, end_line] (1-indexed)\n"
                f"- context_summary: one sentence describing where/how it was inserted\n\n"
                f"Return ONLY valid JSON, no markdown fences."
            )

            response = await client.messages.create(
                model=spec.payload.insertion_model,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text.strip()
            data = json.loads(raw)

            target_file.write_text(data["modified_content"])

            rel_path = str(target_file.relative_to(corpus_dir))
            insertion_results.append({
                "item_id": item.item_id,
                "inserted_text": data["inserted_text"],
                "file_path": rel_path,
                "line_range": data["line_range"],
                "context_summary": data["context_summary"],
            })

            answer_key_items.append(
                AnswerKeyItem(
                    item_id=item.item_id,
                    inserted_text=data["inserted_text"],
                    file_path=rel_path,
                    line_range=data["line_range"],
                    context_summary=data["context_summary"],
                )
            )

        expected_answers = await self._generate_expected_answers(
            spec, insertion_results, client
        )

        answer_key = AnswerKey(
            parametrisation_id=spec.experiment_id,
            generated_at=datetime.now(timezone.utc).isoformat(),
            items=answer_key_items,
            expected_answers=expected_answers,
            rubric_criteria=spec.rubric_criteria,
        )
        answer_key.to_yaml(answer_key_path)
        return answer_key

    async def _generate_expected_answers(
        self,
        spec: ExperimentSpec,
        insertion_results: list[dict],
        client,
    ) -> ExpectedAnswers:
        insertions_summary = "\n".join(
            f"- {r['item_id']}: '{r['inserted_text']}' in {r['file_path']}"
            for r in insertion_results
        )
        prompt = (
            f"An agent will be asked: {spec.question}\n\n"
            f"The following items were inserted into a codebase:\n"
            f"{insertions_summary}\n\n"
            f"Generate expected answer criteria. Return ONLY a JSON object with:\n"
            f"- correctness: what a correct answer must include\n"
            f"- completeness: what a complete answer covers\n\n"
            f"Return ONLY valid JSON, no markdown fences."
        )

        response = await client.messages.create(
            model=spec.payload.insertion_model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()
        data = json.loads(raw)

        return ExpectedAnswers(
            question=spec.question,
            correctness=data.get("correctness", ""),
            completeness=data.get("completeness", ""),
        )
