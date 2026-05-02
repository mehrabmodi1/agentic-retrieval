from __future__ import annotations

from pathlib import Path

from agent_retrieval.generator.assembler import assemble_corpus
from agent_retrieval.generator.corpus_files import iter_corpus_files
from agent_retrieval.generator.grid import expand_grid
from agent_retrieval.generator.insertion import insert_payloads
from agent_retrieval.generator.insertion_fixed import insert_fixed_payloads
from agent_retrieval.generator.pool import generate_pool
from agent_retrieval.generator.pure_reasoning_gen import generate_pure_reasoning_cell
from agent_retrieval.schema.template import ExperimentTemplate


async def generate_experiment_v2(
    template: ExperimentTemplate,
    workspace_dir: Path,
    skip_existing: bool = True,
) -> list[str]:
    parametrisations = expand_grid(template)
    generated_ids: list[str] = []

    if template.experiment_type == "pure_reasoning":
        # No corpus, no insertion agent — just write answer keys.
        for param in parametrisations:
            pid = param.parametrisation_id
            answer_key_path = workspace_dir / "judge" / "answer_keys" / f"{pid}.yaml"
            if skip_existing and answer_key_path.exists():
                print(f"  Skipping {pid} (already exists)")
                continue
            generate_pure_reasoning_cell(
                template=template,
                parametrisation=param,
                answer_key_path=answer_key_path,
            )
            generated_ids.append(pid)
            print(f"  Done: {pid}")
        return generated_ids

    # Corpus-based experiments (single_needle, multi_chain, multi_reasoning, multi_retrieval).
    profiles_needed = {p.content_profile for p in parametrisations}
    for profile_name in profiles_needed:
        pool_dir = workspace_dir / "background_corpora" / profile_name
        if pool_dir.exists() and any(iter_corpus_files(pool_dir)):
            print(f"Background pool for '{profile_name}' already exists, skipping.")
            continue
        print(f"Ensuring background pool for '{profile_name}'...")
        await generate_pool(profile_name, pool_dir)

    for param in parametrisations:
        pid = param.parametrisation_id
        corpus_dir = workspace_dir / "runner" / "corpora" / pid
        answer_key_path = workspace_dir / "judge" / "answer_keys" / f"{pid}.yaml"

        if skip_existing and corpus_dir.exists() and answer_key_path.exists():
            print(f"  Skipping {pid} (already exists)")
            continue

        pool_dir = workspace_dir / "background_corpora" / param.content_profile
        print(f"  Assembling corpus for {pid}...")
        assemble_corpus(pool_dir, corpus_dir, param)

        print(f"  Inserting payloads for {pid}...")
        if template.experiment_type == "multi_retrieval":
            await insert_fixed_payloads(template, param, corpus_dir, answer_key_path)
        else:
            await insert_payloads(template, param, corpus_dir, answer_key_path)

        generated_ids.append(pid)
        print(f"  Done: {pid}")

    return generated_ids
