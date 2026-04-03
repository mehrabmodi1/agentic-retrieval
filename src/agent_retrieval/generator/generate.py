from __future__ import annotations

from pathlib import Path

from agent_retrieval.generator.assembler import assemble_corpus
from agent_retrieval.generator.background import BackgroundGenerator
from agent_retrieval.generator.grid import expand_grid
from agent_retrieval.generator.insertion import insert_payloads
from agent_retrieval.generator.payload import PayloadInserter
from agent_retrieval.generator.pool import generate_pool
from agent_retrieval.schema.answer_key import AnswerKey
from agent_retrieval.schema.experiment import ExperimentSpec
from agent_retrieval.schema.template import ExperimentTemplate


async def generate_experiment(
    spec: ExperimentSpec,
    workspace_dir: Path,
    skip_existing: bool = True,
) -> AnswerKey:
    corpus_dir = workspace_dir / "runner" / "corpora" / spec.experiment_id
    answer_key_path = workspace_dir / "judge" / "answer_keys" / f"{spec.experiment_id}.yaml"

    if skip_existing and corpus_dir.exists() and answer_key_path.exists():
        return AnswerKey.from_yaml(answer_key_path)

    bg = BackgroundGenerator()
    await bg.generate(spec, corpus_dir, red_herring_hint=spec.payload.red_herring_hint)

    inserter = PayloadInserter()
    answer_key = await inserter.insert(spec, corpus_dir, answer_key_path)
    return answer_key


async def generate_experiment_v2(
    template: ExperimentTemplate,
    workspace_dir: Path,
    skip_existing: bool = True,
) -> list[str]:
    """Generate all parametrisations for a v2 experiment template.

    Returns list of parametrisation IDs that were generated.
    """
    parametrisations = expand_grid(template)
    generated_ids: list[str] = []

    # Phase 1: Ensure background pools exist for all needed profiles
    profiles_needed = {p.content_profile for p in parametrisations}
    for profile_name in profiles_needed:
        pool_dir = workspace_dir / "runner" / "pools" / profile_name
        if pool_dir.exists() and any(pool_dir.rglob("*.md")):
            print(f"Background pool for '{profile_name}' already exists, skipping.")
            continue
        print(f"Ensuring background pool for '{profile_name}'...")
        await generate_pool(profile_name, pool_dir)

    # Phase 2 & 3: Assemble corpus and insert payloads per parametrisation
    for param in parametrisations:
        pid = param.parametrisation_id
        corpus_dir = workspace_dir / "runner" / "corpora" / pid
        answer_key_path = workspace_dir / "judge" / "answer_keys" / f"{pid}.yaml"

        if skip_existing and corpus_dir.exists() and answer_key_path.exists():
            print(f"  Skipping {pid} (already exists)")
            continue

        # Phase 2: Assemble corpus from pool
        pool_dir = workspace_dir / "runner" / "pools" / param.content_profile
        print(f"  Assembling corpus for {pid}...")
        assemble_corpus(pool_dir, corpus_dir, param)

        # Phase 3: Insert payloads
        print(f"  Inserting payloads for {pid}...")
        await insert_payloads(template, param, corpus_dir, answer_key_path)

        generated_ids.append(pid)
        print(f"  Done: {pid}")

    return generated_ids
