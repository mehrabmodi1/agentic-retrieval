from __future__ import annotations

from pathlib import Path

from agent_retrieval.generator.background import BackgroundGenerator
from agent_retrieval.generator.payload import PayloadInserter
from agent_retrieval.schema.answer_key import AnswerKey
from agent_retrieval.schema.experiment import ExperimentSpec


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
