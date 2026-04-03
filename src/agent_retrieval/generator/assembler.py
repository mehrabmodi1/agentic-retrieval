from __future__ import annotations

import random
import shutil
from pathlib import Path

from agent_retrieval.schema.template import Parametrisation


def assemble_corpus(
    pool_dir: Path,
    corpus_dir: Path,
    parametrisation: Parametrisation,
) -> None:
    if corpus_dir.exists() and any(corpus_dir.rglob("*.md")):
        return

    corpus_dir.mkdir(parents=True, exist_ok=True)

    all_files = sorted(f for f in pool_dir.rglob("*.md") if f.is_file())
    if not all_files:
        return

    rng = random.Random(hash(parametrisation.parametrisation_id))
    rng.shuffle(all_files)

    budget = parametrisation.corpus_token_count
    accumulated_tokens = 0

    for src_file in all_files:
        content = src_file.read_text()
        file_tokens = len(content) // 4

        rel_path = src_file.relative_to(pool_dir)
        dest = corpus_dir / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_file, dest)

        accumulated_tokens += file_tokens
        if accumulated_tokens >= budget:
            break
