from __future__ import annotations

import random
from pathlib import Path

from agent_retrieval.generator.llm_client import get_llm_client
from agent_retrieval.generator.profiles.base import GenerationContext
from agent_retrieval.generator.profiles.registry import get_profile
from agent_retrieval.schema.experiment import ExperimentSpec

_RED_HERRING_DENSITY_MAP: dict[str, float] = {
    "none": 0.0,
    "low": 0.1,
    "medium": 0.25,
    "high": 0.5,
}


class BackgroundGenerator:
    async def generate(
        self,
        spec: ExperimentSpec,
        corpus_dir: Path,
        red_herring_hint: str | None = None,
    ) -> list[Path]:
        profile = get_profile(spec.corpus.content_profile)
        file_paths = profile.generate_folder_structure(spec.corpus)

        density = _RED_HERRING_DENSITY_MAP.get(spec.corpus.red_herring_density, 0.0)
        rng = random.Random(hash(spec.experiment_id))
        red_herring_set: set[int] = set()
        if density > 0.0 and red_herring_hint:
            count = max(1, round(len(file_paths) * density))
            indices = rng.sample(range(len(file_paths)), min(count, len(file_paths)))
            red_herring_set = set(indices)

        client = get_llm_client()
        created: list[Path] = []

        for i, rel_path in enumerate(file_paths):
            is_red_herring = i in red_herring_set
            context = GenerationContext(
                corpus_spec=spec.corpus,
                red_herring_hint=red_herring_hint,
                is_red_herring_file=is_red_herring,
            )
            prompt = profile.generate_file_prompt(rel_path, context)
            response = await client.messages.create(
                model=spec.corpus.generation_model,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
            )
            content = response.content[0].text

            abs_path = corpus_dir / rel_path
            abs_path.parent.mkdir(parents=True, exist_ok=True)
            abs_path.write_text(content)
            created.append(abs_path)

        return created
