from __future__ import annotations

import random
from pathlib import Path

from agent_retrieval.generator.profiles.base import ContentProfile, GenerationContext
from agent_retrieval.schema.experiment import CorpusSpec

PACKAGE_NAMES = [
    "auth", "api", "models", "services", "utils", "config",
    "handlers", "middleware", "core", "db", "cache", "tasks",
    "validators", "serializers", "exceptions", "logging_config",
]

FILE_STEMS = [
    "manager", "handler", "service", "client", "factory",
    "processor", "validator", "serializer", "helper", "adapter",
    "controller", "provider", "resolver", "transformer", "monitor",
    "scheduler", "dispatcher", "registry", "builder", "wrapper",
]


class PythonRepoProfile(ContentProfile):
    def generate_folder_structure(self, spec: CorpusSpec) -> list[Path]:
        rng = random.Random(hash(spec.content_profile + str(spec.target_file_count)))
        paths: list[Path] = []

        # Reserve a few top-level files
        top_level = ["README.md", "requirements.txt", "config.yaml"]
        for name in top_level:
            if len(paths) < spec.target_file_count:
                paths.append(Path(name))

        # Generate package directories up to folder_depth
        packages = rng.sample(PACKAGE_NAMES, min(len(PACKAGE_NAMES), spec.folder_depth * 3))
        dirs: list[Path] = []
        for i, pkg in enumerate(packages):
            if i < spec.folder_depth:
                dirs.append(Path("src") / pkg)
            else:
                parent = rng.choice(dirs[:max(1, len(dirs))])
                child = Path(parent) / pkg
                if len(child.parts) < spec.folder_depth:
                    dirs.append(child)

        # Fill directories with .py files
        while len(paths) < spec.target_file_count:
            d = rng.choice(dirs) if dirs else Path("src")
            stem = rng.choice(FILE_STEMS)
            suffix = f"_{rng.randint(1, 99)}" if rng.random() > 0.5 else ""
            filepath = d / f"{stem}{suffix}.py"
            if filepath not in paths:
                paths.append(filepath)

        return paths[:spec.target_file_count]

    def generate_file_prompt(self, path: Path, context: GenerationContext) -> str:
        file_type = path.suffix
        dir_name = path.parent.name if path.parent != Path(".") else "root"

        base_prompt = (
            f"Generate realistic Python source code for a file at '{path}' "
            f"in a '{dir_name}' package of a medium-sized web application. "
            f"Include imports, classes or functions, and docstrings. "
            f"Make it look like production code written by a competent developer. "
            f"The file should be 50-150 lines long. "
            f"Do not include any comments about this being generated."
        )

        if file_type == ".md":
            base_prompt = (
                f"Generate a realistic README.md for a Python web application project. "
                f"Include sections for setup, usage, and configuration. 50-100 lines."
            )
        elif file_type == ".yaml":
            base_prompt = (
                f"Generate a realistic YAML configuration file for a Python web app. "
                f"Include database, cache, logging, and API settings. 30-60 lines."
            )
        elif file_type == ".txt":
            base_prompt = (
                f"Generate a realistic Python requirements.txt with 15-25 common packages "
                f"and pinned versions."
            )

        if context.is_red_herring_file and context.red_herring_hint:
            base_prompt += (
                f"\n\nIMPORTANT: This file should contain content that is thematically "
                f"similar to but distinct from the following: {context.red_herring_hint}. "
                f"Include plausible-looking values that could be confused for the real target."
            )

        return base_prompt
