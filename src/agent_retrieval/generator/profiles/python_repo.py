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

    def pool_generation_brief(self, target_token_count: int) -> str:
        return (
            "You are generating background files for a realistic Python web application repository. "
            "Each file should be a Markdown (.md) file containing realistic Python source code, "
            "configuration, documentation, or project files. Files should look like they belong to "
            "a medium-to-large production web application with database access, caching, API endpoints, "
            "authentication, background tasks, and logging.\n\n"
            "Requirements:\n"
            "- Every file must have a .md extension\n"
            "- Code files should contain 50-200 lines of realistic Python code\n"
            "- Include imports, classes, functions, docstrings, and occasional inline comments\n"
            "- Configuration files should use YAML or environment variable patterns\n"
            "- Documentation files should describe setup, usage, and architecture\n"
            "- Include realistic variable names, function signatures, and data structures\n"
            "- Do NOT include any comments about the files being generated or synthetic\n"
            f"- Target total output: approximately {target_token_count:,} tokens across all files\n"
        )

    def skeleton(self, target_token_count: int) -> list[dict]:
        tokens_per_section = target_token_count // 8

        return [
            {
                "name": "core",
                "description": "Core application models, configuration, and entry points",
                "target_tokens": tokens_per_section,
                "files": [
                    "README.md", "config/settings.md", "config/database.md",
                    "config/logging.md", "core/models.md", "core/exceptions.md",
                    "core/constants.md", "core/app.md",
                ],
            },
            {
                "name": "api",
                "description": "API endpoints, serializers, and middleware",
                "target_tokens": tokens_per_section,
                "files": [
                    "api/routes.md", "api/auth.md", "api/middleware.md",
                    "api/serializers.md", "api/validators.md", "api/pagination.md",
                    "api/rate_limiting.md", "api/error_handlers.md",
                ],
            },
            {
                "name": "services",
                "description": "Business logic services and domain operations",
                "target_tokens": tokens_per_section,
                "files": [
                    "services/user_service.md", "services/auth_service.md",
                    "services/notification_service.md", "services/payment_service.md",
                    "services/search_service.md", "services/analytics_service.md",
                    "services/export_service.md", "services/scheduling_service.md",
                ],
            },
            {
                "name": "db",
                "description": "Database access layer, migrations, and connection management",
                "target_tokens": tokens_per_section,
                "files": [
                    "db/connection.md", "db/migrations.md", "db/repositories.md",
                    "db/query_builder.md", "db/pool.md", "db/cache_layer.md",
                    "db/seeds.md", "db/health_check.md",
                ],
            },
            {
                "name": "workers",
                "description": "Background task workers and job processing",
                "target_tokens": tokens_per_section,
                "files": [
                    "workers/task_runner.md", "workers/email_worker.md",
                    "workers/report_generator.md", "workers/cleanup_worker.md",
                    "workers/sync_worker.md", "workers/retry_handler.md",
                ],
            },
            {
                "name": "utils",
                "description": "Utility functions, helpers, and shared components",
                "target_tokens": tokens_per_section,
                "files": [
                    "utils/crypto.md", "utils/date_helpers.md", "utils/file_utils.md",
                    "utils/http_client.md", "utils/logging_utils.md",
                    "utils/string_utils.md", "utils/validators.md",
                ],
            },
            {
                "name": "tests",
                "description": "Test files and test utilities",
                "target_tokens": tokens_per_section,
                "files": [
                    "tests/conftest.md", "tests/test_auth.md", "tests/test_api.md",
                    "tests/test_services.md", "tests/test_db.md",
                    "tests/test_workers.md", "tests/fixtures.md",
                ],
            },
            {
                "name": "deploy",
                "description": "Deployment configuration, CI/CD, and infrastructure",
                "target_tokens": tokens_per_section,
                "files": [
                    "deploy/dockerfile.md", "deploy/docker_compose.md",
                    "deploy/ci_pipeline.md", "deploy/nginx_config.md",
                    "deploy/env_template.md", "deploy/monitoring.md",
                ],
            },
        ]
