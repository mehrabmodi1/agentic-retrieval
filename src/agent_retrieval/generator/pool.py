from __future__ import annotations

from pathlib import Path

from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, query

from agent_retrieval.generator.corpus_files import iter_corpus_files
from agent_retrieval.generator.profiles.registry import get_profile
from agent_retrieval.schema.experiment import POOL_GENERATION_MODEL


def estimate_token_count(directory: Path) -> int:
    total_chars = 0
    if not directory.exists():
        return 0
    for f in iter_corpus_files(directory):
        if f.is_file():
            total_chars += len(f.read_text())
    return total_chars // 4


async def generate_pool(
    profile_name: str,
    pool_dir: Path,
    target_token_count: int = 1_000_000,
) -> None:
    if pool_dir.exists() and estimate_token_count(pool_dir) >= target_token_count:
        return

    pool_dir.mkdir(parents=True, exist_ok=True)
    profile = get_profile(profile_name)
    sections = profile.skeleton(target_token_count)
    brief = profile.pool_generation_brief(target_token_count)

    for section in sections:
        current_tokens = estimate_token_count(pool_dir)
        if current_tokens >= target_token_count:
            break

        files_list = "\n".join(f"- {f}" for f in section["files"])
        section_prompt = (
            f"Generate the following files for the '{section['name']}' section.\n"
            f"Section description: {section['description']}\n"
            f"Target tokens for this section: ~{section.get('target_tokens', 50000):,}\n\n"
            f"Files to create:\n{files_list}\n\n"
            f"Create each file using the Write tool. Make sure each file path is exactly "
            f"as listed above."
        )

        options = ClaudeAgentOptions(
            model=POOL_GENERATION_MODEL,
            system_prompt=brief,
            cwd=str(pool_dir),
            allowed_tools=["Write"],
            permission_mode="acceptEdits",
            max_turns=50,
        )

        async for message in query(prompt=section_prompt, options=options):
            if isinstance(message, ResultMessage):
                break

        print(
            f"  Pool '{profile_name}' section '{section['name']}' done. "
            f"Cumulative tokens: ~{estimate_token_count(pool_dir):,}"
        )
