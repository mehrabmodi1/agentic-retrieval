from pathlib import Path

import pytest

from agent_retrieval.generator.profiles.noir_fiction import NoirFictionProfile
from agent_retrieval.generator.profiles.registry import get_profile


class TestNoirFictionProfile:
    def test_pool_generation_brief_returns_string(self):
        profile = NoirFictionProfile()
        brief = profile.pool_generation_brief(target_token_count=100000)
        assert isinstance(brief, str)
        assert len(brief) > 100
        assert "noir" in brief.lower() or "detective" in brief.lower()

    def test_skeleton_returns_sections(self):
        profile = NoirFictionProfile()
        sections = profile.skeleton(target_token_count=100000)
        assert isinstance(sections, list)
        assert len(sections) > 0
        for section in sections:
            assert "name" in section
            assert "description" in section
            assert "files" in section

    def test_skeleton_file_paths_are_md(self):
        profile = NoirFictionProfile()
        sections = profile.skeleton(target_token_count=100000)
        for section in sections:
            for f in section["files"]:
                assert f.endswith(".md"), f"Expected .md file, got {f}"

    def test_registered_in_registry(self):
        profile = get_profile("noir_fiction")
        assert isinstance(profile, NoirFictionProfile)
