from pathlib import Path

import pytest

from agent_retrieval.generator.profiles.base import ContentProfile, GenerationContext
from agent_retrieval.generator.profiles.python_repo import PythonRepoProfile
from agent_retrieval.generator.profiles.registry import get_profile
from agent_retrieval.schema.experiment import CorpusSpec


@pytest.fixture
def corpus_spec() -> CorpusSpec:
    return CorpusSpec(
        content_profile="python_repo",
        target_token_count=10_000,
        target_file_count=10,
        folder_depth=2,
        folder_distribution="balanced",
        generation_model="haiku",
        red_herring_density="low",
    )


class TestPythonRepoProfile:
    def test_generates_folder_structure(self, corpus_spec):
        profile = PythonRepoProfile()
        paths = profile.generate_folder_structure(corpus_spec)
        assert len(paths) == corpus_spec.target_file_count
        assert all(isinstance(p, Path) for p in paths)
        assert all(p.suffix == ".py" or p.name in ("README.md", "requirements.txt", "config.yaml") for p in paths)

    def test_folder_depth_respected(self, corpus_spec):
        profile = PythonRepoProfile()
        paths = profile.generate_folder_structure(corpus_spec)
        max_depth = max(len(p.parts) - 1 for p in paths)  # -1 for filename
        assert max_depth <= corpus_spec.folder_depth

    def test_generate_file_prompt_returns_string(self, corpus_spec):
        profile = PythonRepoProfile()
        paths = profile.generate_folder_structure(corpus_spec)
        ctx = GenerationContext(
            corpus_spec=corpus_spec,
            red_herring_hint=None,
            is_red_herring_file=False,
        )
        prompt = profile.generate_file_prompt(paths[0], ctx)
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_red_herring_prompt_differs(self, corpus_spec):
        profile = PythonRepoProfile()
        paths = profile.generate_folder_structure(corpus_spec)
        ctx_normal = GenerationContext(
            corpus_spec=corpus_spec,
            red_herring_hint=None,
            is_red_herring_file=False,
        )
        ctx_herring = GenerationContext(
            corpus_spec=corpus_spec,
            red_herring_hint="Variables with similar names",
            is_red_herring_file=True,
        )
        normal = profile.generate_file_prompt(paths[0], ctx_normal)
        herring = profile.generate_file_prompt(paths[0], ctx_herring)
        assert normal != herring


class TestProfileRegistry:
    def test_get_python_repo(self):
        profile = get_profile("python_repo")
        assert isinstance(profile, PythonRepoProfile)

    def test_unknown_profile_raises(self):
        with pytest.raises(KeyError):
            get_profile("nonexistent_profile")
