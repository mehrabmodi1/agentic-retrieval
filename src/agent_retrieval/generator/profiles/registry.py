from __future__ import annotations

from agent_retrieval.generator.profiles.base import ContentProfile
from agent_retrieval.generator.profiles.python_repo import PythonRepoProfile

_PROFILES: dict[str, type[ContentProfile]] = {
    "python_repo": PythonRepoProfile,
}


def get_profile(name: str) -> ContentProfile:
    if name not in _PROFILES:
        raise KeyError(f"Unknown content profile: '{name}'. Available: {list(_PROFILES.keys())}")
    return _PROFILES[name]()
