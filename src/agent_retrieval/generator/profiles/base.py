from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from agent_retrieval.schema.experiment import CorpusSpec


@dataclass
class GenerationContext:
    corpus_spec: CorpusSpec
    red_herring_hint: str | None
    is_red_herring_file: bool


class ContentProfile(ABC):
    @abstractmethod
    def generate_folder_structure(self, spec: CorpusSpec) -> list[Path]:
        """Return list of relative file paths to create."""
        ...

    @abstractmethod
    def generate_file_prompt(self, path: Path, context: GenerationContext) -> str:
        """Return the LLM prompt to generate content for this file."""
        ...
