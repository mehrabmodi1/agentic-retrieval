from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

CORPUS_FILE_GLOBS = ("*.md", "*.py")


def iter_corpus_files(directory: Path) -> Iterator[Path]:
    """Yield every corpus file under directory (.md and .py)."""
    for pattern in CORPUS_FILE_GLOBS:
        yield from directory.rglob(pattern)
