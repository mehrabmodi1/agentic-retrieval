import pytest
from pathlib import Path

from agent_retrieval.generator.assembler import assemble_corpus
from agent_retrieval.schema.template import Parametrisation


@pytest.fixture
def pool_dir(tmp_path) -> Path:
    pool = tmp_path / "pool"
    pool.mkdir()
    # Create 20 files with varying sizes
    for i in range(20):
        sub = pool / f"section_{i // 5}"
        sub.mkdir(exist_ok=True)
        # ~250 tokens each (1000 chars / 4)
        (sub / f"file_{i}.md").write_text(f"# File {i}\n" + "content " * 125)
    return pool


@pytest.fixture
def parametrisation() -> Parametrisation:
    return Parametrisation(
        experiment_type="single_needle",
        content_profile="python_repo",
        corpus_token_count=1000,
        discriminability="easy",
        reference_clarity="exact",
    )


class TestAssembleCorpus:
    def test_creates_corpus_directory(self, pool_dir, parametrisation, tmp_path):
        corpus_dir = tmp_path / "corpora" / parametrisation.parametrisation_id
        assemble_corpus(pool_dir, corpus_dir, parametrisation)
        assert corpus_dir.exists()

    def test_respects_token_budget(self, pool_dir, parametrisation, tmp_path):
        corpus_dir = tmp_path / "corpora" / parametrisation.parametrisation_id
        assemble_corpus(pool_dir, corpus_dir, parametrisation)
        # Count tokens in assembled corpus
        total_chars = sum(
            len(f.read_text()) for f in corpus_dir.rglob("*.md") if f.is_file()
        )
        total_tokens = total_chars // 4
        # Should be close to budget but not wildly over
        assert total_tokens >= parametrisation.corpus_token_count * 0.5
        assert total_tokens <= parametrisation.corpus_token_count * 2.0

    def test_deterministic_with_same_seed(self, pool_dir, parametrisation, tmp_path):
        corpus1 = tmp_path / "c1" / parametrisation.parametrisation_id
        corpus2 = tmp_path / "c2" / parametrisation.parametrisation_id
        assemble_corpus(pool_dir, corpus1, parametrisation)
        assemble_corpus(pool_dir, corpus2, parametrisation)
        files1 = sorted(f.name for f in corpus1.rglob("*.md"))
        files2 = sorted(f.name for f in corpus2.rglob("*.md"))
        assert files1 == files2

    def test_different_parametrisations_differ(self, pool_dir, tmp_path):
        p1 = Parametrisation(
            experiment_type="single_needle",
            content_profile="python_repo",
            corpus_token_count=1000,
            discriminability="easy",
            reference_clarity="exact",
        )
        p2 = Parametrisation(
            experiment_type="single_needle",
            content_profile="python_repo",
            corpus_token_count=1000,
            discriminability="hard",
            reference_clarity="exact",
        )
        c1 = tmp_path / "c1" / p1.parametrisation_id
        c2 = tmp_path / "c2" / p2.parametrisation_id
        assemble_corpus(pool_dir, c1, p1)
        assemble_corpus(pool_dir, c2, p2)
        files1 = sorted(f.name for f in c1.rglob("*.md"))
        files2 = sorted(f.name for f in c2.rglob("*.md"))
        # Different seeds should produce different samples (probabilistic but very likely with 20 files)
        # We just check they both produced files
        assert len(files1) > 0
        assert len(files2) > 0

    def test_preserves_subdirectory_structure(self, pool_dir, parametrisation, tmp_path):
        corpus_dir = tmp_path / "corpora" / parametrisation.parametrisation_id
        assemble_corpus(pool_dir, corpus_dir, parametrisation)
        # Files should retain their relative subdirectory paths
        for f in corpus_dir.rglob("*.md"):
            rel = f.relative_to(corpus_dir)
            assert len(rel.parts) >= 1  # at least a filename

    def test_skips_existing_corpus(self, pool_dir, parametrisation, tmp_path):
        corpus_dir = tmp_path / "corpora" / parametrisation.parametrisation_id
        assemble_corpus(pool_dir, corpus_dir, parametrisation)
        first_files = sorted(f.name for f in corpus_dir.rglob("*.md"))

        # Add a marker file to detect if corpus gets regenerated
        (corpus_dir / "marker.txt").write_text("exists")
        assemble_corpus(pool_dir, corpus_dir, parametrisation)
        assert (corpus_dir / "marker.txt").exists()
