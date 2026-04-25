from agent_retrieval.generator.corpus_files import iter_corpus_files


class TestIterCorpusFiles:
    def test_finds_md_and_py(self, tmp_path):
        (tmp_path / "a.md").write_text("md")
        (tmp_path / "b.py").write_text("py")
        (tmp_path / "c.txt").write_text("txt")  # excluded
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "d.md").write_text("md")
        (sub / "e.py").write_text("py")

        names = sorted(p.name for p in iter_corpus_files(tmp_path))
        assert names == ["a.md", "b.py", "d.md", "e.py"]

    def test_empty_directory_yields_nothing(self, tmp_path):
        assert list(iter_corpus_files(tmp_path)) == []

    def test_only_py_pool_yields_files(self, tmp_path):
        # Mirrors the python_repo pool (no .md, only .py).
        (tmp_path / "api" / "routes.py").parent.mkdir(parents=True)
        (tmp_path / "api" / "routes.py").write_text("py")
        (tmp_path / "tests" / "test_x.py").parent.mkdir(parents=True)
        (tmp_path / "tests" / "test_x.py").write_text("py")
        files = list(iter_corpus_files(tmp_path))
        assert len(files) == 2
