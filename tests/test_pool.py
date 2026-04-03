import pytest
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

from agent_retrieval.generator.pool import generate_pool, estimate_token_count


class TestEstimateTokenCount:
    def test_empty_dir(self, tmp_path):
        assert estimate_token_count(tmp_path) == 0

    def test_counts_characters_div_4(self, tmp_path):
        f = tmp_path / "test.md"
        f.write_text("a" * 400)  # 400 chars = ~100 tokens
        assert estimate_token_count(tmp_path) == 100

    def test_counts_multiple_files(self, tmp_path):
        (tmp_path / "a.md").write_text("x" * 800)
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "b.md").write_text("y" * 400)
        assert estimate_token_count(tmp_path) == 300  # (800 + 400) / 4


class TestGeneratePool:
    @pytest.mark.asyncio
    async def test_creates_pool_directory(self, tmp_path):
        pool_dir = tmp_path / "pools" / "python_repo"

        mock_query = AsyncMock()
        # Simulate the agent creating files as a side effect
        async def fake_query(prompt, options):
            # Agent "creates" files during execution
            pool_dir.mkdir(parents=True, exist_ok=True)
            for i in range(5):
                f = pool_dir / f"file_{i}.md"
                f.write_text("x" * 4000)  # 1000 tokens each
            # Yield a ResultMessage-like object
            result = MagicMock()
            result.session_id = "test-session"
            yield result

        with patch("agent_retrieval.generator.pool.query", side_effect=fake_query):
            await generate_pool("python_repo", pool_dir, target_token_count=5000)

        assert pool_dir.exists()

    @pytest.mark.asyncio
    async def test_skips_existing_pool(self, tmp_path):
        pool_dir = tmp_path / "pools" / "python_repo"
        pool_dir.mkdir(parents=True)
        # Create enough files to meet budget
        for i in range(10):
            (pool_dir / f"file_{i}.md").write_text("x" * 4000)

        with patch("agent_retrieval.generator.pool.query") as mock_query:
            await generate_pool("python_repo", pool_dir, target_token_count=5000)
            mock_query.assert_not_called()
