import pytest

from agent_retrieval.generator.fixed_pool import sample_fixed_pool


class TestSampleFixedPool:
    def test_samples_n_items(self):
        pool = [{"inserted_text": f"item_{i}"} for i in range(16)]
        sampled = sample_fixed_pool(pool, n=4, parametrisation_id="test__a__n4")
        assert len(sampled) == 4

    def test_deterministic_for_same_parametrisation_id(self):
        pool = [{"inserted_text": f"item_{i}"} for i in range(16)]
        s1 = sample_fixed_pool(pool, n=8, parametrisation_id="test__a__n8")
        s2 = sample_fixed_pool(pool, n=8, parametrisation_id="test__a__n8")
        assert s1 == s2

    def test_different_pid_gives_different_sample(self):
        pool = [{"inserted_text": f"item_{i}"} for i in range(16)]
        s1 = sample_fixed_pool(pool, n=8, parametrisation_id="test__a__n8")
        s2 = sample_fixed_pool(pool, n=8, parametrisation_id="test__b__n8")
        # Different IDs should very likely give different orderings.
        assert s1 != s2

    def test_n_equal_to_pool_returns_full_pool(self):
        pool = [{"inserted_text": f"item_{i}"} for i in range(16)]
        sampled = sample_fixed_pool(pool, n=16, parametrisation_id="test__a__n16")
        # Same items, possibly reordered.
        assert sorted(s["inserted_text"] for s in sampled) == sorted(p["inserted_text"] for p in pool)

    def test_n_greater_than_pool_raises(self):
        pool = [{"inserted_text": f"item_{i}"} for i in range(16)]
        with pytest.raises(ValueError, match="n=20 exceeds pool size 16"):
            sample_fixed_pool(pool, n=20, parametrisation_id="test__a__n20")
