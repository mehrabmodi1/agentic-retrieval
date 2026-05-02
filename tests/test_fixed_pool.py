import subprocess
import sys

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

    def test_sample_is_cross_process_deterministic(self, tmp_path):
        """The sample for a given pid must be identical across Python processes."""
        pool_repr = repr([{"inserted_text": f"item_{i}"} for i in range(16)])
        code = (
            f"from agent_retrieval.generator.fixed_pool import sample_fixed_pool;"
            f"pool = {pool_repr};"
            f"s = sample_fixed_pool(pool, n=8, parametrisation_id='test__a__n8');"
            f"print(','.join(it['inserted_text'] for it in s))"
        )
        out1 = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
        out2 = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
        assert out1.stdout == out2.stdout, (
            f"Cross-process sampling diverged:\n  proc1: {out1.stdout}\n  proc2: {out2.stdout}"
        )


class TestSampleFixedPoolStratified:
    def _mixed_pool(self) -> list[dict]:
        pool = []
        for i in range(8):
            pool.append({"text": f"lower_{i}", "direction": "lower"})
        for i in range(8):
            pool.append({"text": f"upper_{i}", "direction": "upper"})
        return pool

    def test_balance_key_none_matches_legacy(self):
        """No balance_key produces same output as calling without the parameter."""
        pool = [{"inserted_text": f"item_{i}"} for i in range(16)]
        pid = "test__legacy__n8"
        s1 = sample_fixed_pool(pool, n=8, parametrisation_id=pid)
        s2 = sample_fixed_pool(pool, n=8, parametrisation_id=pid, balance_key=None)
        assert s1 == s2

    def test_n2_with_balance_key_picks_one_of_each(self):
        """For n=2 with balance_key, every sample must have exactly 1 lower and 1 upper."""
        pool = self._mixed_pool()
        for i in range(50):
            pid = f"test__stratified__n2__pid{i}"
            sampled = sample_fixed_pool(pool, n=2, parametrisation_id=pid, balance_key="direction")
            assert len(sampled) == 2
            directions = [it["direction"] for it in sampled]
            assert directions.count("lower") == 1, f"pid={pid}: expected 1 lower, got {directions}"
            assert directions.count("upper") == 1, f"pid={pid}: expected 1 upper, got {directions}"

    def test_n4_with_balance_key_guarantees_at_least_one_of_each(self):
        """For n=4 with balance_key, every sample has at least 1 lower and 1 upper."""
        pool = self._mixed_pool()
        for i in range(50):
            pid = f"test__stratified__n4__pid{i}"
            sampled = sample_fixed_pool(pool, n=4, parametrisation_id=pid, balance_key="direction")
            assert len(sampled) == 4
            directions = [it["direction"] for it in sampled]
            assert "lower" in directions, f"pid={pid}: no lower in {directions}"
            assert "upper" in directions, f"pid={pid}: no upper in {directions}"

    def test_balance_key_falls_back_when_n_lt_2(self):
        """For n=1 with balance_key, returns 1 item without raising."""
        pool = self._mixed_pool()
        sampled = sample_fixed_pool(pool, n=1, parametrisation_id="test__n1__fallback", balance_key="direction")
        assert len(sampled) == 1

    def test_balance_key_falls_back_when_pool_has_one_group(self):
        """Pool with all items the same value for balance_key falls back to plain random sample."""
        pool = [{"text": f"lower_{i}", "direction": "lower"} for i in range(8)]
        sampled = sample_fixed_pool(pool, n=4, parametrisation_id="test__one_group", balance_key="direction")
        assert len(sampled) == 4
        assert all(it["direction"] == "lower" for it in sampled)

    def test_balance_key_is_cross_process_deterministic(self):
        """Stratified sampling must produce identical output across Python processes."""
        pool = self._mixed_pool()
        pool_repr = repr(pool)
        code = (
            f"from agent_retrieval.generator.fixed_pool import sample_fixed_pool;"
            f"pool = {pool_repr};"
            f"s = sample_fixed_pool(pool, n=4, parametrisation_id='test__strat__xproc', balance_key='direction');"
            f"print(','.join(it['text'] for it in s))"
        )
        out1 = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
        out2 = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
        assert out1.stdout == out2.stdout, (
            f"Stratified cross-process sampling diverged:\n  proc1: {out1.stdout}\n  proc2: {out2.stdout}"
        )
