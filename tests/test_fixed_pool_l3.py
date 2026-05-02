from agent_retrieval.generator.fixed_pool import sample_fixed_pool_l3


def _make_pool():
    """16-item pool: 4 each of (live, lower), (live, upper), (dead, lower), (dead, upper)."""
    pool = []
    for live in (True, False):
        for direction in ("lower", "upper"):
            for i in range(4):
                pool.append({
                    "text": f"{'L' if live else 'D'}_{direction}_{i}",
                    "bound_direction": direction,
                    "live": live,
                    "bound_value": str(i * 10 + (0 if direction == "lower" else 1000)),
                })
    return pool


class TestSampleFixedPoolL3:
    def test_n2_returns_one_live_lower_and_one_live_upper(self):
        pool = _make_pool()
        sample = sample_fixed_pool_l3(pool, n=2, parametrisation_id="pid_a")
        assert len(sample) == 2
        live = [it for it in sample if it["live"]]
        assert len(live) == 2
        directions = sorted(it["bound_direction"] for it in live)
        assert directions == ["lower", "upper"]

    def test_n3_returns_balanced_live_pair_plus_one_more_live(self):
        pool = _make_pool()
        sample = sample_fixed_pool_l3(pool, n=3, parametrisation_id="pid_b")
        assert len(sample) == 3
        live = [it for it in sample if it["live"]]
        assert len(live) == 3
        directions = [it["bound_direction"] for it in live]
        assert "lower" in directions and "upper" in directions

    def test_n4_covers_all_four_quadrants(self):
        pool = _make_pool()
        sample = sample_fixed_pool_l3(pool, n=4, parametrisation_id="pid_c")
        assert len(sample) == 4
        quadrants = {(it["live"], it["bound_direction"]) for it in sample}
        assert quadrants == {(True, "lower"), (True, "upper"), (False, "lower"), (False, "upper")}

    def test_n8_has_balanced_first4_and_4_random(self):
        pool = _make_pool()
        sample = sample_fixed_pool_l3(pool, n=8, parametrisation_id="pid_d")
        assert len(sample) == 8
        # All 4 quadrants must appear at least once across the 8 items.
        quadrants = {(it["live"], it["bound_direction"]) for it in sample}
        assert quadrants == {(True, "lower"), (True, "upper"), (False, "lower"), (False, "upper")}

    def test_deterministic_across_calls(self):
        pool = _make_pool()
        a = sample_fixed_pool_l3(pool, n=4, parametrisation_id="same_pid")
        b = sample_fixed_pool_l3(pool, n=4, parametrisation_id="same_pid")
        assert [it["text"] for it in a] == [it["text"] for it in b]

    def test_different_pids_produce_different_samples(self):
        pool = _make_pool()
        a = sample_fixed_pool_l3(pool, n=8, parametrisation_id="pid_x")
        b = sample_fixed_pool_l3(pool, n=8, parametrisation_id="pid_y")
        # Highly unlikely to be identical; if they are something is wrong.
        assert [it["text"] for it in a] != [it["text"] for it in b]

    def test_n_too_large_raises(self):
        pool = _make_pool()
        import pytest
        with pytest.raises(ValueError):
            sample_fixed_pool_l3(pool, n=20, parametrisation_id="pid")

    def test_n_below_two_raises(self):
        pool = _make_pool()
        import pytest
        with pytest.raises(ValueError):
            sample_fixed_pool_l3(pool, n=1, parametrisation_id="pid")
