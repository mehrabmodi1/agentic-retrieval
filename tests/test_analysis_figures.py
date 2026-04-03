from pathlib import Path
import pandas as pd
import pytest
from agent_retrieval.analysis.figures import (
    plot_accuracy_vs_corpus_size, plot_context_usage,
    plot_tool_distribution, plot_cross_type_comparison,
)
from agent_retrieval.analysis.figures import (
    plot_accuracy_vs_n_items,
    plot_accuracy_by_discriminability,
    plot_accuracy_by_reference_clarity,
    plot_profile_comparison,
)

@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame([
        {"experiment_type": "needle_in_haystack", "target_token_count": 100_000,
         "weighted_score": 0.9, "total_context_tokens": 50_000, "tool_Grep": 5, "tool_Read": 3},
        {"experiment_type": "needle_in_haystack", "target_token_count": 100_000,
         "weighted_score": 0.85, "total_context_tokens": 55_000, "tool_Grep": 6, "tool_Read": 4},
        {"experiment_type": "needle_in_haystack", "target_token_count": 500_000,
         "weighted_score": 0.7, "total_context_tokens": 120_000, "tool_Grep": 12, "tool_Read": 8},
        {"experiment_type": "chain_of_retrieval", "target_token_count": 100_000,
         "weighted_score": 0.95, "total_context_tokens": 60_000, "tool_Grep": 8, "tool_Read": 5},
    ])

class TestFigures:
    def test_accuracy_vs_corpus_size_creates_file(self, sample_df, tmp_path):
        out = tmp_path / "accuracy_vs_corpus_size.png"
        plot_accuracy_vs_corpus_size(sample_df, out)
        assert out.exists() and out.stat().st_size > 0

    def test_context_usage_creates_file(self, sample_df, tmp_path):
        out = tmp_path / "context_usage.png"
        plot_context_usage(sample_df, out)
        assert out.exists()

    def test_tool_distribution_creates_file(self, sample_df, tmp_path):
        out = tmp_path / "tool_dist.png"
        plot_tool_distribution(sample_df, out)
        assert out.exists()

    def test_cross_type_creates_file(self, sample_df, tmp_path):
        out = tmp_path / "cross_type.png"
        plot_cross_type_comparison(sample_df, out)
        assert out.exists()


class TestV2Figures:
    @pytest.fixture
    def v2_df(self):
        return pd.DataFrame([
            {"experiment_type": "single_needle", "content_profile": "python_repo", "corpus_token_count": 20000,
             "discriminability": "easy", "reference_clarity": "exact", "weighted_score": 0.9},
            {"experiment_type": "single_needle", "content_profile": "python_repo", "corpus_token_count": 20000,
             "discriminability": "hard", "reference_clarity": "exact", "weighted_score": 0.6},
            {"experiment_type": "single_needle", "content_profile": "noir_fiction", "corpus_token_count": 20000,
             "discriminability": "easy", "reference_clarity": "exact", "weighted_score": 0.85},
            {"experiment_type": "multi_chain", "content_profile": "python_repo", "corpus_token_count": 40000,
             "discriminability": "easy", "reference_clarity": "synonym", "weighted_score": 0.7, "n_items": 2},
            {"experiment_type": "multi_chain", "content_profile": "python_repo", "corpus_token_count": 40000,
             "discriminability": "easy", "reference_clarity": "synonym", "weighted_score": 0.5, "n_items": 8},
        ])

    def test_plot_accuracy_vs_n_items(self, v2_df, tmp_path):
        out = tmp_path / "n_items.png"
        plot_accuracy_vs_n_items(v2_df, out)
        assert out.exists()

    def test_plot_accuracy_by_discriminability(self, v2_df, tmp_path):
        out = tmp_path / "disc.png"
        plot_accuracy_by_discriminability(v2_df, out)
        assert out.exists()

    def test_plot_accuracy_by_reference_clarity(self, v2_df, tmp_path):
        out = tmp_path / "ref.png"
        plot_accuracy_by_reference_clarity(v2_df, out)
        assert out.exists()

    def test_plot_profile_comparison(self, v2_df, tmp_path):
        out = tmp_path / "profile.png"
        plot_profile_comparison(v2_df, out)
        assert out.exists()
