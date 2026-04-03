from agent_retrieval.analysis.analyze import run_analysis
from agent_retrieval.analysis.figures import (
    plot_accuracy_vs_corpus_size,
    plot_context_usage,
    plot_cross_type_comparison,
    plot_tool_distribution,
    plot_accuracy_vs_n_items,
    plot_accuracy_by_discriminability,
    plot_accuracy_by_reference_clarity,
    plot_profile_comparison,
)
from agent_retrieval.analysis.loader import load_batch_results
from agent_retrieval.analysis.tables import (
    accuracy_by_param,
    accuracy_by_type,
    tool_usage_by_type,
)

__all__ = [
    "run_analysis",
    "load_batch_results",
    "accuracy_by_type",
    "accuracy_by_param",
    "tool_usage_by_type",
    "plot_accuracy_vs_corpus_size",
    "plot_context_usage",
    "plot_tool_distribution",
    "plot_cross_type_comparison",
    "plot_accuracy_vs_n_items",
    "plot_accuracy_by_discriminability",
    "plot_accuracy_by_reference_clarity",
    "plot_profile_comparison",
]
