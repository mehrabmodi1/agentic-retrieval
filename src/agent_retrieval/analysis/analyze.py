from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from agent_retrieval.analysis.figures import (
    plot_accuracy_vs_corpus_size,
    plot_context_usage,
    plot_cross_type_comparison,
    plot_tool_distribution,
)
from agent_retrieval.analysis.loader import load_batch_results
from agent_retrieval.analysis.report import render_report
from agent_retrieval.analysis.tables import (
    accuracy_by_param,
    accuracy_by_type,
    tool_usage_by_type,
)


def run_analysis(
    batch_name: str,
    workspace_dir: Path,
    specs_dir: Path,
) -> Path:
    """Top-level orchestrator: load data, save CSVs, figures, and HTML report.

    Returns the path to the generated report.html.
    """
    output_dir = workspace_dir / "analysis" / batch_name
    output_dir.mkdir(parents=True, exist_ok=True)

    tables_dir = output_dir / "tables"
    tables_dir.mkdir(exist_ok=True)

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    # --- Load data ---
    df = load_batch_results(batch_name, workspace_dir=workspace_dir, specs_dir=specs_dir)

    # --- Save summary CSV ---
    df.to_csv(output_dir / "summary.csv", index=False)

    # --- Tables ---
    abt = accuracy_by_type(df)
    abt.to_csv(tables_dir / "accuracy_by_type.csv", index=False)

    abp = accuracy_by_param(df, param_column="target_token_count")
    abp.to_csv(tables_dir / "accuracy_by_token_count.csv", index=False)

    tut = tool_usage_by_type(df)
    tut.to_csv(tables_dir / "tool_usage_by_type.csv", index=False)

    # --- Figures ---
    fig_accuracy = figures_dir / "accuracy_vs_corpus_size.png"
    plot_accuracy_vs_corpus_size(df, fig_accuracy)

    fig_context = figures_dir / "context_usage.png"
    plot_context_usage(df, fig_context)

    fig_tools = figures_dir / "tool_distribution.png"
    plot_tool_distribution(df, fig_tools)

    fig_cross = figures_dir / "cross_type_comparison.png"
    plot_cross_type_comparison(df, fig_cross)

    figure_paths = [
        str(fig_accuracy.relative_to(output_dir)),
        str(fig_context.relative_to(output_dir)),
        str(fig_tools.relative_to(output_dir)),
        str(fig_cross.relative_to(output_dir)),
    ]

    # --- Report ---
    generated_at = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    report_path = output_dir / "report.html"
    render_report(
        batch_name=batch_name,
        df=df,
        accuracy_by_type_html=abt.to_html(index=False, classes="table"),
        accuracy_by_param_html=abp.to_html(index=False, classes="table"),
        tool_usage_html=tut.to_html(index=False, classes="table"),
        figure_paths=figure_paths,
        output_path=report_path,
        generated_at=generated_at,
    )

    return report_path
