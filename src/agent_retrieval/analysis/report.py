from __future__ import annotations

from pathlib import Path

import pandas as pd
from jinja2 import Environment, FileSystemLoader


_TEMPLATES_DIR = Path(__file__).parent / "templates"


def render_report(
    batch_name: str,
    df: pd.DataFrame,
    accuracy_by_type_html: str,
    accuracy_by_param_html: str,
    tool_usage_html: str,
    figure_paths: list[str],
    output_path: Path,
    generated_at: str = "",
) -> None:
    """Render the HTML report via Jinja2 and write to output_path."""
    env = Environment(loader=FileSystemLoader(str(_TEMPLATES_DIR)), autoescape=False)
    template = env.get_template("report.html.j2")

    context = {
        "batch_name": batch_name,
        "generated_at": generated_at,
        "total_runs": len(df),
        "total_experiments": df["experiment_id"].nunique() if "experiment_id" in df.columns else 0,
        "mean_score": float(df["weighted_score"].mean()) if "weighted_score" in df.columns else 0.0,
        "std_score": float(df["weighted_score"].std()) if "weighted_score" in df.columns else 0.0,
        "experiment_types": list(df["experiment_type"].unique()) if "experiment_type" in df.columns else [],
        "accuracy_by_type_html": accuracy_by_type_html,
        "accuracy_by_param_html": accuracy_by_param_html,
        "tool_usage_html": tool_usage_html,
        "figure_paths": figure_paths,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(template.render(**context), encoding="utf-8")
