from __future__ import annotations

import pandas as pd


def accuracy_by_type(df: pd.DataFrame) -> pd.DataFrame:
    """Group by experiment_type; aggregate mean/std/count on weighted_score."""
    result = (
        df.groupby("experiment_type")["weighted_score"]
        .agg(mean="mean", std="std", count="count")
        .reset_index()
    )
    return result


def accuracy_by_param(df: pd.DataFrame, param_column: str) -> pd.DataFrame:
    """Group by experiment_type + param_column; aggregate mean/std/count on weighted_score."""
    result = (
        df.groupby(["experiment_type", param_column])["weighted_score"]
        .agg(mean="mean", std="std", count="count")
        .reset_index()
    )
    return result


def tool_usage_by_type(df: pd.DataFrame) -> pd.DataFrame:
    """Group by experiment_type; aggregate mean/std on tool_ columns.

    Returns a DataFrame with experiment_type as index and tool names as columns.
    Multi-level columns are flattened to ``<tool>_mean`` / ``<tool>_std``.
    """
    tool_cols = [c for c in df.columns if c.startswith("tool_")]
    if not tool_cols:
        return pd.DataFrame()

    result = (
        df.groupby("experiment_type")[tool_cols]
        .agg(["mean", "std"])
    )
    # flatten multi-level column index: (tool_Grep, mean) -> Grep_mean
    result.columns = [
        f"{col.replace('tool_', '')}_{stat}" for col, stat in result.columns
    ]
    result = result.reset_index()

    # also expose flat tool-name columns (mean values) for easy access in tests
    for tc in tool_cols:
        tool_name = tc.replace("tool_", "")
        if f"{tool_name}_mean" in result.columns:
            result[tool_name] = result[f"{tool_name}_mean"]

    return result
