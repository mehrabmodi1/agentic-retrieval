from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def _save_and_close(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=100)
    plt.close(fig)


def plot_accuracy_vs_corpus_size(df: pd.DataFrame, output_path: Path) -> None:
    """Errorbar plot of weighted_score vs target_token_count, grouped by experiment_type."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for exp_type, group in df.groupby("experiment_type"):
        stats = (
            group.groupby("target_token_count")["weighted_score"]
            .agg(mean="mean", std="std")
            .reset_index()
        )
        ax.errorbar(
            stats["target_token_count"],
            stats["mean"],
            yerr=stats["std"].fillna(0),
            label=exp_type,
            marker="o",
            capsize=4,
        )

    ax.set_xlabel("Corpus Size (tokens)")
    ax.set_ylabel("Weighted Score")
    ax.set_title("Accuracy vs Corpus Size")
    ax.legend()
    _save_and_close(fig, output_path)


def plot_context_usage(df: pd.DataFrame, output_path: Path) -> None:
    """Errorbar plot of total_context_tokens vs target_token_count, grouped by experiment_type."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for exp_type, group in df.groupby("experiment_type"):
        stats = (
            group.groupby("target_token_count")["total_context_tokens"]
            .agg(mean="mean", std="std")
            .reset_index()
        )
        ax.errorbar(
            stats["target_token_count"],
            stats["mean"],
            yerr=stats["std"].fillna(0),
            label=exp_type,
            marker="s",
            capsize=4,
        )

    ax.set_xlabel("Corpus Size (tokens)")
    ax.set_ylabel("Total Context Tokens")
    ax.set_title("Context Usage vs Corpus Size")
    ax.legend()
    _save_and_close(fig, output_path)


def plot_tool_distribution(df: pd.DataFrame, output_path: Path) -> None:
    """Stacked bar chart of mean tool usage per experiment_type."""
    tool_cols = [c for c in df.columns if c.startswith("tool_")]

    fig, ax = plt.subplots(figsize=(8, 5))

    if not tool_cols:
        ax.set_title("Tool Distribution (no data)")
        _save_and_close(fig, output_path)
        return

    agg = df.groupby("experiment_type")[tool_cols].mean()
    agg.columns = [c.replace("tool_", "") for c in agg.columns]
    agg.plot(kind="bar", stacked=True, ax=ax)

    ax.set_xlabel("Experiment Type")
    ax.set_ylabel("Mean Tool Calls")
    ax.set_title("Tool Distribution by Experiment Type")
    ax.legend(title="Tool", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.xticks(rotation=30, ha="right")
    _save_and_close(fig, output_path)


def plot_cross_type_comparison(df: pd.DataFrame, output_path: Path) -> None:
    """Boxplot of weighted_score distributions per experiment_type."""
    fig, ax = plt.subplots(figsize=(8, 5))

    groups = [grp["weighted_score"].values for _, grp in df.groupby("experiment_type")]
    labels = list(df["experiment_type"].unique())

    ax.boxplot(groups, tick_labels=labels, patch_artist=True)
    ax.set_xlabel("Experiment Type")
    ax.set_ylabel("Weighted Score")
    ax.set_title("Score Distribution by Experiment Type")
    plt.xticks(rotation=30, ha="right")
    _save_and_close(fig, output_path)


def plot_accuracy_vs_n_items(df: pd.DataFrame, output_path: Path) -> None:
    """Errorbar plot of weighted_score vs n_items, grouped by experiment_type."""
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_df = df.dropna(subset=["n_items"])

    if plot_df.empty:
        ax.set_title("Accuracy vs N-Items (no data)")
        _save_and_close(fig, output_path)
        return

    for exp_type, group in plot_df.groupby("experiment_type"):
        stats = (
            group.groupby("n_items")["weighted_score"]
            .agg(mean="mean", std="std")
            .reset_index()
        )
        ax.errorbar(
            stats["n_items"], stats["mean"], yerr=stats["std"].fillna(0),
            label=exp_type, marker="o", capsize=4,
        )

    ax.set_xlabel("Number of Items")
    ax.set_ylabel("Weighted Score")
    ax.set_title("Accuracy vs Number of Items")
    ax.legend()
    _save_and_close(fig, output_path)


def plot_accuracy_by_discriminability(df: pd.DataFrame, output_path: Path) -> None:
    """Grouped bar chart of weighted_score by discriminability, per experiment_type."""
    fig, ax = plt.subplots(figsize=(8, 5))

    if "discriminability" not in df.columns:
        ax.set_title("Accuracy by Discriminability (no data)")
        _save_and_close(fig, output_path)
        return

    stats = (
        df.groupby(["experiment_type", "discriminability"])["weighted_score"]
        .agg(mean="mean", std="std")
        .reset_index()
    )
    pivot = stats.pivot(index="experiment_type", columns="discriminability", values="mean")
    pivot.plot(kind="bar", ax=ax, capsize=4)

    ax.set_xlabel("Experiment Type")
    ax.set_ylabel("Weighted Score")
    ax.set_title("Accuracy by Discriminability")
    ax.legend(title="Discriminability")
    plt.xticks(rotation=30, ha="right")
    _save_and_close(fig, output_path)


def plot_accuracy_by_reference_clarity(df: pd.DataFrame, output_path: Path) -> None:
    """Grouped bar chart of weighted_score by reference_clarity, per experiment_type."""
    fig, ax = plt.subplots(figsize=(8, 5))

    if "reference_clarity" not in df.columns:
        ax.set_title("Accuracy by Reference Clarity (no data)")
        _save_and_close(fig, output_path)
        return

    stats = (
        df.groupby(["experiment_type", "reference_clarity"])["weighted_score"]
        .agg(mean="mean", std="std")
        .reset_index()
    )
    pivot = stats.pivot(index="experiment_type", columns="reference_clarity", values="mean")
    pivot.plot(kind="bar", ax=ax, capsize=4)

    ax.set_xlabel("Experiment Type")
    ax.set_ylabel("Weighted Score")
    ax.set_title("Accuracy by Reference Clarity")
    ax.legend(title="Reference Clarity")
    plt.xticks(rotation=30, ha="right")
    _save_and_close(fig, output_path)


def plot_profile_comparison(df: pd.DataFrame, output_path: Path) -> None:
    """Grouped bar chart comparing content profiles across experiment types."""
    fig, ax = plt.subplots(figsize=(8, 5))

    if "content_profile" not in df.columns:
        ax.set_title("Profile Comparison (no data)")
        _save_and_close(fig, output_path)
        return

    stats = (
        df.groupby(["experiment_type", "content_profile"])["weighted_score"]
        .agg(mean="mean", std="std")
        .reset_index()
    )
    pivot = stats.pivot(index="experiment_type", columns="content_profile", values="mean")
    pivot.plot(kind="bar", ax=ax, capsize=4)

    ax.set_xlabel("Experiment Type")
    ax.set_ylabel("Weighted Score")
    ax.set_title("Profile Comparison: Python Repo vs Noir Fiction")
    ax.legend(title="Content Profile")
    plt.xticks(rotation=30, ha="right")
    _save_and_close(fig, output_path)
