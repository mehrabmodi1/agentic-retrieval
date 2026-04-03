from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from agent_retrieval.schema.experiment import ExperimentSpec
from agent_retrieval.schema.verdict import Verdict


def load_batch_results(
    batch_name: str,
    workspace_dir: Path,
    specs_dir: Path,
) -> pd.DataFrame:
    """Load all verdict YAMLs for a batch into a flat DataFrame.

    Enriches each row with spec metadata (experiment_type, target_token_count, etc.).
    Flattens tool_calls as ``tool_<name>`` columns and per-criterion scores as
    ``score_<criterion>`` columns.
    """
    judgements_dir = workspace_dir / "judge" / "judgements" / batch_name

    rows: list[dict] = []
    for verdict_path in sorted(judgements_dir.rglob("*.yaml")):
        verdict = Verdict.from_yaml(verdict_path)

        spec_path = specs_dir / f"{verdict.experiment_id}.yaml"
        spec = ExperimentSpec.from_yaml(spec_path)

        row: dict = {
            "experiment_id": verdict.experiment_id,
            "run_id": verdict.run_id,
            "batch_name": verdict.batch_name,
            "weighted_score": verdict.weighted_score,
            # spec metadata
            "experiment_type": spec.experiment_type,
            "target_token_count": spec.corpus.target_token_count,
            "target_file_count": spec.corpus.target_file_count,
            "content_profile": spec.corpus.content_profile,
            "generation_model": spec.corpus.generation_model,
            "agent_model": spec.runner.agent_model,
            # session metrics
            "total_context_tokens": verdict.session_metrics.total_context_tokens,
            "total_turns": verdict.session_metrics.total_turns,
            "duration_seconds": verdict.session_metrics.duration_seconds,
        }

        # flatten tool calls
        for tool_name, count in verdict.session_metrics.tool_calls.items():
            row[f"tool_{tool_name}"] = count

        # flatten per-criterion scores
        for score_entry in verdict.scores:
            row[f"score_{score_entry.criterion}"] = score_entry.score

        rows.append(row)

    df = pd.DataFrame(rows)
    return df
