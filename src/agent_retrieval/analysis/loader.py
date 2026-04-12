from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from agent_retrieval.schema.answer_key import AnswerKey
from agent_retrieval.schema.experiment import ExperimentSpec
from agent_retrieval.schema.verdict import Verdict


def load_batch_results(
    batch_name: str,
    workspace_dir: Path,
    specs_dir: Path | None = None,
) -> pd.DataFrame:
    judgements_dir = workspace_dir / "judge" / "judgements" / batch_name
    answer_keys_dir = workspace_dir / "judge" / "answer_keys"

    rows: list[dict] = []
    for verdict_path in sorted(judgements_dir.rglob("*.yaml")):
        verdict = Verdict.from_yaml(verdict_path)

        row: dict = {
            "experiment_id": verdict.parametrisation_id,
            "run_id": verdict.run_id,
            "batch_name": verdict.batch_name,
            "weighted_score": verdict.weighted_score,
            "total_context_tokens": verdict.session_metrics.total_context_tokens,
            "total_turns": verdict.session_metrics.total_turns,
            "duration_seconds": verdict.session_metrics.duration_seconds,
        }

        # Try to load metadata from v2 answer key first
        ak_path = answer_keys_dir / f"{verdict.parametrisation_id}.yaml"
        if ak_path.exists():
            ak = AnswerKey.from_yaml(ak_path)
            if ak.parameters:
                row["content_profile"] = ak.parameters.get("content_profile", "")
                row["corpus_token_count"] = ak.parameters.get("corpus_token_count", 0)
                row["discriminability"] = ak.parameters.get("discriminability", "")
                row["reference_clarity"] = ak.parameters.get("reference_clarity", "")
                row["n_items"] = ak.parameters.get("n_items")
                # Extract n_items from parametrisation_id if not in parameters
                # e.g. "multi_chain__noir_fiction__20k__easy__exact__n16" -> 16
                if row["n_items"] is None:
                    parts = ak.parametrisation_id.split("__")
                    for part in parts:
                        if part.startswith("n") and part[1:].isdigit():
                            row["n_items"] = int(part[1:])
                            break
                # Derive experiment_type from parametrisation_id
                if ak.parametrisation_id:
                    row["experiment_type"] = ak.parametrisation_id.split("__")[0]
                else:
                    row["experiment_type"] = ""
            else:
                # V1 fallback: load from spec file
                spec_path = specs_dir / f"{verdict.parametrisation_id}.yaml" if specs_dir else None
                if spec_path and spec_path.exists():
                    spec = ExperimentSpec.from_yaml(spec_path)
                    row["experiment_type"] = spec.experiment_type
                    row["content_profile"] = spec.corpus.content_profile
                    row["target_token_count"] = spec.corpus.target_token_count
                    row["target_file_count"] = spec.corpus.target_file_count
                    row["agent_model"] = spec.runner.agent_model
                else:
                    row["experiment_type"] = ""
        else:
            # V1 fallback: load from spec file
            spec_path = specs_dir / f"{verdict.parametrisation_id}.yaml" if specs_dir else None
            if spec_path and spec_path.exists():
                spec = ExperimentSpec.from_yaml(spec_path)
                row["experiment_type"] = spec.experiment_type
                row["content_profile"] = spec.corpus.content_profile
                row["target_token_count"] = spec.corpus.target_token_count
                row["target_file_count"] = spec.corpus.target_file_count
                row["agent_model"] = spec.runner.agent_model
            else:
                row["experiment_type"] = ""

        # Flatten tool calls
        for tool_name, count in verdict.session_metrics.tool_calls.items():
            row[f"tool_{tool_name}"] = count

        # Flatten per-criterion scores
        for score_entry in verdict.scores:
            row[f"score_{score_entry.criterion}"] = score_entry.score

        rows.append(row)

    return pd.DataFrame(rows)
