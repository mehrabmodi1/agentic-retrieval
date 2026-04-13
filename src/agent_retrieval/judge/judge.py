from __future__ import annotations
import json
from collections import defaultdict
from itertools import zip_longest
from pathlib import Path

from agent_retrieval.judge.metrics import extract_session_metrics
from agent_retrieval.judge.scoring import JUDGE_MODEL, score_response
from agent_retrieval.schema.answer_key import AnswerKey
from agent_retrieval.schema.verdict import Verdict


async def judge_run(
    run_dir: Path, answer_key: AnswerKey,
    batch_run_name: str, verdict_path: Path,
) -> Verdict:
    response_path = run_dir / "response.json"
    session_path = run_dir / "session.jsonl"
    response_data = json.loads(response_path.read_text())
    agent_response = response_data["response_text"]
    scores = await score_response(agent_response, answer_key)
    total_weight = sum(s.weight for s in scores)
    weighted_score = sum(s.score * s.weight for s in scores) / total_weight if total_weight > 0 else 0.0
    metrics = extract_session_metrics(session_path, response_path)
    run_id = run_dir.name
    verdict = Verdict(
        parametrisation_id=answer_key.parametrisation_id, run_id=run_id,
        batch_name=batch_run_name, judge_model=JUDGE_MODEL, scores=scores,
        weighted_score=round(weighted_score, 4), session_metrics=metrics,
    )
    verdict.to_yaml(verdict_path)
    return verdict


async def judge_batch(
    batch_run_name: str,
    workspace_dir: Path,
    rejudge: bool = False,
) -> list[Verdict]:
    runs_dir = workspace_dir / "runner" / "runs" / batch_run_name
    answer_keys_dir = workspace_dir / "judge" / "answer_keys"
    judgements_dir = workspace_dir / "judge" / "judgements"

    if not runs_dir.exists():
        print(f"No runs found at {runs_dir}")
        return []

    # Group parametrisation dirs by experiment type, then round-robin
    # so coverage spreads evenly across types as judging progresses.
    by_type: dict[str, list[Path]] = defaultdict(list)
    for pid_dir in sorted(runs_dir.iterdir()):
        if not pid_dir.is_dir():
            continue
        exp_type = pid_dir.name.split("__")[0]
        by_type[exp_type].append(pid_dir)

    interleaved: list[Path] = []
    for group in zip_longest(*by_type.values()):
        for pid_dir in group:
            if pid_dir is not None:
                interleaved.append(pid_dir)

    verdicts: list[Verdict] = []
    for pid_dir in interleaved:
        pid = pid_dir.name
        ak_path = answer_keys_dir / f"{pid}.yaml"
        if not ak_path.exists():
            print(f"SKIP {pid}: no answer key")
            continue
        answer_key = AnswerKey.from_yaml(ak_path)

        for run_dir in sorted(pid_dir.iterdir()):
            if not (run_dir / "response.json").exists():
                continue
            verdict_path = judgements_dir / batch_run_name / pid / f"{run_dir.name}.yaml"
            if verdict_path.exists() and not rejudge:
                verdicts.append(Verdict.from_yaml(verdict_path))
                continue
            try:
                verdict = await judge_run(
                    run_dir=run_dir, answer_key=answer_key,
                    batch_run_name=batch_run_name, verdict_path=verdict_path,
                )
                verdicts.append(verdict)
                print(f"Judged {pid}/{run_dir.name}: weighted_score={verdict.weighted_score}")
            except Exception as e:
                print(f"FAILED {pid}/{run_dir.name}: {e}")

    return verdicts
