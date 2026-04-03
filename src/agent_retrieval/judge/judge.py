from __future__ import annotations
import json
from pathlib import Path
from agent_retrieval.judge.metrics import extract_session_metrics
from agent_retrieval.judge.scoring import score_response
from agent_retrieval.schema.answer_key import AnswerKey
from agent_retrieval.schema.batch import BatchConfig
from agent_retrieval.schema.verdict import Verdict

async def judge_run(run_dir: Path, answer_key: AnswerKey, judge_model: str,
                    batch_name: str, verdict_path: Path) -> Verdict:
    response_path = run_dir / "response.json"
    session_path = run_dir / "session.jsonl"
    response_data = json.loads(response_path.read_text())
    agent_response = response_data["response_text"]
    scores = await score_response(agent_response, answer_key, judge_model)
    total_weight = sum(s.weight for s in scores)
    weighted_score = sum(s.score * s.weight for s in scores) / total_weight if total_weight > 0 else 0.0
    metrics = extract_session_metrics(session_path)
    run_id = run_dir.name
    verdict = Verdict(experiment_id=answer_key.experiment_id, run_id=run_id, batch_name=batch_name,
                      scores=scores, weighted_score=round(weighted_score, 4), session_metrics=metrics)
    verdict.to_yaml(verdict_path)
    return verdict

async def judge_batch(batch: BatchConfig, workspace_dir: Path, rejudge: bool = False) -> list[Verdict]:
    runs_dir = workspace_dir / "runner" / "runs"
    answer_keys_dir = workspace_dir / "judge" / "answer_keys"
    judgements_dir = workspace_dir / "judge" / "judgements"
    verdicts: list[Verdict] = []
    for run_config in batch.runs:
        exp_id = run_config.experiment_id
        judge_model = run_config.judge_model or batch.judge_model
        answer_key = AnswerKey.from_yaml(answer_keys_dir / f"{exp_id}.yaml")
        exp_runs_dir = runs_dir / batch.batch_name / exp_id
        if not exp_runs_dir.exists():
            continue
        for run_dir in sorted(exp_runs_dir.iterdir()):
            if not (run_dir / "response.json").exists():
                continue
            verdict_path = judgements_dir / batch.batch_name / exp_id / f"{run_dir.name}.yaml"
            if verdict_path.exists() and not rejudge:
                verdicts.append(Verdict.from_yaml(verdict_path))
                continue
            verdict = await judge_run(run_dir=run_dir, answer_key=answer_key, judge_model=judge_model,
                                       batch_name=batch.batch_name, verdict_path=verdict_path)
            verdicts.append(verdict)
            print(f"Judged {exp_id}/{run_dir.name}: weighted_score={verdict.weighted_score}")
    return verdicts
