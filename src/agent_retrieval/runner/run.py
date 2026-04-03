from __future__ import annotations
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

from agent_retrieval.runner.session import AgentResult, get_claude_version, run_agent_session
from agent_retrieval.runner.state import RunStateManager
from agent_retrieval.schema.batch import BatchConfig
from agent_retrieval.schema.experiment import ExperimentSpec


async def run_batch(batch: BatchConfig, specs_dir: Path, workspace_dir: Path) -> None:
    runs_dir = workspace_dir / "runner" / "runs"
    corpora_dir = workspace_dir / "runner" / "corpora"
    state_mgr = RunStateManager(runs_dir)

    claude_version = get_claude_version()
    print(f"Claude Code version: {claude_version}")

    recovered = state_mgr.recover_interrupted(batch.batch_name)
    if recovered:
        print(f"Recovered {len(recovered)} interrupted runs")

    for run_config in batch.runs:
        exp_id = run_config.experiment_id
        existing = state_mgr.get_runs_by_status(batch.batch_name, exp_id, "pending")
        existing += state_mgr.get_runs_by_status(batch.batch_name, exp_id, "completed")
        n_needed = run_config.n_repeats - len(existing)
        if n_needed > 0:
            state_mgr.create_pending_runs(batch, exp_id, n_needed, claude_version)

    all_pending: list[tuple[str, str, Path]] = []
    for run_config in batch.runs:
        pending = state_mgr.get_runs_by_status(batch.batch_name, run_config.experiment_id, "pending")
        for run_id, run_dir in pending:
            all_pending.append((run_config.experiment_id, run_id, run_dir))

    if batch.retry_failed:
        for run_config in batch.runs:
            failed = state_mgr.get_runs_by_status(batch.batch_name, run_config.experiment_id, "failed")
            for run_id, run_dir in failed:
                state_mgr.update_status(run_dir, "pending")
                all_pending.append((run_config.experiment_id, run_id, run_dir))

    total = len(all_pending)
    print(f"Running {total} experiment sessions (max_parallel={batch.max_parallel})")

    semaphore = asyncio.Semaphore(batch.max_parallel)
    completed = 0

    async def run_one(exp_id: str, run_id: str, run_dir: Path) -> None:
        nonlocal completed
        async with semaphore:
            spec = ExperimentSpec.from_yaml(specs_dir / f"{exp_id}.yaml")
            corpus_dir = corpora_dir / exp_id
            run_config = next(r for r in batch.runs if r.experiment_id == exp_id)
            agent_model = run_config.agent_model or spec.runner.agent_model

            state_mgr.update_status(run_dir, "running", started_at=datetime.now(timezone.utc).isoformat())

            try:
                result = await run_agent_session(
                    question=spec.question, corpus_dir=corpus_dir, model=agent_model,
                    allowed_tools=spec.runner.allowed_tools, max_tokens=spec.runner.max_tokens,
                    run_id=run_id, run_dir=run_dir,
                )
                response_path = run_dir / "response.json"
                response_path.write_text(json.dumps({
                    "response_text": result.response_text,
                    "session_id": result.session_id,
                    "num_turns": result.num_turns,
                    "total_cost_usd": result.total_cost_usd,
                    "usage": result.usage,
                }, indent=2))
                state_mgr.update_status(run_dir, "completed", completed_at=datetime.now(timezone.utc).isoformat())
                completed += 1
                print(f"[{completed}/{total}] Completed {exp_id} run {run_id}")
            except Exception as e:
                state_mgr.update_status(run_dir, "failed", error_message=str(e))
                completed += 1
                print(f"[{completed}/{total}] FAILED {exp_id} run {run_id}: {e}")

    tasks = [run_one(exp_id, run_id, run_dir) for exp_id, run_id, run_dir in all_pending]
    await asyncio.gather(*tasks)
    print(f"Batch '{batch.batch_name}' complete.")
