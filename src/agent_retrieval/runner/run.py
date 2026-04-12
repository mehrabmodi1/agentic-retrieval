from __future__ import annotations
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

from agent_retrieval.generator.grid import expand_grid, filter_parametrisations
from agent_retrieval.runner.session import get_claude_version, run_agent_session
from agent_retrieval.runner.state import RunStateManager
from agent_retrieval.schema.batch import BatchConfig
from agent_retrieval.schema.template import ExperimentTemplate


async def run_batch(
    batch: BatchConfig,
    experiments_dir: Path,
    workspace_dir: Path,
    resume: str | None = None,
) -> None:
    if resume:
        batch_run_name = resume
    else:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        batch_run_name = f"{batch.batch_name}__{ts}"

    runs_dir = workspace_dir / "runner" / "runs"
    corpora_dir = workspace_dir / "runner" / "corpora"
    answer_keys_dir = workspace_dir / "judge" / "answer_keys"
    state_mgr = RunStateManager(runs_dir)

    claude_version = get_claude_version()
    print(f"Claude Code version: {claude_version}")
    print(f"Batch run: {batch_run_name}")

    recovered = state_mgr.recover_interrupted(batch_run_name)
    if recovered:
        print(f"Recovered {len(recovered)} interrupted runs")

    # Resolve parametrisation IDs from experiment templates + filters
    pid_to_template: dict[str, ExperimentTemplate] = {}
    for entry in batch.experiments:
        template_path = experiments_dir / f"{entry.experiment_type}.yaml"
        template = ExperimentTemplate.from_yaml(template_path)
        params = expand_grid(template)
        if entry.filter:
            params = filter_parametrisations(params, entry.filter)
        for p in params:
            pid_to_template[p.parametrisation_id] = template

    # Reset failed runs to pending for retry (before counting)
    if batch.retry_failed:
        for pid in pid_to_template:
            failed = state_mgr.get_runs_by_status(batch_run_name, pid, "failed")
            for run_id, run_dir in failed:
                state_mgr.update_status(run_dir, "pending")

    # Create pending runs — count ALL existing runs to avoid duplicates
    for pid, template in pid_to_template.items():
        n_repeats = template.runner.n_repeats
        existing = state_mgr.get_runs_by_status(batch_run_name, pid, "pending")
        existing += state_mgr.get_runs_by_status(batch_run_name, pid, "completed")
        existing += state_mgr.get_runs_by_status(batch_run_name, pid, "running")
        n_needed = n_repeats - len(existing)
        if n_needed > 0:
            state_mgr.create_pending_runs(
                batch_run_name, pid, n_needed, claude_version,
                agent_model=template.runner.agent_model,
                effort_mode=template.runner.effort_mode,
            )

    # Collect pending runs (capped at n_repeats - completed per pid)
    all_pending: list[tuple[str, str, Path]] = []
    for pid in pid_to_template:
        n_repeats = pid_to_template[pid].runner.n_repeats
        n_completed = len(state_mgr.get_runs_by_status(batch_run_name, pid, "completed"))
        n_to_run = max(0, n_repeats - n_completed)
        if n_to_run == 0:
            continue
        pending = state_mgr.get_runs_by_status(batch_run_name, pid, "pending")
        for run_id, run_dir in pending[:n_to_run]:
            all_pending.append((pid, run_id, run_dir))

    total = len(all_pending)
    print(f"Running {total} experiment sessions (max_parallel={batch.max_parallel})")

    if not all_pending:
        print("Nothing to run.")
        return

    semaphore = asyncio.Semaphore(batch.max_parallel)
    completed = 0

    async def run_one(pid: str, run_id: str, run_dir: Path) -> None:
        nonlocal completed
        async with semaphore:
            template = pid_to_template[pid]
            corpus_dir = corpora_dir / pid
            ak_path = answer_keys_dir / f"{pid}.yaml"

            try:
                with open(ak_path) as f:
                    ak = yaml.safe_load(f)
                question = ak["expected_answers"]["question"]
            except Exception as e:
                state_mgr.update_status(run_dir, "failed", error_message=f"bad answer key: {e}")
                completed += 1
                print(f"[{completed}/{total}] FAILED {pid} run {run_id}: bad answer key")
                return

            agent_model = template.runner.agent_model
            effort_mode = template.runner.effort_mode
            allowed_tools = template.runner.allowed_tools
            max_tokens = template.runner.max_tokens

            state_mgr.update_status(run_dir, "running", started_at=datetime.now(timezone.utc).isoformat())

            try:
                result = await run_agent_session(
                    question=question, corpus_dir=corpus_dir, model=agent_model,
                    allowed_tools=allowed_tools, max_tokens=max_tokens,
                    run_id=run_id, run_dir=run_dir, effort_mode=effort_mode,
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
                print(f"[{completed}/{total}] Completed {pid} run {run_id}")
            except Exception as e:
                state_mgr.update_status(run_dir, "failed", error_message=str(e))
                completed += 1
                print(f"[{completed}/{total}] FAILED {pid} run {run_id}: {e}")

    tasks = [run_one(pid, run_id, run_dir) for pid, run_id, run_dir in all_pending]
    await asyncio.gather(*tasks)
    print(f"Batch '{batch_run_name}' complete.")
