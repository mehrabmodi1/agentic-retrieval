"""Parallel corpus generation orchestrator.

Dynamically discovers remaining work, spawns N parallel workers,
and is fully idempotent — safe to re-run after interruptions.

Usage:
    poetry run python scripts/generate_parallel.py --workers 3
    poetry run python scripts/generate_parallel.py --workers 3 --experiments single_needle
    poetry run python scripts/generate_parallel.py --workers 3 --dry-run
"""
import argparse
import asyncio
import shutil
import time
from pathlib import Path

from agent_retrieval.generator.assembler import assemble_corpus
from agent_retrieval.generator.grid import expand_grid
from agent_retrieval.generator.insertion import InsertionStats, insert_payloads
from agent_retrieval.schema.template import ExperimentTemplate


def discover_remaining(
    experiments_dir: Path,
    workspace_dir: Path,
    experiment_filter: list[str] | None = None,
) -> list[tuple[str, str, ExperimentTemplate]]:
    """Return (experiment_name, parametrisation_id, template) for all incomplete work."""
    answer_keys_dir = workspace_dir / "judge" / "answer_keys"
    existing = {p.stem for p in answer_keys_dir.glob("*.yaml")} if answer_keys_dir.exists() else set()

    remaining = []
    for yaml_path in sorted(experiments_dir.glob("*.yaml")):
        name = yaml_path.stem
        if experiment_filter and name not in experiment_filter:
            continue
        template = ExperimentTemplate.from_yaml(yaml_path)
        for param in expand_grid(template):
            if param.parametrisation_id not in existing:
                remaining.append((name, param.parametrisation_id, template))
    return remaining


def _format_stats(stats: InsertionStats) -> str:
    """Format InsertionStats into a compact one-line summary."""
    duration_s = stats.duration_ms / 1000
    edits = stats.tool_calls.count("Edit")
    writes = stats.tool_calls.count("Write")
    ak = "written" if stats.answer_key_written else "MISSING"
    tokens = f"{stats.input_tokens + stats.output_tokens:,}tok"
    cost = f"${stats.total_cost_usd:.4f}" if stats.total_cost_usd else "$?"

    parts = [
        f"{duration_s:.1f}s",
        f"{stats.num_turns}t",
        cost,
        tokens,
        f"edits:{edits}",
        f"writes:{writes}",
        f"ak:{ak}",
    ]
    if stats.is_error:
        parts.append(f"ERR:{'; '.join(stats.errors)}")
    return "  ".join(parts)


class ProgressTracker:
    """Thread-safe progress tracker for parallel generation."""

    def __init__(self, total: int):
        self.total = total
        self.completed = 0
        self.failed = 0
        self.skipped = 0
        self.total_cost = 0.0
        self.total_tokens = 0
        self.start_time = time.time()

    def record(self, stats: InsertionStats | None, success: bool) -> None:
        if stats is None:
            self.skipped += 1
            return
        if success:
            self.completed += 1
        else:
            self.failed += 1
        self.total_cost += stats.total_cost_usd
        self.total_tokens += stats.input_tokens + stats.output_tokens

    def summary_line(self) -> str:
        elapsed = time.time() - self.start_time
        done = self.completed + self.failed + self.skipped
        rate = self.completed / elapsed * 60 if elapsed > 0 else 0
        return (
            f"=== {done}/{self.total} "
            f"({self.completed} ok, {self.failed} fail, {self.skipped} skip) | "
            f"${self.total_cost:.4f} | {self.total_tokens:,}tok | "
            f"{rate:.1f}/min | {elapsed:.0f}s ==="
        )


async def process_one(
    experiment_name: str,
    parametrisation_id: str,
    template: ExperimentTemplate,
    workspace_dir: Path,
    worker_id: int,
) -> tuple[str, bool, str, InsertionStats | None]:
    """Process a single parametrisation. Returns (pid, success, message, stats)."""
    params = expand_grid(template)
    param = next(p for p in params if p.parametrisation_id == parametrisation_id)

    pid = param.parametrisation_id
    corpus_dir = workspace_dir / "runner" / "corpora" / pid
    answer_key_path = workspace_dir / "judge" / "answer_keys" / f"{pid}.yaml"
    pool_dir = workspace_dir / "background_corpora" / param.content_profile

    # Double-check idempotency (another worker may have finished it)
    if answer_key_path.exists():
        return (pid, True, "already exists", None)

    if not pool_dir.exists() or not any(pool_dir.rglob("*.md")):
        return (pid, False, f"pool missing: {pool_dir}", None)

    try:
        assemble_corpus(pool_dir, corpus_dir, param)
        stats = await insert_payloads(template, param, corpus_dir, answer_key_path)
        if stats and stats.is_error:
            return (pid, False, f"agent error: {'; '.join(stats.errors)}", stats)
        if stats and not stats.answer_key_written:
            return (pid, False, "answer key not written", stats)
        return (pid, True, "done", stats)
    except Exception as e:
        return (pid, False, str(e), None)


async def run_workers(
    remaining: list[tuple[str, str, ExperimentTemplate]],
    workspace_dir: Path,
    max_workers: int,
) -> None:
    semaphore = asyncio.Semaphore(max_workers)
    total = len(remaining)
    tracker = ProgressTracker(total)

    async def worker(idx: int, exp_name: str, pid: str, template: ExperimentTemplate):
        async with semaphore:
            seq = tracker.completed + tracker.failed + tracker.skipped + 1
            print(f"[{seq}/{total}] W{idx % max_workers}: {pid}")
            result_pid, success, msg, stats = await process_one(
                exp_name, pid, template, workspace_dir, idx,
            )
            tracker.record(stats, success)

            if stats:
                status = "OK " if success else "FAIL"
                print(f"  {status} {_format_stats(stats)}  ({msg})")
            else:
                status = "OK " if success else "FAIL"
                print(f"  {status} {msg}")

            # Print summary every 10 completions
            done = tracker.completed + tracker.failed + tracker.skipped
            if done % 10 == 0 or done == total:
                print(tracker.summary_line())

    tasks = [
        worker(i, exp_name, pid, template)
        for i, (exp_name, pid, template) in enumerate(remaining)
    ]
    await asyncio.gather(*tasks)

    print(f"\n{tracker.summary_line()}")


def main():
    parser = argparse.ArgumentParser(description="Parallel corpus generation")
    parser.add_argument("--workers", type=int, default=3, help="Number of concurrent workers")
    parser.add_argument("--workspace", default="workspace", help="Workspace directory")
    parser.add_argument("--experiments-dir", default="experiments", help="Experiments directory")
    parser.add_argument("--experiments", nargs="*", help="Filter to specific experiment types")
    parser.add_argument("--dry-run", action="store_true", help="Just show what would be done")
    args = parser.parse_args()

    workspace_dir = Path(args.workspace).resolve()
    experiments_dir = Path(args.experiments_dir).resolve()

    # Pre-flight: delete any corpus dirs that lack a matching answer key
    corpora_dir = workspace_dir / "runner" / "corpora"
    ak_dir = workspace_dir / "judge" / "answer_keys"
    if corpora_dir.exists():
        existing_aks = {p.stem for p in ak_dir.glob("*.yaml")} if ak_dir.exists() else set()
        cleaned = 0
        for corpus in sorted(corpora_dir.iterdir()):
            if corpus.is_dir() and corpus.name not in existing_aks:
                shutil.rmtree(corpus)
                cleaned += 1
        if cleaned:
            print(f"Pre-flight cleanup: deleted {cleaned} corrupted corpora (missing answer keys)")

    remaining = discover_remaining(experiments_dir, workspace_dir, args.experiments)

    # Summary
    by_type: dict[str, int] = {}
    for exp_name, pid, _ in remaining:
        by_type[exp_name] = by_type.get(exp_name, 0) + 1

    print(f"Remaining: {len(remaining)} parametrisations")
    for exp_name, count in sorted(by_type.items()):
        print(f"  {exp_name}: {count}")
    print(f"Workers: {args.workers}")

    if args.dry_run:
        for exp_name, pid, _ in remaining:
            print(f"  would generate: {pid}")
        return

    if not remaining:
        print("Nothing to do!")
        return

    asyncio.run(run_workers(remaining, workspace_dir, args.workers))


if __name__ == "__main__":
    main()
