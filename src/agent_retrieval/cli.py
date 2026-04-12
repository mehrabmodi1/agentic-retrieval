from __future__ import annotations
import argparse
import asyncio
from pathlib import Path

from agent_retrieval.schema.batch import BatchConfig


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="agent-retrieval", description="Agent Retrieval Experiment Framework")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="Generate corpus and answer key")
    gen.add_argument("config_path", help="Path to experiment template YAML")
    gen.add_argument("--workspace", default="workspace", help="Workspace directory")

    gen_pool = sub.add_parser("generate-pool", help="Generate a background pool for a content profile")
    gen_pool.add_argument("profile_name", help="Content profile name (e.g. python_repo, noir_fiction)")
    gen_pool.add_argument("--workspace", default="workspace", help="Workspace directory")
    gen_pool.add_argument("--target-tokens", type=int, default=1_000_000, help="Target token count")

    run = sub.add_parser("run", help="Run experiments in a batch")
    run.add_argument("config_path", help="Path to batch YAML")
    run.add_argument("--resume", help="Resume a previous batch run (folder name)")
    run.add_argument("--workspace", default="workspace", help="Workspace directory")
    run.add_argument("--experiments-dir", default="experiments", help="Experiments directory")

    judge = sub.add_parser("judge", help="Judge completed runs")
    judge.add_argument("batch_run_name", help="Batch run folder name (e.g. smoke-test__20260408T133300)")
    judge.add_argument("--judge-model", default="claude-sonnet-4-6", help="Model for judging")
    judge.add_argument("--workspace", default="workspace", help="Workspace directory")
    judge.add_argument("--rejudge", action="store_true", help="Re-judge existing verdicts")

    analyze = sub.add_parser("analyze", help="Analyze judged results")
    analyze.add_argument("config_path", help="Path to batch YAML")
    analyze.add_argument("--workspace", default="workspace", help="Workspace directory")

    return parser.parse_args(argv)


async def _generate(args: argparse.Namespace) -> None:
    from agent_retrieval.generator import generate_experiment_v2
    from agent_retrieval.schema.template import ExperimentTemplate
    config_path = Path(args.config_path)
    template = ExperimentTemplate.from_yaml(config_path)
    print(f"Generating experiment '{template.experiment_type}'...")
    ids = await generate_experiment_v2(template, Path(args.workspace))
    print(f"Generated {len(ids)} parametrisations")


async def _generate_pool(args: argparse.Namespace) -> None:
    from agent_retrieval.generator.pool import generate_pool
    workspace_dir = Path(args.workspace)
    pool_dir = workspace_dir / "background_corpora" / args.profile_name
    print(f"Generating background pool for '{args.profile_name}'...")
    await generate_pool(args.profile_name, pool_dir, target_token_count=args.target_tokens)
    print(f"Done: pool at {pool_dir}")


async def _run(args: argparse.Namespace) -> None:
    from agent_retrieval.runner import run_batch
    batch = BatchConfig.from_yaml(Path(args.config_path))
    await run_batch(
        batch, Path(args.experiments_dir).resolve(), Path(args.workspace).resolve(),
        resume=args.resume,
    )


async def _judge(args: argparse.Namespace) -> None:
    from agent_retrieval.judge import judge_batch
    await judge_batch(
        args.batch_run_name, args.judge_model,
        Path(args.workspace).resolve(), rejudge=args.rejudge,
    )


def _analyze(args: argparse.Namespace) -> None:
    from agent_retrieval.analysis import run_analysis
    batch = BatchConfig.from_yaml(Path(args.config_path))
    run_analysis(batch.batch_name, Path(args.workspace))


def main() -> None:
    args = parse_args()
    if args.command == "generate-pool":
        asyncio.run(_generate_pool(args))
    elif args.command == "generate":
        asyncio.run(_generate(args))
    elif args.command == "run":
        asyncio.run(_run(args))
    elif args.command == "judge":
        asyncio.run(_judge(args))
    elif args.command == "analyze":
        _analyze(args)


if __name__ == "__main__":
    main()
