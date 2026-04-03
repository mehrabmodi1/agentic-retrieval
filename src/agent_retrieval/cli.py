from __future__ import annotations
import argparse
import asyncio
import sys
from pathlib import Path
from agent_retrieval.schema.batch import BatchConfig
from agent_retrieval.schema.experiment import ExperimentSpec

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="agent-retrieval", description="Agent Retrieval Experiment Framework")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="Generate corpus and answer key")
    gen.add_argument("config_path", help="Path to spec YAML or batch YAML")
    gen.add_argument("--workspace", default="workspace", help="Workspace directory")

    gen_pool = sub.add_parser("generate-pool", help="Generate a background pool for a content profile")
    gen_pool.add_argument("profile_name", help="Content profile name (e.g. python_repo, noir_fiction)")
    gen_pool.add_argument("--workspace", default="workspace", help="Workspace directory")
    gen_pool.add_argument("--target-tokens", type=int, default=1_000_000, help="Target token count")

    run = sub.add_parser("run", help="Run experiments in a batch")
    run.add_argument("config_path", help="Path to batch YAML")
    run.add_argument("--workspace", default="workspace", help="Workspace directory")
    run.add_argument("--specs-dir", default="specs", help="Specs directory")
    run.add_argument("--experiments-dir", default="experiments", help="Experiments directory (v2)")

    judge = sub.add_parser("judge", help="Judge completed runs")
    judge.add_argument("config_path", help="Path to batch YAML")
    judge.add_argument("--workspace", default="workspace", help="Workspace directory")
    judge.add_argument("--rejudge", action="store_true", help="Re-judge existing verdicts")

    analyze = sub.add_parser("analyze", help="Analyze judged results")
    analyze.add_argument("config_path", help="Path to batch YAML")
    analyze.add_argument("--workspace", default="workspace", help="Workspace directory")
    analyze.add_argument("--specs-dir", default="specs", help="Specs directory")

    return parser.parse_args(argv)

def _is_batch_file(path: Path) -> bool:
    if "batch" in path.stem.lower() or path.parent.name == "batches":
        return True
    import yaml
    with open(path) as f:
        data = yaml.safe_load(f)
    return "batch_name" in data

def _is_v2_experiment(path: Path) -> bool:
    import yaml
    with open(path) as f:
        data = yaml.safe_load(f)
    return data.get("schema_version") == "2.0"

async def _generate(args: argparse.Namespace) -> None:
    config_path = Path(args.config_path)
    workspace_dir = Path(args.workspace)

    if _is_v2_experiment(config_path):
        from agent_retrieval.generator import generate_experiment_v2
        from agent_retrieval.schema.template import ExperimentTemplate
        template = ExperimentTemplate.from_yaml(config_path)
        print(f"Generating v2 experiment '{template.experiment_type}'...")
        ids = await generate_experiment_v2(template, workspace_dir)
        print(f"Generated {len(ids)} parametrisations")
        return

    if _is_batch_file(config_path):
        from agent_retrieval.generator import generate_experiment
        batch = BatchConfig.from_yaml(config_path)
        specs_dir = config_path.parent.parent / "specs"
        for run_config in batch.runs:
            spec = ExperimentSpec.from_yaml(specs_dir / f"{run_config.experiment_id}.yaml")
            print(f"Generating corpus for {spec.experiment_id}...")
            await generate_experiment(spec, workspace_dir)
            print(f"  Done: {spec.experiment_id}")
    else:
        from agent_retrieval.generator import generate_experiment
        spec = ExperimentSpec.from_yaml(config_path)
        print(f"Generating corpus for {spec.experiment_id}...")
        await generate_experiment(spec, workspace_dir)
        print(f"  Done: {spec.experiment_id}")

async def _generate_pool(args: argparse.Namespace) -> None:
    from agent_retrieval.generator.pool import generate_pool
    workspace_dir = Path(args.workspace)
    pool_dir = workspace_dir / "runner" / "pools" / args.profile_name
    print(f"Generating background pool for '{args.profile_name}'...")
    await generate_pool(args.profile_name, pool_dir, target_token_count=args.target_tokens)
    print(f"Done: pool at {pool_dir}")

async def _run(args: argparse.Namespace) -> None:
    from agent_retrieval.runner import run_batch
    batch = BatchConfig.from_yaml(Path(args.config_path))
    await run_batch(batch, Path(args.specs_dir), Path(args.workspace))

async def _judge(args: argparse.Namespace) -> None:
    from agent_retrieval.judge import judge_batch
    batch = BatchConfig.from_yaml(Path(args.config_path))
    await judge_batch(batch, Path(args.workspace), rejudge=args.rejudge)

def _analyze(args: argparse.Namespace) -> None:
    from agent_retrieval.analysis import run_analysis
    batch = BatchConfig.from_yaml(Path(args.config_path))
    run_analysis(batch.batch_name, Path(args.workspace), Path(args.specs_dir))

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
