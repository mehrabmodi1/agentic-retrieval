from __future__ import annotations
import uuid
from pathlib import Path
from agent_retrieval.schema.run_state import RunState


class RunStateManager:
    def __init__(self, runs_dir: Path):
        self.runs_dir = runs_dir

    def create_pending_runs(
        self, batch_name: str, parametrisation_id: str, n_runs: int,
        claude_version: str, agent_model: str, effort_mode: str,
        max_turns: int, allowed_tools: list[str],
    ) -> list[str]:
        run_ids = []
        for _ in range(n_runs):
            run_id = uuid.uuid4().hex[:12]
            state = RunState(
                parametrisation_id=parametrisation_id, run_id=run_id,
                batch_name=batch_name, status="pending", claude_code_version=claude_version,
                agent_model=agent_model, effort_mode=effort_mode,
                max_turns=max_turns, allowed_tools=allowed_tools,
            )
            run_dir = self.runs_dir / batch_name / parametrisation_id / run_id
            state.to_yaml(run_dir / "state.yaml")
            run_ids.append(run_id)
        return run_ids

    def recover_interrupted(self, batch_name: str) -> list[str]:
        recovered = []
        batch_dir = self.runs_dir / batch_name
        if not batch_dir.exists():
            return recovered
        for state_path in batch_dir.rglob("state.yaml"):
            state = RunState.from_yaml(state_path)
            if state.status == "running":
                state.status = "pending"
                state.to_yaml(state_path)
                recovered.append(state.run_id)
        return recovered

    def get_runs_by_status(
        self, batch_name: str, parametrisation_id: str, status: str,
    ) -> list[tuple[str, Path]]:
        exp_dir = self.runs_dir / batch_name / parametrisation_id
        if not exp_dir.exists():
            return []
        results = []
        for run_dir in sorted(exp_dir.iterdir()):
            state_path = run_dir / "state.yaml"
            if state_path.exists():
                state = RunState.from_yaml(state_path)
                if state.status == status:
                    results.append((state.run_id, run_dir))
        return results

    def update_status(self, run_dir: Path, status: str, **extra_fields: str) -> None:
        state = RunState.from_yaml(run_dir / "state.yaml")
        state.status = status
        for k, v in extra_fields.items():
            setattr(state, k, v)
        state.to_yaml(run_dir / "state.yaml")
