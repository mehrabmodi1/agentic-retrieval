# Agent Retrieval Experiment

## Environment

- **Package manager:** Poetry
- **Python:** 3.12+ (currently running 3.14)
- **Source package:** `src/agent_retrieval/` (installed as editable via Poetry)

### Commands

All commands must be run through Poetry to use the correct environment:

```bash
# Run tests
poetry run pytest -v

# Run a specific test file or class
poetry run pytest tests/test_insertion.py::TestExtractFragment -v

# Run the parallel generator
poetry run python scripts/generate_parallel.py --workers 3

# Run a single experiment type sequentially
poetry run python scripts/generate_parallel.py --workers 1 --experiments single_needle
```

**Do not** use bare `python` or `pytest` — always prefix with `poetry run`.

### Installing dependencies

```bash
poetry install            # all groups
poetry install --with dev # dev only (pytest, pytest-asyncio)
```

## Project Structure

- `experiments/*.yaml` — v2 experiment templates defining parameter grids
- `src/agent_retrieval/generator/` — corpus generation pipeline (pool → assemble → insert)
- `src/agent_retrieval/runner/` — agent runner (executes retrieval tasks against corpora)
- `src/agent_retrieval/judge/` — scoring and evaluation
- `src/agent_retrieval/analysis/` — analysis, figures, and reporting
- `scripts/` — CLI entry points for generation
- `workspace/` — runtime data (pools, corpora, answer keys, runs)
- `tests/` — pytest test suite

## Testing

- Framework: pytest with pytest-asyncio
- Async mode: `auto` (no need for `@pytest.mark.asyncio` on every test, but it's used in existing tests)
- Agent SDK calls (`claude_agent_sdk.query`) are mocked in tests via `unittest.mock.patch`
