# Agent Retrieval Experiment

A framework for measuring how well AI agents retrieve information from large corpora. Generates controlled experimental corpora with hidden "needles" (inserted facts, config values, cross-references), then runs AI agents against them and scores their retrieval performance.

## What It Does

1. **Generates** realistic background corpora (Python repos, noir fiction) from content profile templates
2. **Inserts** needle payloads at controlled difficulty levels — varying discriminability (easy/hard), reference clarity (exact/synonym/contextual), and needle count
3. **Runs** AI agents (via Claude Agent SDK) against the corpora with retrieval questions
4. **Judges** agent responses against answer keys using rubric-based scoring
5. **Analyzes** results across parameter grids to identify retrieval performance patterns

## Experiment Types

| Type | Description |
|------|-------------|
| `single_needle` | One hidden fact; agent must find and report it |
| `multi_chain` | N linked needles forming a chain; agent follows references to a final value |
| `multi_reasoning` | N independent needles; agent must find all and reason across them |

Each type is parameterized across content profiles, corpus sizes (20k–800k tokens), discriminability levels, reference clarity levels, and needle counts — producing a full factorial grid of experimental conditions.

## Getting Started

### Prerequisites

- Python 3.12+
- [Poetry](https://python-poetry.org/docs/#installation)
- An Anthropic API key (set `ANTHROPIC_API_KEY` in your environment)

### Install

```bash
git clone <repo-url>
cd agent_retrieval_expt
poetry install
```

### Run Tests

```bash
poetry run pytest -v
```

### Generate Corpora

Generate all remaining parametrisations across experiment types:

```bash
poetry run python scripts/generate_parallel.py --workers 3
```

Filter to a specific experiment type:

```bash
poetry run python scripts/generate_parallel.py --workers 3 --experiments single_needle
```

Preview what would be generated without running:

```bash
poetry run python scripts/generate_parallel.py --workers 3 --dry-run
```

Generate a single parametrisation:

```bash
poetry run python scripts/generate_parallel.py --workers 1 --experiments single_needle
```

### Project Layout

```
experiments/          Experiment template YAMLs (parameter grids)
src/agent_retrieval/  Source package
  generator/          Corpus generation pipeline
  runner/             Agent execution
  judge/              Scoring and evaluation
  analysis/           Results analysis and figures
  schema/             Pydantic models
scripts/              CLI entry points
workspace/            Runtime data (pools, corpora, answer keys)
tests/                Test suite
```
