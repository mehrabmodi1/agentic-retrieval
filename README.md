# The Multi-Needle Problem

**Agentic retrieval and reasoning fails to scale.**

A parametric experiment measuring how well an AI agent (Claude Sonnet 4.6, effort: low) retrieves information and reasons over large text corpora. Single-fact retrieval works well (~90% accuracy), but performance degrades sharply with task complexity — multi-chain retrieval drops to ~50%, and multi-reasoning to ~20%.

See the full analysis: [`notebooks/full-sweep_sonnet-4-6_effort_low_20260409.ipynb`](notebooks/full-sweep_sonnet-4-6_effort_low_20260409.ipynb)

## Experiment Types

| Type | Description | Accuracy |
|------|-------------|----------|
| `single_needle` | Find one hidden fact in the corpus | ~90% |
| `multi_chain` | Follow a chain of N cross-references to a final value | ~50% |
| `multi_reasoning` | Locate N scattered clues and synthesise an answer | ~20% |

Each type is parameterized across:
- **Content profiles**: Python repository, noir detective fiction
- **Corpus sizes**: 20k, 40k, 160k, 800k tokens
- **Reference clarity**: exact keyword, synonym, contextual paraphrase
- **Needle counts**: 2, 8, 16 items (multi-chain and multi-reasoning)
- **Discriminability**: easy, hard

This produces a full factorial grid of 336 experimental conditions, each run 3 times.

## How It Works

1. **Generate** — builds realistic background corpora and inserts needle payloads at controlled difficulty levels
2. **Run** — executes the agent (via Claude Agent SDK) against each corpus with a retrieval question
3. **Judge** — scores agent responses against answer keys using rubric-based LLM evaluation (correctness + completeness)
4. **Analyse** — loads verdicts into a notebook for visualisation and interpretation

## Getting Started

### Prerequisites

- Python 3.12+
- [Poetry](https://python-poetry.org/docs/#installation)
- An Anthropic API key (set `ANTHROPIC_API_KEY` in your environment)

### Install

```bash
git clone https://github.com/mehrabmodi1/agentic-retrieval.git
cd agentic-retrieval
poetry install
```

### Run Tests

```bash
poetry run pytest -v
```

### Generate Corpora

```bash
poetry run python scripts/generate_parallel.py --workers 3
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
workspace/            Runtime data (pools, corpora, answer keys, runs, verdicts)
notebooks/            Analysis notebooks
tests/                Test suite
```
