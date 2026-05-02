# Agentic Retrieval Experiment Framework

A framework for measuring how well AI agents retrieve information and reason over large text corpora. Generates controlled experimental corpora with hidden "needles," runs agents against them, scores their responses, and produces analysis-ready data.

Designed for use with [Claude Code](https://docs.anthropic.com/en/docs/claude-code) and the Claude Agent SDK, but the generated corpora and answer keys can be used to evaluate any agent.

## Experiment Types

| Type | Description |
|------|-------------|
| `single_needle` | Find one hidden fact in the corpus |
| `multi_chain` | Follow a chain of N cross-references across files to reach a final value |
| `multi_reasoning` | Locate N scattered clues and synthesise them to answer a question |
| `multi_retrieval` | Retrieve N category-coherent pre-authored items from a large corpus and report each verbatim. Tests pure retrieval+retention without reasoning. |
| `pure_reasoning` | Reason across N facts handed in the prompt to derive a structured interval-shaped answer. No corpus, no retrieval. |

Each type can be parameterized across:
- **Content profiles**: e.g. Python repository, noir detective fiction
- **Corpus sizes**: e.g. 20k, 40k, 160k, 800k tokens
- **Reference clarity**: exact keyword, synonym, contextual paraphrase
- **Needle counts**: e.g. 2, 8, 16 items (multi-chain and multi-reasoning)
- **Discriminability**: easy, hard

Note: `pure_reasoning` has no corpus, so corpus_size, reference_clarity, and discriminability don't apply.

Define your own parameter grids in `experiments/*.yaml`.

## Pipeline

1. **Generate** — builds realistic background corpora from content profile templates and inserts needle payloads at controlled difficulty levels. `multi_retrieval` and `pure_reasoning` use hand-authored fixed-pool needles committed to YAML rather than LLM-generated needles; the other types use LLM-generated needles.
2. **Run** — executes the agent against each corpus with a retrieval question
3. **Judge** — scores agent responses against answer keys using rubric-based LLM evaluation (correctness + completeness)
4. **Analyse** — loads verdicts into notebooks for visualisation and interpretation

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

Filter to a specific experiment type:

```bash
poetry run python scripts/generate_parallel.py --workers 3 --experiments single_needle
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
