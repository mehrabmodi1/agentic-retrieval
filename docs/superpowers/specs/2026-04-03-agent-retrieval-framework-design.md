# Agent Retrieval Experiment Framework — Design Spec

## Overview

A framework for experimentally measuring how well AI agents retrieve information from large document sets and reason about it. The framework generates synthetic corpora with embedded "needles," runs agents against them via the Claude Agent SDK, judges their outputs against ground truth, and produces standardized analysis.

## Key Dimensions Under Study

- **Background corpus size**: token count, file count, folder depth/structure
- **Retrieval complexity**: single-item retrieval (needle in haystack) vs. chain-of-dependency retrieval
- **Reasoning complexity**: reasoning across multiple items with increasing interdependency

## Architecture: Single Package, Directory Contracts

One Python package (`agent_retrieval`) with four components that communicate exclusively through well-defined directory structures. Leakage prevention is enforced by what each component is pointed at, not by package boundaries. The Agent SDK sandbox is the runtime enforcement mechanism.

### Project Layout

```
agent_retrieval_expt/
├── src/agent_retrieval/
│   ├── schema/              # Pydantic models for specs, answer keys, verdicts
│   ├── generator/           # Phase 1 (background) + Phase 2 (payload insertion)
│   │   └── profiles/        # Content profile plugins (python_repo, technical_docs, etc.)
│   ├── runner/              # Batch orchestration, agent session management
│   ├── judge/               # LLM scoring + JSONL metrics extraction
│   └── analysis/            # Aggregation, tables, figures, report generation
├── specs/                   # Experiment spec YAMLs (human-authored)
├── batches/                 # Batch config YAMLs (human-authored)
├── notebooks/
│   └── analysis_template.ipynb
├── workspace/               # All generated/runtime data
│   ├── runner/
│   │   ├── corpora/{exp_id}/
│   │   └── runs/{batch_name}/{exp_id}/{run_id}/
│   ├── judge/
│   │   ├── answer_keys/{exp_id}.yaml
│   │   └── judgements/{batch_name}/{exp_id}/
│   └── analysis/{batch_name}/
├── pyproject.toml
└── tests/
```

### Isolation Matrix

| Component  | Writes to                                    | Reads from                                                        |
|------------|----------------------------------------------|-------------------------------------------------------------------|
| Generator  | `workspace/runner/corpora/`                  | `specs/`                                                          |
| Generator  | `workspace/judge/answer_keys/`               | `specs/`                                                          |
| Runner     | `workspace/runner/runs/`                     | `specs/`, `batches/`                                              |
| Agent      | — (sandboxed, read-only)                     | `workspace/runner/corpora/{exp_id}/`                              |
| Judge      | `workspace/judge/judgements/`                | `workspace/judge/answer_keys/`, `workspace/runner/runs/`          |
| Analysis   | `workspace/analysis/`                        | `workspace/judge/judgements/`, `specs/`                            |

Leakage safeguards:
- The generator cleans metadata/comments from generated files that could hint at the generation process
- Answer key filenames use experiment IDs, not descriptive names
- The runner's system prompt to the agent contains only the question string — no experiment ID, spec content, or batch context
- The agent is sandboxed to `workspace/runner/corpora/{exp_id}/` via Agent SDK `working_directory`

---

## Component 1: Experiment Spec Schema

YAML files validated by Pydantic models. Each spec defines one experiment.

### Spec Shape

```yaml
schema_version: "1.0"
experiment_id: "needle-in-haystack-001"
experiment_type: "needle_in_haystack"  # or "chain_of_retrieval", "chain_of_reasoning"

corpus:
  content_profile: "python_repo"       # pluggable profile name
  target_token_count: 500_000
  target_file_count: 200
  folder_depth: 4
  folder_distribution: "balanced"      # or "skewed", "flat"
  generation_model: "haiku"            # cheap model for background
  red_herring_density: "medium"        # how much noise resembles the signal

payload:
  insertion_model: "sonnet"            # smarter model for placing needles
  red_herring_hint: "Configuration values with similar names like connection_pool_size, request_timeout, but with different values"
  items:
    - item_id: "target_001"
      item_type: "config_value"        # or "function_def", "fact", "cross_reference"
      content_hint: "A database connection timeout set to a specific number of seconds"
      placement:
        strategy: "random_file"        # or "specific_depth", "specific_filetype"
        depth: null
        filetype: null
      camouflage: "high"               # how much it blends in
    - item_id: "target_002"
      depends_on: "target_001"         # for chain experiments
      item_type: "cross_reference"
      content_hint: "A module that imports and uses the timeout from target_001"
      placement:
        strategy: "specific_depth"
        depth: 3
      camouflage: "medium"

question: "What is the database connection timeout value used in production, and which module consumes it?"

rubric_criteria:
  - criterion: "correctness"
    weight: 1.0
  - criterion: "completeness"
    weight: 1.0

runner:
  n_repeats: 5
  agent_model: "sonnet"
  max_tokens: 200_000                  # token budget per session
  allowed_tools: ["Read", "Glob", "Grep", "Bash"]
```

Key design decisions:
- `content_hint` tells the generator what kind of content to insert, not the exact text. The generator produces the actual content; the exact text goes into the answer key — never the spec.
- `red_herring_hint` gives the background generator guidance on what plausible distractors should look like.
- `depends_on` creates the dependency graph for chain experiments. The generator resolves insertion order (leaves first, then dependents).
- `rubric_criteria` defines what dimensions the judge evaluates on — but NOT the expected values. Those are generated and stored in the answer key.
- Runner config is co-located with the experiment so each experiment can control model, tool access, and budget.

---

## Component 2: Synthetic Data Generator

Reads a spec YAML, produces a corpus directory and an answer key. Two phases.

### Phase 1: Background Corpus Generation

1. Reads the spec's `corpus` section
2. Loads a **content profile** — a pluggable module that knows how to prompt a cheap model for a particular document type
3. Generates the folder structure deterministically from spec params (depth, distribution, file count)
4. Fills each file by prompting the cheap model with context-appropriate instructions
5. Uses the `red_herring_hint` from the payload section to seed some background files with plausible distractors
6. Tracks cumulative tokens to hit the target token count

### Phase 2: Payload Insertion

1. Reads the spec's `payload` section
2. Resolves the `depends_on` graph to determine insertion order (leaves first, then dependents)
3. For each item, prompts the insertion model with:
   - The `content_hint`
   - The target file's existing content (so the needle blends in)
   - The `camouflage` level
   - Context from any items this one `depends_on` (so cross-references are internally consistent)
4. The insertion model returns: the modified file content AND a structured block describing exactly what was inserted
5. Generator writes the modified file to the corpus and appends the structured block to the answer key

### Content Profiles

Python classes with a common interface:

```python
class ContentProfile(ABC):
    def generate_folder_structure(self, spec: CorpusSpec) -> list[Path]: ...
    def generate_file_prompt(self, path: Path, context: GenerationContext) -> str: ...
```

New document types are added by implementing a new profile. The spec references them by name.

### Answer Key Output

Written to `workspace/judge/answer_keys/{exp_id}.yaml`:

```yaml
experiment_id: "needle-in-haystack-001"
generated_at: "2026-04-03T10:30:00Z"
items:
  - item_id: "target_001"
    inserted_text: "CONNECTION_TIMEOUT = 42  # seconds, per SRE policy"
    file_path: "src/config/database.py"
    line_range: [47, 47]
    context_summary: "Inserted as a module-level constant in database config"
  - item_id: "target_002"
    inserted_text: "from config.database import CONNECTION_TIMEOUT"
    file_path: "src/services/pool_manager.py"
    line_range: [3, 3]
    context_summary: "Added as import in the connection pool service"
expected_answers:
  question: "What is the database connection timeout value used in production, and which module consumes it?"
  correctness: "CONNECTION_TIMEOUT is 42 seconds, defined in src/config/database.py"
  completeness: "It is consumed by src/services/pool_manager.py via direct import"
rubric_criteria:
  - criterion: "correctness"
    weight: 1.0
  - criterion: "completeness"
    weight: 1.0
```

### Leakage Prevention

The generator is the only component that crosses the runner/judge boundary. After generation:
- No symlinks or shared references between `workspace/runner/corpora/` and `workspace/judge/answer_keys/`
- No metadata in the corpus files pointing back to the answer key or generation process
- Generated files are scrubbed of any generation artifacts

---

## Component 3: Experiment Runner

Orchestrates agent sessions via the Claude Agent SDK. Reads specs and batch files, spawns agents, collects outputs. Never sees answer keys.

### Batch File

Human-authored YAML at `batches/{batch_name}.yaml`:

```yaml
batch_name: "scaling-test-v1"
max_parallel: 4
retry_failed: true
judge_model: "opus"

runs:
  - experiment_id: "needle-001"
    n_repeats: 5
  - experiment_id: "needle-002"
    n_repeats: 5
    judge_model: "sonnet"     # per-experiment override
  - experiment_id: "chain-003"
    n_repeats: 10
    agent_model: "opus"       # per-experiment override
```

CLI:

```bash
agent-retrieval run batches/scaling-test-v1.yaml
```

### Agent Sandboxing

```python
session = AgentSession(
    model=spec.runner.agent_model,
    system_prompt=f"Answer the following question by searching the provided codebase:\n\n{spec.question}",
    working_directory=f"workspace/runner/corpora/{spec.experiment_id}/",
    allowed_tools=spec.runner.allowed_tools,
    max_tokens=spec.runner.max_tokens,
)
```

Isolation:
- `working_directory` restricts the agent's filesystem view to the corpus only
- `allowed_tools` is an explicit whitelist
- System prompt contains only the question — no hints, rubric, or experiment metadata
- Each session gets a unique `run_id` (UUID), injected into the system prompt so the agent includes it in its response

### Run State Machine

Each run has state persisted to `workspace/runner/runs/{batch_name}/{exp_id}/{run_id}/state.yaml`:

```
pending → running → completed
                  → failed → pending (on retry)
```

On startup, the runner scans for interrupted runs (state = `running`) and resets them to `pending`. This is the recovery mechanism.

### Run Output

Per run directory (`workspace/runner/runs/{batch_name}/{exp_id}/{run_id}/`):

```
state.yaml          # status, timestamps, spec reference, claude_code_version
response.json       # agent's final answer
session.jsonl       # full Agent SDK session log
```

The runner captures `claude --version` at batch start and records it in each run's `state.yaml` for reproducibility.

### Batch Orchestration

- `asyncio` with semaphore for parallelism (`max_parallel`)
- Per-run token budgets from the spec
- Progress logging: `[3/25] Running needle-001 repeat 3...`
- Idempotent: skips completed runs, retries failed if `retry_failed: true`

---

## Component 4: LLM Judge

Reads agent outputs and answer keys, produces structured verdicts. No access to corpora or specs.

### Inputs

- `workspace/judge/answer_keys/{exp_id}.yaml` — ground truth
- `workspace/runner/runs/{batch_name}/{exp_id}/{run_id}/response.json` — agent's answer
- `workspace/runner/runs/{batch_name}/{exp_id}/{run_id}/session.jsonl` — full session log

### Judging Process

1. **Correctness & completeness scoring**: Prompts a judge model (configurable in batch file, defaults to Opus) with the answer key's expected answers, rubric criteria, and the agent's response. Returns a structured score per criterion (0.0 to 1.0).

2. **Session metrics extraction**: Parses the JSONL directly (no LLM needed):
   - Total context tokens consumed
   - Tool calls by type with counts
   - Number of turns
   - Duration from timestamps

### Verdict Output

Written to `workspace/judge/judgements/{batch_name}/{exp_id}/{run_id}.yaml`:

```yaml
experiment_id: "needle-001"
run_id: "a1b2c3d4"
batch_name: "scaling-test-v1"

scores:
  - criterion: "correctness"
    score: 0.85
    weight: 1.0
    reasoning: "Found the correct timeout value but cited wrong file path"
  - criterion: "completeness"
    score: 1.0
    weight: 1.0
    reasoning: "Identified both the source and consuming module"
  weighted_score: 0.925

session_metrics:
  total_context_tokens: 84_320
  total_turns: 7
  tool_calls:
    Grep: 12
    Read: 8
    Glob: 3
    Bash: 0
  duration_seconds: 45.2
```

### CLI

```bash
agent-retrieval judge batches/scaling-test-v1.yaml
```

Idempotent: skips runs that already have a verdict. Use `--rejudge` to re-evaluate.

### Judge Model Configuration

Specified in the batch file at batch level with per-experiment overrides. Allows comparing judge models across batches.

---

## Component 5: Analysis Module

Reads judgement files and experiment specs, produces summary tables, figures, and reports. Purely computational — no LLM calls.

### Inputs

- `workspace/judge/judgements/{batch_name}/**/*.yaml` — all verdicts
- `specs/` — experiment type and parameterization metadata

### Outputs

Written to `workspace/analysis/{batch_name}/`:

```
summary.csv                          # one row per experiment x repeat
tables/
├── accuracy_by_type.csv             # mean/std accuracy per experiment type
├── accuracy_by_param.csv            # accuracy by parameterization within type
└── tool_usage_by_type.csv           # tool call distributions per type
figures/
├── accuracy_vs_corpus_size.png      # how accuracy degrades with corpus size
├── accuracy_vs_chain_length.png     # for chain experiments
├── context_usage_vs_corpus_size.png # context consumption scaling
├── tool_distribution_by_type.png    # stacked bar: tool mix per experiment type
└── cross_type_comparison.png        # metrics across types on shared dimensions
report.html                          # single-page summary with embedded figures/tables
```

### Standard Analyses

| Metric                    | Sliced by                                                     |
|---------------------------|---------------------------------------------------------------|
| Accuracy (weighted score) | corpus size, chain length, reasoning complexity, camouflage   |
| Completeness score        | same dimensions                                               |
| Context tokens consumed   | same dimensions                                               |
| Tool call counts          | by tool type, per experiment type                             |
| Variance across repeats   | error bars / confidence intervals on all of the above         |

Cross-type comparison overlays metrics across experiment types on shared dimensions (e.g. corpus size).

### Implementation

- pandas for aggregation
- matplotlib/seaborn for figures
- Jinja template for `report.html` — single file, no frontend framework

### Interactive Notebook

`notebooks/analysis_template.ipynb` — parameterized notebook that imports the same analysis functions:

```python
# Cell 1: Configuration
BATCH_NAME = "scaling-test-v1"

# Cell 2: Load data
from agent_retrieval.analysis import load_batch_results
results = load_batch_results(BATCH_NAME)

# Cell 3+: Analysis sections using shared functions
from agent_retrieval.analysis import (
    accuracy_by_type,
    accuracy_by_param,
    tool_usage_by_type,
    plot_accuracy_vs_corpus_size,
    plot_context_usage,
    plot_cross_type_comparison,
)
```

### CLI

```bash
agent-retrieval analyze batches/scaling-test-v1.yaml
```

---

## CLI Summary

```bash
# Generate corpus + answer key for a single spec
agent-retrieval generate specs/needle-001.yaml

# Generate corpora for all experiments referenced in a batch
agent-retrieval generate batches/scaling-test-v1.yaml

# Run all experiments in batch
agent-retrieval run batches/scaling-test-v1.yaml

# Judge all completed runs in batch
agent-retrieval judge batches/scaling-test-v1.yaml

# Produce analysis outputs for batch
agent-retrieval analyze batches/scaling-test-v1.yaml
```

The `generate` command auto-detects whether it received a spec file or a batch file. When given a batch, it generates corpora for all referenced experiments, skipping any that already exist.

---

## Technology Stack

- Python 3.12+
- Pydantic for schema validation
- Claude Agent SDK for agent sessions
- asyncio for parallel agent orchestration
- PyYAML for spec/batch file parsing
- pandas for data aggregation
- matplotlib / seaborn for figures
- Jinja2 for HTML report templating
- pytest for testing
