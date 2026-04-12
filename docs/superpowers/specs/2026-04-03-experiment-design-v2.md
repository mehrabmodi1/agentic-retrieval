# Experiment Design v2 — Templates, Grids, and Agent-Based Generation

## Overview

This spec defines the experiment design system for the agent retrieval framework. It replaces the v1 approach (one YAML spec per experiment) with a template + parameter grid system, introduces two content profiles, and restructures the generator to use Agent SDK calls with shared background pools.

**Relationship to v1 spec:** This is an additive design that changes how experiments are defined and generated. The runner, judge, and analysis components remain structurally the same but consume the new schema. The generator is substantially reworked.

---

## Experiment Type Taxonomy

Three experiment types, each with distinct cognitive demands on the agent:

| Type | What the agent must do | Key challenge |
|---|---|---|
| `single_needle` | Find one piece of information and report it | Retrieval from noise |
| `multi_chain` | Follow a sequence of N linked clues. Question provides the entry point. Each link points to the next; previous links can be forgotten after following. | Sequential navigation, short-term memory |
| `multi_reasoning` | Find N independent pieces of information, hold them all, answer a question that requires combining them | Parallel retrieval + synthesis, long-term memory |

---

## Parameter Space

### Universal Parameters (all types)

| Parameter | Values | Description |
|---|---|---|
| `content_profile` | `python_repo`, `noir_fiction` | Background corpus domain |
| `corpus_token_count` | 20000, 40000, 160000, 800000 | Total token budget for the sampled corpus |
| `discriminability` | `easy`, `hard` | Combined measure: how well the needle blends in + density of near-miss distractors |
| `reference_clarity` | `exact`, `synonym`, `contextual` | Semantic distance between question phrasing and needle identifiers |

### Multi-Only Parameter

| Parameter | Values | Description |
|---|---|---|
| `n_items` | 2, 8, 16 | Chain length (multi_chain) or independent item count (multi_reasoning) |

### Grid Sizes (full cartesian product)

- **single_needle:** 2 x 4 x 2 x 3 = **48 parametrisations**
- **multi_chain:** 2 x 4 x 2 x 3 x 3 = **144 parametrisations**
- **multi_reasoning:** 2 x 4 x 2 x 3 x 3 = **144 parametrisations**

### Parametrisation IDs

Each parametrisation gets a deterministic ID derived from its parameters:

```
single_needle__python_repo__20k__easy__exact
multi_chain__noir_fiction__160k__hard__synonym__n8
```

---

## Unified Discriminability Rubric

This rubric is included in every payload insertion agent's system prompt. It is profile-agnostic and defined purely in terms of cognitive load.

### Discriminability Levels

| Level | Retrieval method | Cognitive demand |
|---|---|---|
| `easy` | Findable by exact string search on terms in the question | Locate and report |
| `hard` | Embedded in surrounding context; requires reading and comprehending surrounding content to identify | Comprehend context, extract implicit information |

### Reference Clarity Levels

| Level | Definition |
|---|---|
| `exact` | Question uses the same identifier/name as the needle |
| `synonym` | Question uses a different term for the same concept |
| `contextual` | Question describes the concept indirectly; requires domain understanding to connect |

---

## Experiment YAML Format (v2)

One YAML file per experiment type. The grid is defined inline. Three files total: `experiments/single_needle.yaml`, `experiments/multi_chain.yaml`, `experiments/multi_reasoning.yaml`.

### single_needle.yaml

```yaml
schema_version: "2.0"
experiment_type: "single_needle"

payload:
  item_type: "config_value"

question_examples:
  python_repo:
    easy_exact:
      question: "What is the value of MAX_POOL_SIZE defined in this codebase?"
      needle: "MAX_POOL_SIZE = 25"
      answer: "25"
    easy_synonym:
      question: "What is the maximum number of connections in the database pool?"
      needle: "MAX_POOL_SIZE = 25"
      answer: "25, configured as MAX_POOL_SIZE"
    easy_contextual:
      question: "How many concurrent connections does the application allow?"
      needle: "MAX_POOL_SIZE = 25"
      answer: "The application allows 25 concurrent connections, configured via MAX_POOL_SIZE in the connection pool settings."
    hard_exact:
      question: "What is the value of CACHE_TTL_MS defined in this codebase?"
      needle: "ttl_ms = int(os.environ.get('CACHE_DURATION', 1000 * 60 * 30))  # fallback: 30min"
      answer: "1,800,000 milliseconds (30 minutes), set as a fallback default in the environment variable lookup"
    hard_synonym:
      question: "What is the cache expiration duration in this application?"
      needle: "ttl_ms = int(os.environ.get('CACHE_DURATION', 1000 * 60 * 30))  # fallback: 30min"
      answer: "30 minutes (1,800,000 ms), configured as a fallback default via the CACHE_DURATION environment variable"
    hard_contextual:
      question: "How long does the application cache responses before invalidating them?"
      needle: "ttl_ms = int(os.environ.get('CACHE_DURATION', 1000 * 60 * 30))  # fallback: 30min"
      answer: "30 minutes, derived from the default fallback value of 1000 * 60 * 30 milliseconds in the cache configuration"
  noir_fiction:
    easy_exact:
      question: "What was the name of the bar where Mickey Doyle was last seen?"
      needle: "Mickey Doyle had been drinking at The Silver Curtain since noon."
      answer: "The Silver Curtain"
    easy_synonym:
      question: "What establishment was Mickey Doyle's last known drinking spot?"
      needle: "Mickey Doyle had been drinking at The Silver Curtain since noon."
      answer: "The Silver Curtain"
    easy_contextual:
      question: "Where was the victim's last known location before disappearing?"
      needle: "Mickey Doyle had been drinking at The Silver Curtain since noon."
      answer: "The Silver Curtain, a bar where he had been drinking since noon"
    hard_exact:
      question: "What was the license plate number on the sedan parked outside the warehouse?"
      needle: "Rain pooled on the hood of something dark and low-slung across the street — a Buick, maybe, or an Olds. The plates were muddied but she caught the last three: 7-4-9."
      answer: "Partial plate ending in 749, on a dark sedan (possibly Buick or Oldsmobile)"
    hard_synonym:
      question: "What identifying marks were on the vehicle near the warehouse?"
      needle: "Rain pooled on the hood of something dark and low-slung across the street — a Buick, maybe, or an Olds. The plates were muddied but she caught the last three: 7-4-9."
      answer: "Dark sedan (Buick or Oldsmobile) with muddied plates, last three digits 749"
    hard_contextual:
      question: "What vehicle was connected to the suspect's movements on the night of the crime?"
      needle: "Rain pooled on the hood of something dark and low-slung across the street — a Buick, maybe, or an Olds. The plates were muddied but she caught the last three: 7-4-9."
      answer: "A dark sedan, possibly a Buick or Oldsmobile, with a plate ending in 749"

rubric_criteria:
  - criterion: "correctness"
    weight: 1.0
  - criterion: "completeness"
    weight: 0.5

grid:
  content_profile: [python_repo, noir_fiction]
  corpus_token_count: [20000, 40000, 160000, 800000]
  discriminability: [easy, hard]
  reference_clarity: [exact, synonym, contextual]

runner:
  n_repeats: 3
  agent_model: "claude-sonnet-4-6"
  max_tokens: 100000
  allowed_tools: ["Read", "Glob", "Grep", "Bash"]
```

### multi_chain.yaml

```yaml
schema_version: "2.0"
experiment_type: "multi_chain"

payload:
  item_type: "cross_reference"

question_examples:
  python_repo:
    easy_exact:
      question: "Starting from DATABASE_URL in config.md, follow the connection string references. What is the final hostname?"
      chain:
        - needle: "DATABASE_URL = get_connection('primary_pool')"
          file_context: "config/database.md"
        - needle: "def get_connection(pool_name):\n    return POOLS[pool_name].dsn"
          file_context: "db/pools.md"
        - needle: "POOLS = {'primary_pool': PoolConfig(dsn='postgres://db-prod-07.internal:5432/main')}"
          file_context: "db/settings.md"
      answer: "db-prod-07.internal"
    hard_contextual:
      question: "Starting from the main database configuration, trace how connection parameters are resolved. What host does the application ultimately connect to?"
      chain:
        - needle: "conn = build_client(env='production')"
          file_context: "app/startup.md"
        - needle: "def build_client(env):\n    profile = load_profile(f'db_{env}')\n    return connect(**profile)"
          file_context: "app/db/factory.md"
        - needle: "db_production:\n  adapter: postgresql\n  host: rds-4a8b.us-east-1.amazonaws.com\n  port: 5432"
          file_context: "config/profiles.md"
      answer: "rds-4a8b.us-east-1.amazonaws.com, resolved through the production profile loaded by build_client"
  noir_fiction:
    easy_exact:
      question: "Starting from the matchbook found in chapter_02.md, follow the trail of locations. Where does it end?"
      chain:
        - needle: "Inside his coat pocket she found a matchbook stamped 'The Red Lantern — 34th & Vine.'"
          file_context: "chapter_02.md"
        - needle: "The bartender at The Red Lantern remembered him. 'Took a card from the bulletin board — some place called Hargrove Storage, unit 19.'"
          file_context: "chapter_05.md"
        - needle: "Unit 19 was empty except for a forwarding label stuck to the wall: 'Deliver remaining to Pier 11, Warehouse C.'"
          file_context: "chapter_07.md"
      answer: "Pier 11, Warehouse C"
    hard_contextual:
      question: "Starting from the personal effects found on the body, trace where the deceased had been that evening. What was the final destination?"
      chain:
        - needle: "The coat was expensive but worn. In the breast pocket: a receipt from Delmonico's, timestamped 7:15 PM."
          file_context: "chapter_01.md"
        - needle: "The maitre d' at Delmonico's checked his book. 'Table for two. His companion left a card — the Avalon Club, members only.'"
          file_context: "chapter_04.md"
        - needle: "The Avalon's guest log showed a taxi called at 10:40 PM. Destination written in pencil: 'Pier 11, east entrance.'"
          file_context: "chapter_06.md"
      answer: "Pier 11, east entrance, reached by taxi from the Avalon Club at 10:40 PM"

grid:
  content_profile: [python_repo, noir_fiction]
  corpus_token_count: [20000, 40000, 160000, 800000]
  discriminability: [easy, hard]
  reference_clarity: [exact, synonym, contextual]
  n_items: [2, 8, 16]

rubric_criteria:
  - criterion: "correctness"
    weight: 1.0
  - criterion: "completeness"
    weight: 0.5

runner:
  n_repeats: 3
  agent_model: "claude-sonnet-4-6"
  max_tokens: 100000
  allowed_tools: ["Read", "Glob", "Grep", "Bash"]
```

### multi_reasoning.yaml

```yaml
schema_version: "2.0"
experiment_type: "multi_reasoning"

payload:
  item_type: "fact"

question_examples:
  python_repo:
    easy_exact:
      question: "What is the total maximum memory allocation across all cache layers (L1_CACHE_SIZE_MB, L2_CACHE_SIZE_MB, L3_CACHE_SIZE_MB)?"
      items:
        - needle: "L1_CACHE_SIZE_MB = 512"
          file_context: "cache/l1_config.md"
        - needle: "L2_CACHE_SIZE_MB = 2048"
          file_context: "cache/l2_config.md"
        - needle: "L3_CACHE_SIZE_MB = 8192"
          file_context: "cache/l3_config.md"
      answer: "10,752 MB (512 + 2048 + 8192)"
    hard_contextual:
      question: "If the application processes requests at peak throughput, how many database connections would be needed simultaneously?"
      items:
        - needle: "WORKER_THREADS = 8"
          file_context: "server/config.md"
        - needle: "# each worker holds a connection for the duration of the transaction\nCONNS_PER_WORKER = 3  # read, write, advisory lock"
          file_context: "db/pool.md"
        - needle: "REPLICA_FACTOR = 2  # each write is mirrored"
          file_context: "db/replication.md"
      answer: "48 connections (8 workers x 3 connections x 2 replication factor)"
  noir_fiction:
    easy_exact:
      question: "Three witnesses each saw part of the suspect's outfit. What was the full description?"
      items:
        - needle: "'He wore a charcoal overcoat,' said the doorman. 'Double-breasted, expensive.'"
          file_context: "chapter_03.md"
        - needle: "The cigarette girl remembered the hat. 'A grey fedora, tilted low. Hid most of his face.'"
          file_context: "chapter_04.md"
        - needle: "'Red shoes,' the shoeshine boy said without hesitation. 'Oxfords. Nobody wears red oxfords.'"
          file_context: "chapter_06.md"
      answer: "Charcoal double-breasted overcoat, grey fedora tilted low, red Oxford shoes"
    hard_contextual:
      question: "Based on the evidence collected across the investigation, what time window was the crime committed in?"
      items:
        - needle: "The coroner's preliminary note read: 'Rigor onset suggests no more than four hours prior to discovery at 11pm.'"
          file_context: "chapter_03.md"
        - needle: "Mrs. Chen was firm. 'I heard him playing that awful trumpet past nine. Stopped around quarter to ten.'"
          file_context: "chapter_08.md"
        - needle: "The receipt from the pharmacy was timestamped 8:47 PM. The pharmacist confirmed he looked 'perfectly alive and in a hurry.'"
          file_context: "chapter_05.md"
      answer: "Between 9:45 PM and 11:00 PM — alive at 8:47 (pharmacy receipt), heard until ~9:45 (neighbor), dead no more than 4 hours before 11 PM discovery"

grid:
  content_profile: [python_repo, noir_fiction]
  corpus_token_count: [20000, 40000, 160000, 800000]
  discriminability: [easy, hard]
  reference_clarity: [exact, synonym, contextual]
  n_items: [2, 8, 16]

rubric_criteria:
  - criterion: "correctness"
    weight: 1.0
  - criterion: "completeness"
    weight: 0.5

runner:
  n_repeats: 3
  agent_model: "claude-sonnet-4-6"
  max_tokens: 100000
  allowed_tools: ["Read", "Glob", "Grep", "Bash"]
```

---

## Batch File Format (v2)

Batch files reference experiment types, not individual specs. The grid expansion produces the parametrisations.

```yaml
batch_name: "full-sweep-v1"
max_parallel: 4
retry_failed: true
judge_model: "claude-sonnet-4-6"

experiments:
  - "single_needle"
  - "multi_chain"
  - "multi_reasoning"
```

Optional: run a subset of the grid by specifying parameter filters:

```yaml
batch_name: "quick-smoke-test"
max_parallel: 2
retry_failed: true
judge_model: "claude-sonnet-4-6"

experiments:
  - experiment_type: "single_needle"
    filter:
      content_profile: [python_repo]
      corpus_token_count: [20000]
      discriminability: [easy]
      reference_clarity: [exact]
```

---

## Content Profiles

### python_repo

Background files are `.md` files containing realistic Python source code, configuration, documentation, and project files for a medium-sized web application. The existing `PythonRepoProfile` is adapted to produce `.md` files and to work with the new background pool architecture.

### noir_fiction

Background files are `.md` files containing chapters, scenes, and supplementary materials (case notes, evidence logs, witness statements) for a noir detective novel. Narrative prose with realistic dialogue, atmospheric descriptions, and a cast of characters.

Both profiles generate `.md` files. File sizes are non-uniform — natural variance is expected and desired.

### Profile Interface

The content profile interface changes to support the new pool-based generation:

```python
class ContentProfile(ABC):
    @abstractmethod
    def pool_generation_brief(self, target_token_count: int) -> str:
        """Return the system prompt for the background pool generation agent."""
        ...

    @abstractmethod
    def skeleton(self, target_token_count: int) -> dict:
        """Return a folder/file skeleton the pool generation agent should follow."""
        ...
```

The profile no longer generates individual file prompts. Instead, it produces a brief and skeleton for the pool generation agent, which creates all files autonomously.

---

## Generation Pipeline

### Phase 1: Background Pool Generation (once per content profile)

**Input:** Content profile name
**Output:** `workspace/runner/pools/{profile_name}/` — ~1M tokens of `.md` files
**Model:** Haiku via Agent SDK
**Idempotent:** Skips if pool directory already exists and meets token target

The generator:
1. Loads the content profile
2. Gets the pool generation brief and skeleton from the profile
3. Launches a Haiku agent via Agent SDK with:
   - System prompt: the profile's brief
   - Tools: `Write` (to create files)
   - Working directory: `workspace/runner/pools/{profile_name}/`
   - The skeleton as initial context
4. The agent creates files autonomously, following the skeleton structure
5. Generation is done in stages to stay within context limits — the agent is relaunched with progress context until the token budget is met

Since 1M tokens is large, the pool generation happens in batches:
- The skeleton defines logical sections (e.g., chapters 1-5, chapters 6-10 for noir; src/auth, src/api, etc. for python_repo)
- Each batch launches a fresh agent to generate one section
- A coordinator tracks cumulative tokens and stops when the budget is reached

### Phase 2: Corpus Assembly (once per parametrisation, no LLM)

**Input:** Background pool + experiment parametrisation
**Output:** `workspace/runner/corpora/{parametrisation_id}/` — sampled subset of pool files
**Deterministic:** Uses parametrisation ID as random seed

The assembler:
1. Lists all files in the background pool
2. Shuffles using a deterministic seed derived from the parametrisation ID
3. Selects files until `corpus_token_count` budget is reached (token count estimated by character count / 4)
4. Copies selected files into the corpus directory

### Phase 3: Payload Insertion (once per parametrisation, agentic)

**Input:** Assembled corpus + experiment definition + parametrisation params
**Output:** Modified corpus files + answer key at `workspace/judge/answer_keys/{parametrisation_id}.yaml`
**Model:** Sonnet via Agent SDK

The generator launches a Sonnet agent with:
- **Working directory:** `workspace/runner/corpora/{parametrisation_id}/`
- **Tools:** `Read`, `Edit`, `Write`, `Glob`, `Grep`
- **Additional write access:** The agent is granted write access to `workspace/judge/answer_keys/{parametrisation_id}.yaml` (outside its working directory) for the answer key. This is the only cross-boundary write.
- **System prompt:** Contains:
  - Experiment type and what it demands
  - Number of items to insert (`n_items` or 1 for single_needle)
  - Discriminability rubric (unified, profile-agnostic)
  - Reference clarity level and what it means
  - Few-shot examples from the experiment YAML (filtered to matching `{discriminability}_{reference_clarity}`)
  - Instructions to insert needles, generate the question, generate the expected answer, and write the answer key

The agent manages its own context — it reads files as needed, chooses insertion points, and maintains coherence across multi-item insertions.

**For multi_chain:** The agent inserts links sequentially, ensuring each link references the previous. The question includes the entry point.

**For multi_reasoning:** The agent inserts items independently, ensuring they are distinct and non-overlapping. The question requires combining all items.

**Answer key output format:**

```yaml
parametrisation_id: "single_needle__python_repo__20k__easy__exact"
experiment_type: "single_needle"
generated_at: "2026-04-03T10:30:00Z"
parameters:
  content_profile: "python_repo"
  corpus_token_count: 20000
  discriminability: "easy"
  reference_clarity: "exact"
items:
  - item_id: "target_001"
    inserted_text: "MAX_POOL_SIZE = 25"
    file_path: "config/database.md"
    line_range: [47, 47]
    context_summary: "Inserted as a module-level constant in database config"
expected_answers:
  question: "What is the value of MAX_POOL_SIZE defined in this codebase?"
  correctness: "MAX_POOL_SIZE is 25, defined in config/database.md"
  completeness: "The value is 25"
rubric_criteria:
  - criterion: "correctness"
    weight: 1.0
  - criterion: "completeness"
    weight: 0.5
```

### Leakage Prevention

Unchanged from v1:
- No symlinks or shared references between `workspace/runner/corpora/` and `workspace/judge/answer_keys/`
- No metadata in corpus files pointing back to the answer key or generation process
- Generated files are scrubbed of generation artifacts
- The payload insertion agent's working directory is the corpus directory; the answer key write path is passed as a parameter outside the sandbox

### Corpus Fixity Across Repeats

The corpus is generated once per parametrisation. All `n_repeats` runs use the identical corpus. Repeats measure agent behavioral reliability to the same stimulus, not variance from different corpora.

---

## Workspace Directory Layout (v2)

```
workspace/
├── runner/
│   ├── pools/{profile_name}/          # 1M token background pools (NEW)
│   │   ├── *.md
│   │   └── ...
│   ├── corpora/{parametrisation_id}/  # Sampled + payload-injected corpora
│   │   ├── *.md
│   │   └── ...
│   └── runs/{batch_name}/{parametrisation_id}/{run_id}/
│       ├── state.yaml
│       ├── response.json
│       └── session.jsonl
├── judge/
│   ├── answer_keys/{parametrisation_id}.yaml
│   └── judgements/{batch_name}/{parametrisation_id}/
│       └── {run_id}.yaml
└── analysis/{batch_name}/
    ├── summary.csv
    ├── tables/
    ├── figures/
    └── report.html
```

Key change: `pools/` directory added under `runner/`. Corpora are now keyed by `parametrisation_id` instead of `experiment_id`.

---

## Schema Changes from v1

### Removed
- `ExperimentSpec.experiment_id` — replaced by deterministic parametrisation IDs
- `CorpusSpec.target_file_count` — file count is a consequence of token sampling, not a target
- `CorpusSpec.folder_depth` — controlled by the background pool, not per-experiment
- `CorpusSpec.folder_distribution` — same
- `CorpusSpec.generation_model` — pool generation uses Haiku; not configurable per-experiment
- `CorpusSpec.red_herring_density` — collapsed into `discriminability`
- `PayloadItem.camouflage` — collapsed into `discriminability`
- `PayloadItem.placement` — the insertion agent chooses placement autonomously
- `PayloadItem.depends_on` — dependency structure is implicit in experiment type (chain = sequential, reasoning = independent)
- `PayloadSpec.insertion_model` — always Sonnet via Agent SDK
- `PayloadSpec.red_herring_hint` — collapsed into discriminability; the insertion agent handles this
- `llm_client.py` — direct Anthropic API calls replaced by Agent SDK

### Added
- `ExperimentTemplate` — top-level schema for v2 experiment YAML files
- `ExperimentTemplate.grid` — inline parameter grid
- `ExperimentTemplate.question_examples` — few-shot examples keyed by profile and difficulty
- `discriminability` parameter (replaces camouflage + red_herring_density)
- `reference_clarity` parameter
- `n_items` parameter (multi types only)
- Background pool generation via Agent SDK (Haiku)
- Payload insertion via Agent SDK (Sonnet) — agentic, not loop-managed
- Corpus assembly phase (pure code, no LLM)

### Modified
- `ContentProfile` interface — now provides `pool_generation_brief()` and `skeleton()` instead of `generate_folder_structure()` and `generate_file_prompt()`
- `AnswerKey` — keyed by `parametrisation_id` instead of `experiment_id`, includes `parameters` dict
- `BatchConfig` — references experiment types + optional grid filters instead of individual experiment IDs
- All generator code — restructured around the three-phase pipeline

---

## Generator Architecture (Agent SDK)

All LLM calls use the Claude Agent SDK (same mechanism as the experiment runner). No direct Anthropic API client.

### Background Pool Generation Agent

```python
# Pseudocode for pool generation
for section in profile.skeleton(target_token_count=1_000_000):
    agent = AgentSession(
        model="claude-haiku-4-5-20251001",
        system_prompt=profile.pool_generation_brief(section),
        working_directory=pool_dir,
        allowed_tools=["Write"],
    )
    await agent.run()
    # Check cumulative tokens, stop if budget met
```

### Payload Insertion Agent

```python
# Pseudocode for payload insertion
system_prompt = f"""You are inserting {n_items} needle(s) into a corpus for a retrieval experiment.

Experiment type: {experiment_type}
Discriminability: {discriminability}
Reference clarity: {reference_clarity}

## Discriminability Rubric

| Level | Retrieval method | Cognitive demand |
|---|---|---|
| easy | Findable by exact string search on terms in the question | Locate and report |
| hard | Embedded in surrounding context; requires reading and comprehending surrounding content to identify | Comprehend context, extract implicit information |

## Reference Clarity

| Level | Definition |
|---|---|
| exact | Question uses the same identifier/name as the needle |
| synonym | Question uses a different term for the same concept |
| contextual | Question describes the concept indirectly; requires domain understanding to connect |

## Examples
{filtered_examples}

## Instructions
1. Browse the corpus to understand its structure and content style
2. Choose {n_items} file(s) to insert needles into
3. Read each target file to understand the surrounding context
4. Insert each needle so it reads naturally within the file
5. Generate a question and expected answer
6. Write the answer key to {answer_key_path}

The answer key must be valid YAML with this structure:
{answer_key_schema}
"""

agent = AgentSession(
    model="claude-sonnet-4-6",
    system_prompt=system_prompt,
    working_directory=corpus_dir,
    allowed_tools=["Read", "Edit", "Write", "Glob", "Grep"],
)
await agent.run()
```

---

## CLI Changes

```bash
# Generate background pool for a profile (if not exists)
agent-retrieval generate-pool python_repo
agent-retrieval generate-pool noir_fiction

# Generate all corpora for an experiment type (expand grid, assemble, insert)
agent-retrieval generate experiments/single_needle.yaml

# Run all parametrisations in a batch
agent-retrieval run batches/full-sweep-v1.yaml

# Judge and analyze (unchanged)
agent-retrieval judge batches/full-sweep-v1.yaml
agent-retrieval analyze batches/full-sweep-v1.yaml
```

The `generate` command now:
1. Reads the experiment YAML
2. Expands the grid into parametrisations
3. For each parametrisation: checks pool exists, assembles corpus, runs payload insertion
4. Skips parametrisations that already have a corpus + answer key (idempotent)

---

## Analysis Changes

The analysis module's slicing dimensions update to match the new parameters:

| Metric | Sliced by |
|---|---|
| Accuracy (weighted score) | corpus_token_count, discriminability, reference_clarity, n_items, content_profile, experiment_type |
| Completeness score | same |
| Context tokens consumed | same |
| Tool call counts | by tool type, per experiment_type and content_profile |
| Variance across repeats | error bars / confidence intervals on all of the above |

New figures:
- `accuracy_vs_corpus_size.png` — per experiment type, per profile
- `accuracy_vs_n_items.png` — for multi types
- `accuracy_by_discriminability.png` — easy vs hard across types
- `accuracy_by_reference_clarity.png` — exact vs synonym vs contextual
- `profile_comparison.png` — python_repo vs noir_fiction on shared dimensions
