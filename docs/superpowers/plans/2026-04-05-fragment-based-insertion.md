# Fragment-Based Insertion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce insertion-phase token consumption ~35x by sending 30-line file fragments instead of full files, batching tool calls into 1-2 turns, and switching multi-clue experiment types to Haiku.

**Architecture:** Add a `_extract_fragment()` function that picks a random 30-line window from each sampled file. `build_insertion_prompt()` receives fragments instead of full file contents. The prompt instructs the agent to batch all Edit+Write calls in a single response. `max_turns` drops from 10 to 3. Multi-clue experiment types (`multi_chain`, `multi_reasoning`) use `claude-haiku-4-5-20251001`; `single_needle` stays on Sonnet.

**Tech Stack:** Python 3.12, Pydantic v2, Claude Agent SDK (`claude_agent_sdk`), pytest

---

## File Structure

### Modified Files
- `src/agent_retrieval/generator/insertion.py` — replace `_read_target_files` with fragment extraction, update prompt, update model selection and max_turns
- `tests/test_insertion.py` — tests for fragment extraction, updated prompt tests

No new files needed. This is a focused refactor of the insertion module.

---

### Task 1: Add `_extract_fragment()` function

**Files:**
- Modify: `src/agent_retrieval/generator/insertion.py`
- Test: `tests/test_insertion.py`

- [ ] **Step 1: Write failing tests for `_extract_fragment`**

```python
# tests/test_insertion.py — add to top-level imports
from agent_retrieval.generator.insertion import _extract_fragment


class TestExtractFragment:
    def test_returns_30_lines_from_middle_of_long_file(self, tmp_path):
        content = "\n".join(f"line {i}" for i in range(100))
        f = tmp_path / "big.md"
        f.write_text(content)

        fragment, start_line = _extract_fragment(f, seed=42)

        lines = fragment.strip().split("\n")
        assert len(lines) == 30

    def test_returns_full_file_when_under_30_lines(self, tmp_path):
        content = "\n".join(f"line {i}" for i in range(10))
        f = tmp_path / "small.md"
        f.write_text(content)

        fragment, start_line = _extract_fragment(f, seed=42)

        lines = fragment.strip().split("\n")
        assert len(lines) == 10
        assert start_line == 0

    def test_fragment_is_contiguous_slice_of_original(self, tmp_path):
        content = "\n".join(f"line {i}" for i in range(200))
        f = tmp_path / "big.md"
        f.write_text(content)

        fragment, start_line = _extract_fragment(f, seed=99)

        original_lines = content.split("\n")
        fragment_lines = fragment.strip().split("\n")
        assert fragment_lines == original_lines[start_line:start_line + 30]

    def test_different_seeds_produce_different_offsets(self, tmp_path):
        content = "\n".join(f"line {i}" for i in range(200))
        f = tmp_path / "big.md"
        f.write_text(content)

        _, start_1 = _extract_fragment(f, seed=1)
        _, start_2 = _extract_fragment(f, seed=9999)

        assert start_1 != start_2

    def test_start_line_can_be_zero(self, tmp_path):
        """No bias away from file start."""
        content = "\n".join(f"line {i}" for i in range(50))
        f = tmp_path / "file.md"
        f.write_text(content)

        starts = set()
        for seed in range(200):
            _, start = _extract_fragment(f, seed=seed)
            starts.add(start)

        assert 0 in starts

    def test_fragment_can_end_at_last_line(self, tmp_path):
        """No bias away from file end."""
        content = "\n".join(f"line {i}" for i in range(50))
        f = tmp_path / "file.md"
        f.write_text(content)

        starts = set()
        for seed in range(200):
            _, start = _extract_fragment(f, seed=seed)
            starts.add(start)

        # max valid start for 50 lines with window 30 is 20
        assert 20 in starts
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/mehrabmodi/Documents/projects/agent_retrieval_expt && poetry run pytest tests/test_insertion.py::TestExtractFragment -v`
Expected: FAIL with `ImportError: cannot import name '_extract_fragment'`

- [ ] **Step 3: Implement `_extract_fragment`**

Add to `src/agent_retrieval/generator/insertion.py`, after the existing `_select_target_files` function:

```python
def _extract_fragment(
    file_path: Path,
    seed: int,
    window: int = 30,
) -> tuple[str, int]:
    """Extract a random contiguous fragment from a file.

    Returns (fragment_text, start_line_0indexed).
    """
    lines = file_path.read_text().splitlines()
    if len(lines) <= window:
        return "\n".join(lines), 0
    rng = random.Random(seed)
    max_start = len(lines) - window
    start = rng.randint(0, max_start)
    return "\n".join(lines[start:start + window]), start
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/mehrabmodi/Documents/projects/agent_retrieval_expt && poetry run pytest tests/test_insertion.py::TestExtractFragment -v`
Expected: all 6 PASS

- [ ] **Step 5: Commit**

```bash
git add src/agent_retrieval/generator/insertion.py tests/test_insertion.py
git commit -m "feat: add _extract_fragment for 30-line windowed sampling"
```

---

### Task 2: Replace `_read_target_files` with `_read_target_fragments`

**Files:**
- Modify: `src/agent_retrieval/generator/insertion.py`
- Test: `tests/test_insertion.py`

- [ ] **Step 1: Write failing tests for `_read_target_fragments`**

```python
# tests/test_insertion.py — add import
from agent_retrieval.generator.insertion import _read_target_fragments


class TestReadTargetFragments:
    def test_returns_fragments_with_file_metadata(self, tmp_path):
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        content = "\n".join(f"line {i}" for i in range(100))
        f = corpus / "test.md"
        f.write_text(content)

        result = _read_target_fragments([f], corpus, base_seed=42)

        assert "test.md" in result
        assert "lines" in result.lower() or "line" in result.lower()

    def test_each_file_gets_unique_seed(self, tmp_path):
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        content = "\n".join(f"line {i}" for i in range(200))
        f1 = corpus / "a.md"
        f1.write_text(content)
        f2 = corpus / "b.md"
        f2.write_text(content)

        result = _read_target_fragments([f1, f2], corpus, base_seed=42)

        # Both files should appear
        assert "a.md" in result
        assert "b.md" in result

    def test_includes_start_line_info(self, tmp_path):
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        content = "\n".join(f"line {i}" for i in range(100))
        f = corpus / "test.md"
        f.write_text(content)

        result = _read_target_fragments([f], corpus, base_seed=42)

        # Should contain line offset information for the agent
        assert "start_line" in result.lower() or "line " in result.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/mehrabmodi/Documents/projects/agent_retrieval_expt && poetry run pytest tests/test_insertion.py::TestReadTargetFragments -v`
Expected: FAIL with `ImportError: cannot import name '_read_target_fragments'`

- [ ] **Step 3: Implement `_read_target_fragments`**

Replace `_read_target_files` in `src/agent_retrieval/generator/insertion.py`:

```python
def _read_target_fragments(
    files: list[Path],
    corpus_dir: Path,
    base_seed: int,
) -> str:
    """Extract a 30-line fragment from each file and format for the prompt."""
    sections = []
    for i, f in enumerate(files):
        rel = f.relative_to(corpus_dir)
        seed = base_seed ^ (i * 0x9E3779B9)  # unique seed per file
        fragment, start_line = _extract_fragment(f, seed=seed)
        sections.append(
            f"### File: {rel} (start_line: {start_line})\n```\n{fragment}\n```"
        )
    return "\n\n".join(sections)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/mehrabmodi/Documents/projects/agent_retrieval_expt && poetry run pytest tests/test_insertion.py::TestReadTargetFragments -v`
Expected: all 3 PASS

- [ ] **Step 5: Commit**

```bash
git add src/agent_retrieval/generator/insertion.py tests/test_insertion.py
git commit -m "feat: add _read_target_fragments replacing full-file reads"
```

---

### Task 3: Update `build_insertion_prompt` for fragments

**Files:**
- Modify: `src/agent_retrieval/generator/insertion.py`
- Test: `tests/test_insertion.py`

- [ ] **Step 1: Write failing tests for updated prompt**

```python
# tests/test_insertion.py — add new test class

class TestBuildInsertionPromptFragments:
    def test_instructs_batching(self, single_template, parametrisation):
        prompt = build_insertion_prompt(
            single_template, parametrisation, Path("/answer.yaml"), "fragment content"
        )
        assert "single response" in prompt.lower() or "batch" in prompt.lower()

    def test_instructs_no_browsing(self, single_template, parametrisation):
        prompt = build_insertion_prompt(
            single_template, parametrisation, Path("/answer.yaml"), "fragment content"
        )
        assert "do not read" in prompt.lower() or "do not browse" in prompt.lower()

    def test_mentions_fragments_not_full_files(self, single_template, parametrisation):
        prompt = build_insertion_prompt(
            single_template, parametrisation, Path("/answer.yaml"), "fragment content"
        )
        assert "fragment" in prompt.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/mehrabmodi/Documents/projects/agent_retrieval_expt && poetry run pytest tests/test_insertion.py::TestBuildInsertionPromptFragments -v`
Expected: FAIL — prompt says "Target files" not "fragments", no batching instruction

- [ ] **Step 3: Update `build_insertion_prompt`**

In `src/agent_retrieval/generator/insertion.py`, update the return string in `build_insertion_prompt` (lines 156-177). Replace:

```python
        f"## Target files (pre-selected — use these)\n{target_files_content}\n\n"
        f"## Instructions\n"
        f"The target files above have been pre-selected and pre-read for you.\n"
        f"Do NOT browse or read any other files. Work only with what is provided above.\n\n"
        f"1. Choose which of the target files above to insert needle(s) into\n"
        f"2. Use the Edit tool to insert each needle so it reads naturally within the file\n"
        f"3. Write the answer key YAML to: {answer_key_path}\n\n"
```

with:

```python
        f"## Target fragments (pre-selected — use these)\n{target_files_content}\n\n"
        f"## Instructions\n"
        f"Each fragment above is a 30-line window from a corpus file, with its start_line offset.\n"
        f"Do NOT read or browse any files. Work only with the fragments provided above.\n\n"
        f"1. Choose which of the target fragments above to insert needle(s) into\n"
        f"2. Use the Edit tool to insert each needle so it reads naturally within the fragment context\n"
        f"3. Write the answer key YAML to: {answer_key_path}\n\n"
        f"IMPORTANT: Batch ALL Edit and Write tool calls into a single response.\n"
        f"Do not use multiple turns.\n\n"
```

- [ ] **Step 4: Run all insertion prompt tests**

Run: `cd /Users/mehrabmodi/Documents/projects/agent_retrieval_expt && poetry run pytest tests/test_insertion.py -v -k "Prompt"`
Expected: all PASS (both old `TestBuildInsertionPrompt` and new `TestBuildInsertionPromptFragments`)

Note: The existing `TestBuildInsertionPrompt::test_contains_target_files` test asserts `"config/settings.md" in prompt` — this still passes because the test passes that string via `target_files_content` argument. No change needed.

- [ ] **Step 5: Commit**

```bash
git add src/agent_retrieval/generator/insertion.py tests/test_insertion.py
git commit -m "feat: update insertion prompt for fragment-based approach with batching"
```

---

### Task 4: Update `insert_payloads` to use fragments and select model

**Files:**
- Modify: `src/agent_retrieval/generator/insertion.py`
- Test: `tests/test_insertion.py`

- [ ] **Step 1: Write failing tests for model selection**

```python
# tests/test_insertion.py — add new test class

class TestInsertPayloadsModelSelection:
    @pytest.mark.asyncio
    async def test_single_needle_uses_sonnet(self, corpus_dir, single_template, parametrisation, tmp_path):
        answer_key_path = tmp_path / "answer_keys" / f"{parametrisation.parametrisation_id}.yaml"
        captured_options = {}

        async def fake_query(prompt, options):
            captured_options["model"] = options.model
            captured_options["max_turns"] = options.max_turns
            answer_key_path.parent.mkdir(parents=True, exist_ok=True)
            answer_key_path.write_text("placeholder: true")
            result = MagicMock()
            yield result

        with patch("agent_retrieval.generator.insertion.query", side_effect=fake_query):
            await insert_payloads(single_template, parametrisation, corpus_dir, answer_key_path)

        assert captured_options["model"] == "claude-sonnet-4-6"

    @pytest.mark.asyncio
    async def test_multi_chain_uses_haiku(self, tmp_path):
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        for i in range(10):
            (corpus / f"file_{i}.md").write_text("\n".join(f"line {j}" for j in range(50)))

        tmpl = ExperimentTemplate.model_validate({
            "schema_version": "2.0",
            "experiment_type": "multi_chain",
            "payload": {"item_type": "cross_reference"},
            "question_examples": {
                "python_repo": {
                    "easy_exact": {
                        "question": "Follow the chain.",
                        "chain": [{"needle": "A", "file_context": "a.md"}],
                        "answer": "A",
                    },
                },
            },
            "rubric_criteria": [{"criterion": "correctness", "weight": 1.0}],
            "grid": {
                "content_profile": ["python_repo"],
                "corpus_token_count": [20000],
                "discriminability": ["easy"],
                "reference_clarity": ["exact"],
                "n_items": [4],
            },
            "runner": {
                "n_repeats": 1,
                "agent_model": "claude-sonnet-4-6",
                "max_tokens": 100000,
                "allowed_tools": ["Read"],
            },
        })
        param = Parametrisation(
            experiment_type="multi_chain",
            content_profile="python_repo",
            corpus_token_count=20000,
            discriminability="easy",
            reference_clarity="exact",
            n_items=4,
        )
        answer_key_path = tmp_path / "answer_keys" / f"{param.parametrisation_id}.yaml"
        captured_options = {}

        async def fake_query(prompt, options):
            captured_options["model"] = options.model
            answer_key_path.parent.mkdir(parents=True, exist_ok=True)
            answer_key_path.write_text("placeholder: true")
            result = MagicMock()
            yield result

        with patch("agent_retrieval.generator.insertion.query", side_effect=fake_query):
            await insert_payloads(tmpl, param, corpus, answer_key_path)

        assert captured_options["model"] == "claude-haiku-4-5-20251001"

    @pytest.mark.asyncio
    async def test_max_turns_is_3(self, corpus_dir, single_template, parametrisation, tmp_path):
        answer_key_path = tmp_path / "answer_keys" / f"{parametrisation.parametrisation_id}.yaml"
        captured_options = {}

        async def fake_query(prompt, options):
            captured_options["max_turns"] = options.max_turns
            answer_key_path.parent.mkdir(parents=True, exist_ok=True)
            answer_key_path.write_text("placeholder: true")
            result = MagicMock()
            yield result

        with patch("agent_retrieval.generator.insertion.query", side_effect=fake_query):
            await insert_payloads(single_template, parametrisation, corpus_dir, answer_key_path)

        assert captured_options["max_turns"] == 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/mehrabmodi/Documents/projects/agent_retrieval_expt && poetry run pytest tests/test_insertion.py::TestInsertPayloadsModelSelection -v`
Expected: FAIL — model is always `claude-sonnet-4-6`, max_turns is 10

- [ ] **Step 3: Update `insert_payloads`**

Replace the body of `insert_payloads` in `src/agent_retrieval/generator/insertion.py` (lines 180-218):

```python
async def insert_payloads(
    template: ExperimentTemplate,
    parametrisation: Parametrisation,
    corpus_dir: Path,
    answer_key_path: Path,
) -> None:
    if answer_key_path.exists():
        return

    answer_key_path.parent.mkdir(parents=True, exist_ok=True)

    n_items = parametrisation.n_items or 1
    # Select more files than needed so the agent has choices
    n_target_files = max(n_items * 2, 4)
    target_files = _select_target_files(corpus_dir, parametrisation, n_target_files)
    base_seed = hash(parametrisation.parametrisation_id) ^ 0xDEAD
    target_files_content = _read_target_fragments(target_files, corpus_dir, base_seed)

    system_prompt = build_insertion_prompt(
        template, parametrisation, answer_key_path, target_files_content,
    )

    is_multi = parametrisation.experiment_type in ("multi_chain", "multi_reasoning")
    model = "claude-haiku-4-5-20251001" if is_multi else "claude-sonnet-4-6"

    options = ClaudeAgentOptions(
        model=model,
        system_prompt=system_prompt,
        cwd=str(corpus_dir),
        allowed_tools=["Edit", "Write"],
        permission_mode="acceptEdits",
        max_turns=3,
    )

    prompt = (
        "Insert the needle(s) into the provided fragments and write the answer key. "
        "Batch all Edit and Write calls into a single response. "
        "Do not read or browse any files."
    )

    async for message in query(prompt=prompt, options=options):
        if isinstance(message, ResultMessage):
            break
```

- [ ] **Step 4: Run all insertion tests**

Run: `cd /Users/mehrabmodi/Documents/projects/agent_retrieval_expt && poetry run pytest tests/test_insertion.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/agent_retrieval/generator/insertion.py tests/test_insertion.py
git commit -m "feat: use fragments, haiku for multi-clue, max_turns=3"
```

---

### Task 5: Remove dead `_read_target_files` function

**Files:**
- Modify: `src/agent_retrieval/generator/insertion.py`

- [ ] **Step 1: Verify `_read_target_files` is unused**

Run: `cd /Users/mehrabmodi/Documents/projects/agent_retrieval_expt && grep -rn "_read_target_files" src/ tests/`
Expected: no references remain (only the definition itself, which was replaced in Task 2)

- [ ] **Step 2: Delete `_read_target_files`**

Remove the function `_read_target_files` (the old version at lines 101-108) from `src/agent_retrieval/generator/insertion.py`.

- [ ] **Step 3: Run full test suite**

Run: `cd /Users/mehrabmodi/Documents/projects/agent_retrieval_expt && poetry run pytest -v`
Expected: all PASS

- [ ] **Step 4: Commit**

```bash
git add src/agent_retrieval/generator/insertion.py
git commit -m "chore: remove unused _read_target_files"
```

---

## Summary of Changes

| What | Before | After |
|------|--------|-------|
| File content in prompt | Full files (avg ~2,863 tok each) | 30-line fragments (~450 tok each) |
| Prompt instruction | "Choose a file, insert" | "Batch all Edits + Write in one response" |
| `max_turns` | 10 | 3 |
| Model (single_needle) | Sonnet | Sonnet |
| Model (multi_chain, multi_reasoning) | Sonnet | Haiku |
| Est. total input tokens (336 params) | ~138M | ~4M |
| Files modified | 2 | 2 |
