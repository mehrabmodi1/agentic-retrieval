from unittest.mock import AsyncMock, patch
import pytest
from agent_retrieval.cli import parse_args

class TestCLI:
    def test_generate_spec_file(self):
        args = parse_args(["generate", "specs/needle-001.yaml"])
        assert args.command == "generate"
        assert args.config_path == "specs/needle-001.yaml"

    def test_run_batch(self):
        args = parse_args(["run", "batches/test.yaml"])
        assert args.command == "run"
        assert args.config_path == "batches/test.yaml"

    def test_judge_batch(self):
        args = parse_args(["judge", "batches/test.yaml"])
        assert args.command == "judge"

    def test_judge_rejudge_flag(self):
        args = parse_args(["judge", "batches/test.yaml", "--rejudge"])
        assert args.rejudge is True

    def test_analyze_batch(self):
        args = parse_args(["analyze", "batches/test.yaml"])
        assert args.command == "analyze"
