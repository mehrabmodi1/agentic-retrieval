import inspect

from agent_retrieval.runner import session as session_mod
from agent_retrieval.runner import state as state_mod


def test_run_agent_session_drops_max_tokens():
    sig = inspect.signature(session_mod.run_agent_session)
    assert "max_tokens" not in sig.parameters, (
        "max_tokens was dead plumbing — ClaudeAgentOptions has no max_tokens field"
    )


def test_run_agent_session_accepts_max_turns():
    sig = inspect.signature(session_mod.run_agent_session)
    assert "max_turns" in sig.parameters, (
        "max_turns must be a runtime parameter, not hardcoded"
    )


def test_run_agent_session_accepts_allowed_tools():
    sig = inspect.signature(session_mod.run_agent_session)
    assert "allowed_tools" in sig.parameters


def test_max_turns_not_hardcoded():
    src = inspect.getsource(session_mod.run_agent_session)
    assert "max_turns=50" not in src, (
        "max_turns should come from the caller (batch config), not be hardcoded"
    )


def test_run_state_manager_accepts_max_turns_and_allowed_tools():
    sig = inspect.signature(state_mod.RunStateManager.create_pending_runs)
    assert "max_turns" in sig.parameters
    assert "allowed_tools" in sig.parameters
