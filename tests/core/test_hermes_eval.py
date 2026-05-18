"""Tests for real Hermes Agent-backed skill evaluation helpers."""

from types import SimpleNamespace

import pytest

from evolution.core.hermes_eval import (
    HermesSkillEvalCase,
    _load_hermes_symbols,
    build_skill_system_prompt,
    run_skill_eval,
)


class FakeAgent:
    last_init = None
    last_user_message = None

    def __init__(self, **kwargs):
        FakeAgent.last_init = kwargs

    def run_conversation(self, user_message: str, conversation_history=None):
        FakeAgent.last_user_message = user_message
        return {"final_response": "done", "messages": []}


class InlineOnlyAgent(FakeAgent):
    pass


@pytest.fixture(autouse=True)
def reset_fake_agent():
    FakeAgent.last_init = None
    FakeAgent.last_user_message = None


@pytest.fixture
def fake_loader(monkeypatch):
    def _fake_loader(hermes_repo=None):
        def _build_preloaded_skills_prompt(skill_identifiers, task_id=None):
            assert skill_identifiers == ["github-code-review"]
            return ("[SYSTEM: skill prompt]", ["github-code-review"], [])

        return SimpleNamespace(
            AIAgent=FakeAgent,
            build_preloaded_skills_prompt=_build_preloaded_skills_prompt,
        )

    monkeypatch.setattr("evolution.core.hermes_eval._load_hermes_symbols", _fake_loader)


def test_build_skill_system_prompt_can_inline_alternate_skill_body(monkeypatch):
    def _fake_loader(hermes_repo=None):
        def _build_preloaded_skills_prompt(skill_identifiers, task_id=None):
            return ("[SYSTEM: original skill prompt]", ["github-code-review"], [])

        return SimpleNamespace(
            AIAgent=InlineOnlyAgent,
            build_preloaded_skills_prompt=_build_preloaded_skills_prompt,
        )

    monkeypatch.setattr("evolution.core.hermes_eval._load_hermes_symbols", _fake_loader)

    prompt, loaded = build_skill_system_prompt(
        "github-code-review",
        skill_body_override="# EVOLVED\nUse stricter review criteria.",
    )

    assert loaded == ["github-code-review"]
    assert "original skill prompt" not in prompt
    assert "EVOLVED" in prompt
    assert "stricter review criteria" in prompt


def test_load_hermes_symbols_imports_from_repo(monkeypatch, tmp_path):
    repo = tmp_path / "hermes-agent"
    (repo / "agent").mkdir(parents=True)
    (repo / "run_agent.py").write_text(
        'class AIAgent:\n'
        '    pass\n'
    )
    (repo / "agent" / "skill_commands.py").write_text(
        'def build_preloaded_skills_prompt(skill_identifiers, task_id=None):\n'
        '    return "prompt", ["loaded"], []\n'
    )

    symbols = _load_hermes_symbols(repo)

    assert symbols.AIAgent.__name__ == "AIAgent"
    prompt, loaded, missing = symbols.build_preloaded_skills_prompt(["x"])
    assert prompt == "prompt"
    assert loaded == ["loaded"]
    assert missing == []


def test_load_hermes_symbols_tolerates_missing_fire_dependency(monkeypatch, tmp_path):
    repo = tmp_path / "hermes-agent"
    (repo / "agent").mkdir(parents=True)
    (repo / "run_agent.py").write_text(
        'import fire\n'
        'class AIAgent:\n'
        '    pass\n'
    )
    (repo / "agent" / "skill_commands.py").write_text(
        'def build_preloaded_skills_prompt(skill_identifiers, task_id=None):\n'
        '    return "prompt", ["loaded"], []\n'
    )
    monkeypatch.delitem(__import__("sys").modules, "fire", raising=False)

    symbols = _load_hermes_symbols(repo)

    assert symbols.AIAgent.__name__ == "AIAgent"



def test_load_hermes_symbols_adds_repo_to_sys_path_for_local_imports(tmp_path):
    repo = tmp_path / "hermes-agent"
    (repo / "agent").mkdir(parents=True)
    (repo / "hermes_constants.py").write_text('VALUE = "ok"\n')
    (repo / "run_agent.py").write_text(
        'from hermes_constants import VALUE\n'
        'class AIAgent:\n'
        '    value = VALUE\n'
    )
    (repo / "agent" / "skill_commands.py").write_text(
        'def build_preloaded_skills_prompt(skill_identifiers, task_id=None):\n'
        '    return "prompt", ["loaded"], []\n'
    )

    symbols = _load_hermes_symbols(repo)

    assert symbols.AIAgent.value == "ok"


def test_build_skill_system_prompt_returns_loaded_skill_prompt(fake_loader):
    prompt, loaded = build_skill_system_prompt("github-code-review", hermes_repo="/tmp/hermes")

    assert prompt == "[SYSTEM: skill prompt]"
    assert loaded == ["github-code-review"]


def test_build_skill_system_prompt_raises_for_missing_skill(monkeypatch):
    def _fake_loader(hermes_repo=None):
        def _build_preloaded_skills_prompt(skill_identifiers, task_id=None):
            return ("", [], ["missing-skill"])

        return SimpleNamespace(
            AIAgent=FakeAgent,
            build_preloaded_skills_prompt=_build_preloaded_skills_prompt,
        )

    monkeypatch.setattr("evolution.core.hermes_eval._load_hermes_symbols", _fake_loader)

    with pytest.raises(ValueError, match="missing-skill"):
        build_skill_system_prompt("github-code-review")


def test_run_skill_eval_uses_real_agent_shape_and_preloaded_skill_prompt(fake_loader):
    case = HermesSkillEvalCase(
        skill_name="github-code-review",
        task_input="Review this diff for security issues.",
        system_prompt="[SYSTEM: custom evaluator instructions]",
    )

    result = run_skill_eval(case, model="openai/gpt-4.1-mini", hermes_repo="/tmp/hermes")

    assert result.final_response == "done"
    assert result.loaded_skills == ["github-code-review"]
    assert "custom evaluator instructions" in result.effective_system_prompt
    assert "skill prompt" in result.effective_system_prompt
    assert FakeAgent.last_user_message == "Review this diff for security issues."
    assert FakeAgent.last_init["model"] == "openai/gpt-4.1-mini"
    assert FakeAgent.last_init["quiet_mode"] is True
    assert FakeAgent.last_init["skip_context_files"] is True
    assert FakeAgent.last_init["skip_memory"] is True
    assert FakeAgent.last_init["ephemeral_system_prompt"] == result.effective_system_prompt


def test_run_skill_eval_accepts_string_agent_result(fake_loader, monkeypatch):
    class StringAgent(FakeAgent):
        def run_conversation(self, user_message: str, conversation_history=None):
            return "plain string response"

    def _fake_loader(hermes_repo=None):
        def _build_preloaded_skills_prompt(skill_identifiers, task_id=None):
            return ("[SYSTEM: skill prompt]", ["github-code-review"], [])

        return SimpleNamespace(
            AIAgent=StringAgent,
            build_preloaded_skills_prompt=_build_preloaded_skills_prompt,
        )

    monkeypatch.setattr("evolution.core.hermes_eval._load_hermes_symbols", _fake_loader)

    result = run_skill_eval(
        HermesSkillEvalCase(
            skill_name="github-code-review",
            task_input="Review this diff.",
        )
    )

    assert result.final_response == "plain string response"
    assert result.raw_result == "plain string response"
