"""Helpers for evaluating skills with a real Hermes Agent instance.

This bridges hermes-agent-self-evolution to the actual Hermes runtime instead of
only evaluating a DSPy skill wrapper in isolation.
"""

from __future__ import annotations

import importlib.util
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Optional

from evolution.core.config import get_hermes_agent_path


@dataclass
class HermesSkillEvalCase:
    """Single evaluation case for a skill using a real Hermes Agent."""

    skill_name: str
    task_input: str
    system_prompt: str = ""
    conversation_history: Optional[list[dict[str, Any]]] = None


@dataclass
class HermesSkillEvalResult:
    """Result of a real Hermes Agent skill evaluation run."""

    final_response: str
    loaded_skills: list[str]
    effective_system_prompt: str
    raw_result: Any


def _import_module_from_path(module_name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@contextmanager
def _temporary_hermes_import_shims(repo: Path | None = None):
    """Provide tiny stubs and import path support for dynamic Hermes imports."""
    installed: list[str] = []
    inserted_repo = False

    if repo is not None:
        repo_str = str(repo)
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)
            inserted_repo = True

    if "fire" not in sys.modules:
        fire_stub = ModuleType("fire")

        def _fire(*args, **kwargs):
            return None

        fire_stub.Fire = _fire
        sys.modules["fire"] = fire_stub
        installed.append("fire")

    try:
        yield
    finally:
        for module_name in installed:
            sys.modules.pop(module_name, None)
        if inserted_repo:
            try:
                sys.path.remove(repo_str)
            except ValueError:
                pass


def _load_hermes_symbols(hermes_repo: str | Path | None = None) -> SimpleNamespace:
    repo = Path(hermes_repo).expanduser() if hermes_repo else get_hermes_agent_path()
    repo = repo.resolve()

    with _temporary_hermes_import_shims(repo):
        run_agent_module = _import_module_from_path(
            "_self_evolution_run_agent",
            repo / "run_agent.py",
        )
        skill_commands_module = _import_module_from_path(
            "_self_evolution_skill_commands",
            repo / "agent" / "skill_commands.py",
        )

    return SimpleNamespace(
        AIAgent=run_agent_module.AIAgent,
        build_preloaded_skills_prompt=skill_commands_module.build_preloaded_skills_prompt,
    )


def build_skill_system_prompt(
    skill_name: str,
    hermes_repo: str | Path | None = None,
    skill_body_override: str | None = None,
) -> tuple[str, list[str]]:
    """Load a Hermes skill the same way the CLI preloads it for a session.

    When ``skill_body_override`` is provided, inline it as the active skill body so
    we can evaluate candidate variants with the real Hermes runtime before they are
    written back into the target repository.
    """
    if skill_body_override is not None:
        prompt = (
            f'[SYSTEM: The user launched this evaluation session with the "{skill_name}" skill '
            "preloaded. Treat its instructions as active guidance for the duration of this "
            "session unless overridden.]\n\n"
            f"# Active Skill: {skill_name}\n\n{skill_body_override}"
        )
        return prompt, [skill_name]

    symbols = _load_hermes_symbols(hermes_repo)
    skills_prompt, loaded_skills, missing_skills = symbols.build_preloaded_skills_prompt([skill_name])
    if missing_skills:
        missing_display = ", ".join(missing_skills)
        raise ValueError(f"Unknown skill(s): {missing_display}")
    if not skills_prompt:
        raise ValueError(f"Failed to build prompt for skill: {skill_name}")
    return skills_prompt, loaded_skills


def run_skill_eval(
    case: HermesSkillEvalCase,
    *,
    model: str = "openai/gpt-4.1-mini",
    hermes_repo: str | Path | None = None,
    agent_kwargs: Optional[dict[str, Any]] = None,
    skill_body_override: str | None = None,
) -> HermesSkillEvalResult:
    """Run one evaluation case through a real Hermes Agent instance."""
    symbols = _load_hermes_symbols(hermes_repo)
    skills_prompt, loaded_skills = build_skill_system_prompt(
        case.skill_name,
        hermes_repo=hermes_repo,
        skill_body_override=skill_body_override,
    )
    effective_system_prompt = "\n\n".join(
        part for part in (case.system_prompt, skills_prompt) if part
    ).strip()

    merged_agent_kwargs = {
        "model": model,
        "quiet_mode": True,
        "skip_context_files": True,
        "skip_memory": True,
        "ephemeral_system_prompt": effective_system_prompt or None,
    }
    if agent_kwargs:
        merged_agent_kwargs.update(agent_kwargs)

    agent = symbols.AIAgent(**merged_agent_kwargs)
    raw_result = agent.run_conversation(
        user_message=case.task_input,
        conversation_history=case.conversation_history,
    )

    if isinstance(raw_result, dict):
        final_response = str(raw_result.get("final_response", ""))
    else:
        final_response = str(raw_result)

    return HermesSkillEvalResult(
        final_response=final_response,
        loaded_skills=loaded_skills,
        effective_system_prompt=effective_system_prompt,
        raw_result=raw_result,
    )
