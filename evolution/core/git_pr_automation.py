"""Helpers for turning evolved skill artifacts into git/GitHub automation plans."""

from __future__ import annotations

import json
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any, Callable

from evolution.core.report_artifact import build_github_pr_title, summarize_recommendation

TerminalRunner = Callable[[str], dict[str, Any]]


def _slugify_skill_name(skill_name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", skill_name).strip("-").lower()
    return slug or "skill"


def _skills_root(hermes_repo: str | Path) -> Path:
    repo = Path(hermes_repo).expanduser().resolve()
    return (repo / "skills").resolve(strict=False)


def _ensure_within_skills_root(*, hermes_repo: str | Path, target_path: str | Path) -> Path:
    repo = Path(hermes_repo).expanduser().resolve()
    skills_root = _skills_root(repo)
    target = Path(target_path).expanduser()
    if not target.is_absolute():
        target = repo / target
    target = target.resolve(strict=False)
    try:
        target.relative_to(skills_root)
    except ValueError as exc:
        raise ValueError(f"target skill path must stay within {skills_root}") from exc
    return target


def _default_runner(command: str, *, workdir: str | Path | None = None) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        shell=True,
        cwd=str(workdir) if workdir is not None else None,
        capture_output=True,
        text=True,
    )
    return {
        "exit_code": completed.returncode,
        "output": (completed.stdout or "") + (completed.stderr or ""),
    }


def _run_step(
    *,
    name: str,
    command: str,
    workdir: str | Path,
    runner: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    result = runner(command, workdir=workdir)
    exit_code = result.get("exit_code", 1)
    output = result.get("output", "")
    if exit_code != 0:
        raise RuntimeError(f"{name} failed ({exit_code}): {output}")
    return {
        "name": name,
        "command": command,
        "exit_code": exit_code,
        "output": output,
    }


def build_evolution_branch_name(*, skill_name: str, timestamp: str) -> str:
    """Build a deterministic branch name for an evolution candidate."""
    return f"evolution/{_slugify_skill_name(skill_name)}-{timestamp}"


def build_target_skill_path(*, hermes_repo: str | Path, skill_relpath: str) -> Path:
    """Resolve the target skill path inside a hermes-agent checkout."""
    relative_path = Path(skill_relpath)
    if relative_path.is_absolute():
        raise ValueError("skill_relpath must be relative to the skills directory")
    return _ensure_within_skills_root(
        hermes_repo=hermes_repo,
        target_path=Path("skills") / relative_path,
    )


def write_skill_patch_artifacts(
    *,
    output_dir: str | Path,
    skill_relpath: str,
    evolved_skill_text: str,
    hermes_repo: str | Path,
) -> dict[str, Path]:
    """Write a candidate skill file plus a manifest describing where it should land."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    candidate_skill_file = out_dir / "candidate_skill_patch.md"
    candidate_skill_file.write_text(evolved_skill_text)

    target_skill_path = build_target_skill_path(hermes_repo=hermes_repo, skill_relpath=skill_relpath)
    git_patch_manifest_json = out_dir / "git_patch_manifest.json"
    git_patch_manifest_json.write_text(json.dumps({
        "skill_relpath": skill_relpath,
        "target_skill_path": str(target_skill_path),
        "candidate_skill_file": str(candidate_skill_file),
    }, indent=2))

    return {
        "candidate_skill_file": candidate_skill_file,
        "git_patch_manifest_json": git_patch_manifest_json,
        "target_skill_path": target_skill_path,
    }


def build_git_apply_plan(
    *,
    hermes_repo: str | Path,
    branch_name: str,
    target_skill_path: str | Path,
    candidate_skill_file: str | Path,
    commit_message: str,
    push_remote: str = "origin",
) -> str:
    """Build a shell plan that applies the evolved skill in a hermes repo."""
    repo = Path(hermes_repo).expanduser().resolve()
    target = _ensure_within_skills_root(hermes_repo=repo, target_path=target_skill_path)
    candidate = Path(candidate_skill_file).expanduser().resolve(strict=False)

    return "\n".join([
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"cd {shlex.quote(str(repo))}",
        f"git checkout -b {shlex.quote(branch_name)}",
        f"mkdir -p {shlex.quote(str(target.parent))}",
        f"cp {shlex.quote(str(candidate))} {shlex.quote(str(target))}",
        f"git add {shlex.quote(str(target))}",
        f"git commit -m {shlex.quote(commit_message)}",
        f"git push -u {shlex.quote(push_remote)} {shlex.quote(branch_name)}",
        "",
    ])


def write_git_apply_plan_artifacts(
    *,
    output_dir: str | Path,
    hermes_repo: str | Path,
    branch_name: str,
    target_skill_path: str | Path,
    candidate_skill_file: str | Path,
    commit_message: str,
    push_remote: str = "origin",
) -> dict[str, Path]:
    """Write shell and markdown plans for applying the evolved skill via git."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    git_apply_plan_sh = out_dir / "git_apply_plan.sh"
    git_apply_plan_md = out_dir / "git_apply_plan.md"

    shell_plan = build_git_apply_plan(
        hermes_repo=hermes_repo,
        branch_name=branch_name,
        target_skill_path=target_skill_path,
        candidate_skill_file=candidate_skill_file,
        commit_message=commit_message,
        push_remote=push_remote,
    )
    git_apply_plan_sh.write_text(shell_plan)
    git_apply_plan_md.write_text(
        "\n".join([
            "# Git Apply Plan",
            "",
            f"- Hermes repo: `{hermes_repo}`",
            f"- Branch: `{branch_name}`",
            f"- Target skill path: `{target_skill_path}`",
            f"- Candidate file: `{candidate_skill_file}`",
            f"- Commit message: `{commit_message}`",
            "",
            "## Shell Plan",
            "```bash",
            shell_plan.rstrip(),
            "```",
            "",
        ])
    )

    return {
        "git_apply_plan_sh": git_apply_plan_sh,
        "git_apply_plan_md": git_apply_plan_md,
    }


def write_git_pr_automation_artifacts(
    *,
    output_dir: str | Path,
    metrics: dict[str, Any],
    hermes_repo: str | Path,
    skill_relpath: str,
    evolved_skill_text: str,
    github_pr_body_path: str | Path,
    push_remote: str = "origin",
) -> dict[str, Path | None]:
    """Write the full git/GitHub handoff pack for applying an evolved skill."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    patch_paths = write_skill_patch_artifacts(
        output_dir=out_dir,
        skill_relpath=skill_relpath,
        evolved_skill_text=evolved_skill_text,
        hermes_repo=hermes_repo,
    )

    branch_name = build_evolution_branch_name(
        skill_name=metrics.get("skill_name", "skill"),
        timestamp=metrics.get("timestamp", "unknown"),
    )
    commit_message = build_github_pr_title(metrics)
    plan_paths = write_git_apply_plan_artifacts(
        output_dir=out_dir,
        hermes_repo=hermes_repo,
        branch_name=branch_name,
        target_skill_path=patch_paths["target_skill_path"],
        candidate_skill_file=patch_paths["candidate_skill_file"],
        commit_message=commit_message,
        push_remote=push_remote,
    )

    decision = summarize_recommendation(metrics)["decision"]
    gh_pr_create_after_push_txt = out_dir / "gh_pr_create_after_push.txt"
    if decision == "reject":
        command_path: Path | None = None
        if gh_pr_create_after_push_txt.exists():
            gh_pr_create_after_push_txt.unlink()
    else:
        draft_flag = " --draft" if decision == "review_needed" else ""
        command = (
            f"cd {shlex.quote(str(hermes_repo))} && "
            f"gh pr create --base main --head {shlex.quote(branch_name)} "
            f"--title {shlex.quote(commit_message)} "
            f"--body-file {shlex.quote(str(github_pr_body_path))}{draft_flag}"
        )
        gh_pr_create_after_push_txt.write_text(command + "\n")
        command_path = gh_pr_create_after_push_txt

    return {
        "candidate_skill_file": patch_paths["candidate_skill_file"],
        "git_patch_manifest_json": patch_paths["git_patch_manifest_json"],
        "target_skill_path": patch_paths["target_skill_path"],
        "git_apply_plan_sh": plan_paths["git_apply_plan_sh"],
        "git_apply_plan_md": plan_paths["git_apply_plan_md"],
        "gh_pr_create_after_push_txt": command_path,
    }


def execute_git_apply_plan(
    *,
    hermes_repo: str | Path,
    branch_name: str,
    target_skill_path: str | Path,
    candidate_skill_file: str | Path,
    commit_message: str,
    push_remote: str = "origin",
    run_push: bool = True,
    runner: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Execute the git apply plan step by step with an injectable runner."""
    run = runner or _default_runner
    repo = Path(hermes_repo).expanduser().resolve()
    target = _ensure_within_skills_root(hermes_repo=repo, target_path=target_skill_path)
    candidate = Path(candidate_skill_file).expanduser().resolve(strict=False)

    steps = [
        _run_step(
            name="checkout_branch",
            command=f"git checkout -b {shlex.quote(branch_name)}",
            workdir=repo,
            runner=run,
        ),
        _run_step(
            name="ensure_target_dir",
            command=f"mkdir -p {shlex.quote(str(target.parent))}",
            workdir=repo,
            runner=run,
        ),
        _run_step(
            name="copy_candidate_skill",
            command=f"cp {shlex.quote(str(candidate))} {shlex.quote(str(target))}",
            workdir=repo,
            runner=run,
        ),
        _run_step(
            name="git_add",
            command=f"git add {shlex.quote(str(target))}",
            workdir=repo,
            runner=run,
        ),
        _run_step(
            name="git_commit",
            command=f"git commit -m {shlex.quote(commit_message)}",
            workdir=repo,
            runner=run,
        ),
    ]
    if run_push:
        steps.append(
            _run_step(
                name="push_branch",
                command=f"git push -u {shlex.quote(push_remote)} {shlex.quote(branch_name)}",
                workdir=repo,
                runner=run,
            )
        )

    return {
        "executed": True,
        "steps": steps,
    }


def execute_gh_pr_create(
    *,
    command: str | None,
    hermes_repo: str | Path,
    runner: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Execute a gh pr create command when present."""
    if not command:
        return {"executed": False, "reason": "no_command"}

    run = runner or _default_runner
    result = run(command, workdir=hermes_repo)
    exit_code = result.get("exit_code", 1)
    output = result.get("output", "")
    if exit_code != 0:
        raise RuntimeError(f"gh_pr_create failed ({exit_code}): {output}")
    return {
        "executed": True,
        "command": command,
        "exit_code": exit_code,
        "output": output,
    }


def execute_git_pr_automation(
    *,
    git_apply_plan: dict[str, Any],
    gh_pr_create_command: str | None,
    execute_push: bool,
    execute_pr: bool,
    execute_git_apply_plan_fn=execute_git_apply_plan,
    execute_gh_pr_create_fn=execute_gh_pr_create,
) -> dict[str, Any]:
    """Execute the real git apply flow, optionally followed by PR creation."""
    git_result = execute_git_apply_plan_fn(
        **git_apply_plan,
        run_push=execute_push,
    )

    pr_result = None
    if execute_pr and gh_pr_create_command:
        pr_result = execute_gh_pr_create_fn(
            command=gh_pr_create_command,
            hermes_repo=git_apply_plan["hermes_repo"],
        )

    return {
        "git": git_result,
        "pr": pr_result,
    }
