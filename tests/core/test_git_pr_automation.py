"""Tests for git/GitHub automation artifacts."""

from __future__ import annotations

from pathlib import Path

import pytest

from evolution.core import git_pr_automation as mod


def sample_metrics() -> dict:
    return {
        "skill_name": "github-code-review",
        "timestamp": "20260414_180000",
        "eval_backend": "hermes",
        "baseline_score": 0.42,
        "evolved_score": 0.57,
        "improvement": 0.15,
        "tblite_gate": {
            "passed": True,
            "summary": "TBLite fast gate passed",
        },
    }


def test_build_evolution_branch_name_sanitizes_skill_and_includes_timestamp():
    branch = mod.build_evolution_branch_name(
        skill_name="github/code review",
        timestamp="20260414_180000",
    )

    assert branch == "evolution/github-code-review-20260414_180000"


def test_build_target_skill_path_resolves_skill_location_from_repo_root(tmp_path: Path):
    target = mod.build_target_skill_path(
        hermes_repo=tmp_path / "hermes-agent",
        skill_relpath="github/github-code-review/SKILL.md",
    )

    assert target == tmp_path / "hermes-agent" / "skills" / "github/github-code-review/SKILL.md"


def test_write_skill_patch_artifacts_writes_candidate_skill_file_and_manifest(tmp_path: Path):
    paths = mod.write_skill_patch_artifacts(
        output_dir=tmp_path,
        skill_relpath="github/github-code-review/SKILL.md",
        evolved_skill_text="# evolved skill\n",
        hermes_repo=tmp_path / "hermes-agent",
    )

    assert paths["candidate_skill_file"].exists()
    assert paths["git_patch_manifest_json"].exists()
    assert "github/github-code-review/SKILL.md" in paths["git_patch_manifest_json"].read_text()


def test_build_git_apply_plan_includes_branch_checkout_commit_and_push(tmp_path: Path):
    plan = mod.build_git_apply_plan(
        hermes_repo=tmp_path / "hermes-agent",
        branch_name="evolution/github-code-review-20260414_180000",
        target_skill_path=tmp_path / "hermes-agent/skills/github/github-code-review/SKILL.md",
        candidate_skill_file=tmp_path / "candidate_skill.md",
        commit_message="feat: evolve skill github-code-review (candidate)",
        push_remote="origin",
    )

    assert "git checkout -b evolution/github-code-review-20260414_180000" in plan
    assert "cp" in plan
    assert "git add" in plan
    assert "git commit -m" in plan
    assert "git push -u origin" in plan


def test_write_git_apply_plan_artifacts_writes_plan_files(tmp_path: Path):
    paths = mod.write_git_apply_plan_artifacts(
        output_dir=tmp_path,
        hermes_repo=tmp_path / "hermes-agent",
        branch_name="evolution/github-code-review-20260414_180000",
        target_skill_path=tmp_path / "hermes-agent/skills/github/github-code-review/SKILL.md",
        candidate_skill_file=tmp_path / "candidate_skill.md",
        commit_message="feat: evolve skill github-code-review (candidate)",
    )

    assert paths["git_apply_plan_sh"].exists()
    assert paths["git_apply_plan_md"].exists()
    assert "git checkout -b" in paths["git_apply_plan_sh"].read_text()
    assert "Commit message" in paths["git_apply_plan_md"].read_text()


def test_write_full_git_pr_automation_artifacts_writes_branch_and_pr_handoff(tmp_path: Path):
    paths = mod.write_git_pr_automation_artifacts(
        output_dir=tmp_path,
        metrics=sample_metrics(),
        hermes_repo=tmp_path / "hermes-agent",
        skill_relpath="github/github-code-review/SKILL.md",
        evolved_skill_text="# evolved skill\n",
        github_pr_body_path=tmp_path / "github_pr_body.md",
    )

    assert paths["candidate_skill_file"].exists()
    assert paths["git_apply_plan_sh"].exists()
    assert paths["git_apply_plan_md"].exists()
    assert paths["git_patch_manifest_json"].exists()
    assert paths["gh_pr_create_after_push_txt"].exists()
    assert "gh pr create" in paths["gh_pr_create_after_push_txt"].read_text()



def test_build_target_skill_path_rejects_escape_from_skills_root(tmp_path: Path):
    with pytest.raises(ValueError, match="skills"):
        mod.build_target_skill_path(
            hermes_repo=tmp_path / "hermes-agent",
            skill_relpath="../outside.md",
        )



def test_execute_git_apply_plan_rejects_target_outside_skills_root(tmp_path: Path):
    with pytest.raises(ValueError, match="skills"):
        mod.execute_git_apply_plan(
            hermes_repo=tmp_path / "hermes-agent",
            branch_name="evolution/github-code-review-20260414_180000",
            target_skill_path=tmp_path / "outside.md",
            candidate_skill_file=tmp_path / "candidate_skill.md",
            commit_message="feat: evolve skill github-code-review (candidate)",
            run_push=False,
            runner=lambda command, *, workdir=None: {"exit_code": 0, "output": "ok"},
        )



def test_write_full_git_pr_automation_artifacts_skips_pr_command_for_reject(tmp_path: Path):
    metrics = sample_metrics()
    metrics["tblite_gate"]["passed"] = False

    paths = mod.write_git_pr_automation_artifacts(
        output_dir=tmp_path,
        metrics=metrics,
        hermes_repo=tmp_path / "hermes-agent",
        skill_relpath="github/github-code-review/SKILL.md",
        evolved_skill_text="# evolved skill\n",
        github_pr_body_path=tmp_path / "github_pr_body.md",
    )

    assert paths["gh_pr_create_after_push_txt"] is None


def test_execute_git_apply_plan_runs_copy_commit_and_push_steps_in_order(tmp_path: Path):
    commands = []

    def _fake_runner(command: str, *, workdir=None):
        commands.append((command, workdir))
        return {"exit_code": 0, "output": "ok"}

    result = mod.execute_git_apply_plan(
        hermes_repo=tmp_path / "hermes-agent",
        branch_name="evolution/github-code-review-20260414_180000",
        target_skill_path=tmp_path / "hermes-agent/skills/github/github-code-review/SKILL.md",
        candidate_skill_file=tmp_path / "candidate_skill.md",
        commit_message="feat: evolve skill github-code-review (candidate)",
        run_push=True,
        runner=_fake_runner,
    )

    assert result["steps"][0]["name"] == "checkout_branch"
    assert result["steps"][-1]["name"] == "push_branch"
    assert any("git checkout -b" in cmd for cmd, _ in commands)
    assert any("git commit -m" in cmd for cmd, _ in commands)
    assert any("git push -u origin" in cmd for cmd, _ in commands)


def test_execute_git_apply_plan_can_skip_push(tmp_path: Path):
    commands = []

    def _fake_runner(command: str, *, workdir=None):
        commands.append(command)
        return {"exit_code": 0, "output": "ok"}

    result = mod.execute_git_apply_plan(
        hermes_repo=tmp_path / "hermes-agent",
        branch_name="evolution/github-code-review-20260414_180000",
        target_skill_path=tmp_path / "hermes-agent/skills/github/github-code-review/SKILL.md",
        candidate_skill_file=tmp_path / "candidate_skill.md",
        commit_message="feat: evolve skill github-code-review (candidate)",
        run_push=False,
        runner=_fake_runner,
    )

    assert all(step["name"] != "push_branch" for step in result["steps"])
    assert not any("git push -u origin" in cmd for cmd in commands)


def test_execute_gh_pr_create_runs_only_when_command_is_present(tmp_path: Path):
    commands = []

    def _fake_runner(command: str, *, workdir=None):
        commands.append((command, workdir))
        return {"exit_code": 0, "output": "created"}

    result = mod.execute_gh_pr_create(
        command="gh pr create --base main --head evolution/test --title 'x' --body-file body.md",
        hermes_repo=tmp_path / "hermes-agent",
        runner=_fake_runner,
    )

    assert result["executed"] is True
    assert commands[0][0].startswith("gh pr create")


def test_execute_full_git_pr_automation_skips_pr_creation_without_command(tmp_path: Path):
    calls = []

    def _fake_execute_git_apply_plan(**kwargs):
        calls.append(("apply", kwargs))
        return {"steps": [{"name": "checkout_branch"}]}

    def _fake_execute_gh_pr_create(**kwargs):
        calls.append(("pr", kwargs))
        return {"executed": False}

    result = mod.execute_git_pr_automation(
        git_apply_plan={
            "hermes_repo": tmp_path / "hermes-agent",
            "branch_name": "evolution/github-code-review-20260414_180000",
            "target_skill_path": tmp_path / "hermes-agent/skills/github/github-code-review/SKILL.md",
            "candidate_skill_file": tmp_path / "candidate_skill.md",
            "commit_message": "feat: evolve skill github-code-review (candidate)",
        },
        gh_pr_create_command=None,
        execute_push=True,
        execute_pr=False,
        execute_git_apply_plan_fn=_fake_execute_git_apply_plan,
        execute_gh_pr_create_fn=_fake_execute_gh_pr_create,
    )

    assert result["git"]["steps"][0]["name"] == "checkout_branch"
    assert result["pr"] is None
    assert [kind for kind, _ in calls] == ["apply"]
