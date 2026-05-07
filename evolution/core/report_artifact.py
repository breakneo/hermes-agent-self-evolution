"""Helpers for generating structured evolution report artifacts."""

from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Any


def summarize_recommendation(metrics: dict[str, Any]) -> dict[str, Any]:
    """Derive a human-review recommendation from holdout + gate metrics."""
    reasons: list[str] = []
    tblite_gate = metrics.get("tblite_gate")

    if tblite_gate and not tblite_gate.get("passed", False):
        decision = "reject"
        reasons.append("benchmark_gate_failed")
    elif metrics.get("improvement", 0.0) > 0:
        decision = "candidate_for_review"
        reasons.append("holdout_improved")
        if tblite_gate and tblite_gate.get("passed"):
            reasons.append("benchmark_gate_passed")
    else:
        decision = "review_needed"
        reasons.append("no_holdout_improvement")
        if tblite_gate and tblite_gate.get("passed"):
            reasons.append("benchmark_gate_passed")

    return {
        "skill_name": metrics.get("skill_name"),
        "decision": decision,
        "reasons": reasons,
        "improvement": metrics.get("improvement"),
        "tblite_gate": tblite_gate,
    }


def build_evolution_report(
    *,
    metrics: dict[str, Any],
    baseline_skill_path: str | Path,
    evolved_skill_path: str | Path,
) -> str:
    """Render a markdown report for one evolution run."""
    summary = summarize_recommendation(metrics)
    tblite_gate = metrics.get("tblite_gate")

    lines = [
        "# Evolution Report",
        "",
        f"- Skill: `{metrics.get('skill_name')}`",
        f"- Timestamp: `{metrics.get('timestamp')}`",
        f"- Decision: `{summary['decision']}`",
        f"- Reasons: `{', '.join(summary['reasons']) or 'none'}`",
        "",
        "## Holdout Results",
        "",
        f"- Eval backend: `{metrics.get('eval_backend')}`",
        f"- Baseline score: `{metrics.get('baseline_score'):.3f}`",
        f"- Evolved score: `{metrics.get('evolved_score'):.3f}`",
        f"- Improvement: `{metrics.get('improvement'):+.3f}`",
        "",
        "## Benchmark Gate",
        "",
    ]

    if tblite_gate:
        lines.extend([
            f"- Summary: {tblite_gate.get('summary')}",
            f"- Mode: `{tblite_gate.get('mode')}`",
            f"- Task filter: `{tblite_gate.get('task_filter')}`",
            f"- Baseline pass rate: `{tblite_gate.get('baseline_pass_rate'):.3f}`",
            f"- Evolved pass rate: `{tblite_gate.get('evolved_pass_rate'):.3f}`",
            f"- Delta: `{tblite_gate.get('delta'):+.3f}`",
        ])
    else:
        lines.append("- Not run")

    lines.extend([
        "",
        "## Artifacts",
        "",
        f"- Baseline skill: `{baseline_skill_path}`",
        f"- Evolved skill: `{evolved_skill_path}`",
    ])
    return "\n".join(lines) + "\n"


def build_diff_summary(
    *,
    metrics: dict[str, Any],
    baseline_skill_path: str | Path,
    evolved_skill_path: str | Path,
) -> str:
    """Render a concise markdown diff summary for reviewers."""
    size_delta = metrics.get("evolved_size", 0) - metrics.get("baseline_size", 0)
    return "\n".join([
        "# Diff Summary",
        "",
        f"- Skill: `{metrics.get('skill_name')}`",
        f"- Holdout improvement: `{metrics.get('improvement', 0.0):+.3f}`",
        f"- Skill size delta: `{size_delta:+d}` chars",
        f"- Baseline artifact: `{baseline_skill_path}`",
        f"- Evolved artifact: `{evolved_skill_path}`",
        "",
        "## Reviewer Focus",
        "- Confirm the evolved skill improves behavior without overfitting.",
        "- Inspect prompt growth and instruction clarity.",
    ]) + "\n"


def build_review_checklist(metrics: dict[str, Any]) -> str:
    """Render a reviewer checklist for human approval."""
    summary = summarize_recommendation(metrics)
    tblite_gate = metrics.get("tblite_gate")

    lines = [
        "# Review Checklist",
        "",
        f"- [ ] Decision `{summary['decision']}` matches the evidence",
        f"- [ ] Holdout improvement looks meaningful (`{metrics.get('improvement', 0.0):+.3f}`)",
        "- [ ] Evolved skill wording is clearer or more robust than baseline",
    ]
    if tblite_gate:
        lines.append(f"- [ ] TBLite gate result reviewed (`{tblite_gate.get('summary')}`)")
    else:
        lines.append("- [ ] Confirm whether a benchmark gate is still needed")
    lines.append("- [ ] Approve only after inspecting the actual skill diff")
    return "\n".join(lines) + "\n"


def build_pr_draft(
    *,
    metrics: dict[str, Any],
    baseline_skill_path: str | Path,
    evolved_skill_path: str | Path,
) -> str:
    """Render a PR-ready markdown draft for candidate variants."""
    summary = summarize_recommendation(metrics)
    tblite_gate = metrics.get("tblite_gate")
    gate_line = tblite_gate.get("summary") if tblite_gate else "Not run"

    return "\n".join([
        "# PR Draft",
        "",
        f"Decision: `{summary['decision']}`",
        "",
        "## Summary",
        f"- Candidate skill: `{metrics.get('skill_name')}`",
        f"- Holdout improvement: `{metrics.get('improvement', 0.0):+.3f}`",
        f"- Benchmark gate: {gate_line}",
        "",
        "## Artifacts",
        f"- Baseline skill: `{baseline_skill_path}`",
        f"- Evolved skill: `{evolved_skill_path}`",
        "",
        "## Test Plan",
        "- [x] Focused self-evolution unit tests",
        "- [ ] Human review of the skill diff",
        "- [ ] Optional broader benchmark rerun before merge",
    ]) + "\n"


def build_github_pr_title(metrics: dict[str, Any]) -> str:
    """Build a concise GitHub PR title for an evolved skill candidate."""
    decision = summarize_recommendation(metrics)["decision"]
    suffix = "candidate" if decision == "candidate_for_review" else "review"
    return f"feat: evolve skill {metrics.get('skill_name')} ({suffix})"


def build_github_pr_body(
    *,
    metrics: dict[str, Any],
    baseline_skill_path: str | Path,
    evolved_skill_path: str | Path,
    report_path: str | Path,
    summary_path: str | Path,
    diff_summary_path: str | Path,
    review_checklist_path: str | Path,
) -> str:
    """Render a GitHub-ready PR body from evolution artifacts."""
    summary = summarize_recommendation(metrics)
    tblite_gate = metrics.get("tblite_gate")
    gate_line = tblite_gate.get("summary") if tblite_gate else "Not run"

    return "\n".join([
        "## Summary",
        f"- Evolved skill: `{metrics.get('skill_name')}`",
        f"- Decision: `{summary['decision']}`",
        f"- Reasons: `{', '.join(summary['reasons']) or 'none'}`",
        f"- Holdout improvement: `{metrics.get('improvement', 0.0):+.3f}`",
        "",
        "## Evaluation Evidence",
        f"- Holdout backend: `{metrics.get('eval_backend')}`",
        f"- Baseline score: `{metrics.get('baseline_score', 0.0):.3f}`",
        f"- Evolved score: `{metrics.get('evolved_score', 0.0):.3f}`",
        f"- Benchmark gate: {gate_line}",
        "",
        "## Artifacts",
        f"- Baseline skill: `{baseline_skill_path}`",
        f"- Evolved skill: `{evolved_skill_path}`",
        f"- Evolution report: `{report_path}`",
        f"- Summary JSON: `{summary_path}`",
        f"- Diff summary: `{diff_summary_path}`",
        f"- Review checklist: `{review_checklist_path}`",
        "",
        "## Test Plan",
        "- [x] Focused self-evolution unit tests",
        "- [ ] Review generated artifact set",
        "- [ ] Inspect evolved skill diff in context",
        "- [ ] Re-run broader benchmark gate if needed before merge",
    ]) + "\n"


def build_gh_pr_create_command(
    *,
    metrics: dict[str, Any],
    body_path: str | Path,
    base_branch: str = "main",
) -> str | None:
    """Build a gh CLI command suggestion when the run is PR-worthy."""
    decision = summarize_recommendation(metrics)["decision"]
    if decision == "reject":
        return None

    title = build_github_pr_title(metrics)
    draft_flag = " --draft" if decision == "review_needed" else ""
    return (
        "gh pr create"
        f" --base {shlex.quote(base_branch)}"
        f" --title {shlex.quote(title)}"
        f" --body-file {shlex.quote(str(body_path))}"
        f"{draft_flag}"
    )


def write_report_artifacts(
    *,
    output_dir: str | Path,
    metrics: dict[str, Any],
    baseline_skill_path: str | Path,
    evolved_skill_path: str | Path,
) -> dict[str, Path]:
    """Write markdown + machine-readable summary artifacts."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = summarize_recommendation(metrics)
    report_md = out_dir / "report.md"
    summary_json = out_dir / "summary.json"

    report_md.write_text(
        build_evolution_report(
            metrics=metrics,
            baseline_skill_path=baseline_skill_path,
            evolved_skill_path=evolved_skill_path,
        )
    )
    summary_json.write_text(json.dumps(summary, indent=2))

    return {
        "report_md": report_md,
        "summary_json": summary_json,
    }


def write_pr_ready_artifacts(
    *,
    output_dir: str | Path,
    metrics: dict[str, Any],
    baseline_skill_path: str | Path,
    evolved_skill_path: str | Path,
) -> dict[str, Path]:
    """Write PR-ready markdown artifacts for review and handoff."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pr_draft_md = out_dir / "pr_draft.md"
    review_checklist_md = out_dir / "review_checklist.md"
    diff_summary_md = out_dir / "diff_summary.md"

    pr_draft_md.write_text(
        build_pr_draft(
            metrics=metrics,
            baseline_skill_path=baseline_skill_path,
            evolved_skill_path=evolved_skill_path,
        )
    )
    review_checklist_md.write_text(build_review_checklist(metrics))
    diff_summary_md.write_text(
        build_diff_summary(
            metrics=metrics,
            baseline_skill_path=baseline_skill_path,
            evolved_skill_path=evolved_skill_path,
        )
    )

    return {
        "pr_draft_md": pr_draft_md,
        "review_checklist_md": review_checklist_md,
        "diff_summary_md": diff_summary_md,
    }


def write_github_pr_artifacts(
    *,
    output_dir: str | Path,
    metrics: dict[str, Any],
    baseline_skill_path: str | Path,
    evolved_skill_path: str | Path,
    report_path: str | Path,
    summary_path: str | Path,
    diff_summary_path: str | Path,
    review_checklist_path: str | Path,
    base_branch: str = "main",
) -> dict[str, Path | None]:
    """Write GitHub-ready PR body + optional gh command suggestion."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    github_pr_body_md = out_dir / "github_pr_body.md"
    github_pr_body_md.write_text(
        build_github_pr_body(
            metrics=metrics,
            baseline_skill_path=baseline_skill_path,
            evolved_skill_path=evolved_skill_path,
            report_path=report_path,
            summary_path=summary_path,
            diff_summary_path=diff_summary_path,
            review_checklist_path=review_checklist_path,
        )
    )

    gh_pr_create_command_txt = out_dir / "gh_pr_create_command.txt"
    command = build_gh_pr_create_command(
        metrics=metrics,
        body_path=github_pr_body_md,
        base_branch=base_branch,
    )
    if command is None:
        if gh_pr_create_command_txt.exists():
            gh_pr_create_command_txt.unlink()
        command_path: Path | None = None
    else:
        gh_pr_create_command_txt.write_text(command + "\n")
        command_path = gh_pr_create_command_txt

    return {
        "github_pr_body_md": github_pr_body_md,
        "gh_pr_create_command_txt": command_path,
    }
