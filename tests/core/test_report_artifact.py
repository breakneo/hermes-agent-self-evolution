"""Tests for report artifact helpers."""

from __future__ import annotations

import json
from pathlib import Path

from evolution.core import report_artifact as mod


def sample_metrics() -> dict:
    return {
        "skill_name": "github-code-review",
        "timestamp": "20260414_180000",
        "iterations": 5,
        "optimizer_model": "openai/gpt-4.1",
        "eval_model": "openai/gpt-4.1-mini",
        "eval_backend": "hermes",
        "baseline_score": 0.42,
        "evolved_score": 0.57,
        "improvement": 0.15,
        "baseline_size": 1000,
        "evolved_size": 1100,
        "train_examples": 10,
        "val_examples": 5,
        "holdout_examples": 5,
        "elapsed_seconds": 12.5,
        "constraints_passed": True,
        "tblite_gate": {
            "passed": True,
            "mode": "fast",
            "task_filter": "broken-python,pandas-etl",
            "base_config_path": "/tmp/local.yaml",
            "baseline_pass_rate": 0.50,
            "evolved_pass_rate": 0.48,
            "delta": -0.02,
            "threshold": 0.02,
            "summary": "TBLite fast gate passed",
        },
    }


def test_build_evolution_report_contains_decision_and_key_metrics():
    report = mod.build_evolution_report(
        metrics=sample_metrics(),
        baseline_skill_path="output/github-code-review/20260414_180000/baseline_skill.md",
        evolved_skill_path="output/github-code-review/20260414_180000/evolved_skill.md",
    )

    assert "# Evolution Report" in report
    assert "github-code-review" in report
    assert "candidate_for_review" in report
    assert "TBLite fast gate passed" in report
    assert "output/github-code-review/20260414_180000/evolved_skill.md" in report



def test_summarize_recommendation_rejects_on_failed_gate():
    metrics = sample_metrics()
    metrics["tblite_gate"]["passed"] = False
    metrics["tblite_gate"]["summary"] = "TBLite fast regression detected"

    summary = mod.summarize_recommendation(metrics)

    assert summary["decision"] == "reject"
    assert "benchmark_gate_failed" in summary["reasons"]



def test_write_report_artifacts_creates_markdown_and_summary_json(tmp_path: Path):
    metrics = sample_metrics()

    paths = mod.write_report_artifacts(
        output_dir=tmp_path,
        metrics=metrics,
        baseline_skill_path=tmp_path / "baseline_skill.md",
        evolved_skill_path=tmp_path / "evolved_skill.md",
    )

    assert paths["report_md"].exists()
    assert paths["summary_json"].exists()

    summary = json.loads(paths["summary_json"].read_text())
    assert summary["decision"] == "candidate_for_review"
    assert summary["skill_name"] == "github-code-review"



def test_write_pr_ready_artifacts_creates_pr_files(tmp_path: Path):
    metrics = sample_metrics()

    paths = mod.write_pr_ready_artifacts(
        output_dir=tmp_path,
        metrics=metrics,
        baseline_skill_path=tmp_path / "baseline_skill.md",
        evolved_skill_path=tmp_path / "evolved_skill.md",
    )

    assert paths["pr_draft_md"].exists()
    assert paths["review_checklist_md"].exists()
    assert paths["diff_summary_md"].exists()



def test_build_pr_draft_marks_review_candidate_and_contains_sections(tmp_path: Path):
    metrics = sample_metrics()

    draft = mod.build_pr_draft(
        metrics=metrics,
        baseline_skill_path=tmp_path / "baseline_skill.md",
        evolved_skill_path=tmp_path / "evolved_skill.md",
    )

    assert "# PR Draft" in draft
    assert "candidate_for_review" in draft
    assert "## Summary" in draft
    assert "## Test Plan" in draft



def test_build_review_checklist_mentions_gate_and_holdout():
    checklist = mod.build_review_checklist(sample_metrics())

    assert "Holdout improvement looks meaningful" in checklist
    assert "TBLite gate result reviewed" in checklist



def test_build_diff_summary_mentions_size_and_improvement(tmp_path: Path):
    summary = mod.build_diff_summary(
        metrics=sample_metrics(),
        baseline_skill_path=tmp_path / "baseline_skill.md",
        evolved_skill_path=tmp_path / "evolved_skill.md",
    )

    assert "Skill size delta" in summary
    assert "+0.150" in summary
    assert "baseline_skill.md" in summary


def test_build_github_pr_body_contains_github_sections_and_artifact_paths(tmp_path: Path):
    body = mod.build_github_pr_body(
        metrics=sample_metrics(),
        baseline_skill_path=tmp_path / "baseline_skill.md",
        evolved_skill_path=tmp_path / "evolved_skill.md",
        report_path=tmp_path / "report.md",
        summary_path=tmp_path / "summary.json",
        diff_summary_path=tmp_path / "diff_summary.md",
        review_checklist_path=tmp_path / "review_checklist.md",
    )

    assert "## Summary" in body
    assert "## Evaluation Evidence" in body
    assert "## Artifacts" in body
    assert "## Test Plan" in body
    assert "report.md" in body
    assert "review_checklist.md" in body


def test_build_gh_pr_create_command_marks_review_needed_variants_as_draft(tmp_path: Path):
    metrics = sample_metrics()
    metrics["improvement"] = 0.0
    metrics["evolved_score"] = metrics["baseline_score"]

    command = mod.build_gh_pr_create_command(
        metrics=metrics,
        body_path=tmp_path / "github_pr_body.md",
    )

    assert command is not None
    assert "gh pr create" in command
    assert "--draft" in command
    assert "--body-file" in command


def test_build_gh_pr_create_command_skips_rejected_variants(tmp_path: Path):
    metrics = sample_metrics()
    metrics["tblite_gate"]["passed"] = False
    metrics["tblite_gate"]["summary"] = "TBLite fast regression detected"

    command = mod.build_gh_pr_create_command(
        metrics=metrics,
        body_path=tmp_path / "github_pr_body.md",
    )

    assert command is None


def test_write_github_pr_artifacts_writes_body_and_command_for_candidates(tmp_path: Path):
    paths = mod.write_github_pr_artifacts(
        output_dir=tmp_path,
        metrics=sample_metrics(),
        baseline_skill_path=tmp_path / "baseline_skill.md",
        evolved_skill_path=tmp_path / "evolved_skill.md",
        report_path=tmp_path / "report.md",
        summary_path=tmp_path / "summary.json",
        diff_summary_path=tmp_path / "diff_summary.md",
        review_checklist_path=tmp_path / "review_checklist.md",
    )

    assert paths["github_pr_body_md"].exists()
    assert paths["gh_pr_create_command_txt"].exists()
    command = paths["gh_pr_create_command_txt"].read_text()
    assert "gh pr create" in command
    assert "--title" in command
