"""Core infrastructure shared across all evolution phases."""

from evolution.core.config import EvolutionConfig, get_hermes_agent_path
from evolution.core.benchmark_gate import TBLiteGateResult, run_tblite_benchmark_gate
from evolution.core.git_pr_automation import (
    build_evolution_branch_name,
    build_git_apply_plan,
    build_target_skill_path,
    write_git_apply_plan_artifacts,
    write_git_pr_automation_artifacts,
    write_skill_patch_artifacts,
)
from evolution.core.report_artifact import (
    build_diff_summary,
    build_evolution_report,
    build_github_pr_body,
    build_github_pr_title,
    build_gh_pr_create_command,
    build_pr_draft,
    build_review_checklist,
    summarize_recommendation,
    write_github_pr_artifacts,
    write_pr_ready_artifacts,
    write_report_artifacts,
)
