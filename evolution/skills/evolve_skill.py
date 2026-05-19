"""Evolve a Hermes Agent skill using DSPy + GEPA.

Usage:
    python -m evolution.skills.evolve_skill --skill github-code-review --iterations 10
    python -m evolution.skills.evolve_skill --skill arxiv --eval-source golden --dataset datasets/skills/arxiv/
"""

import json
import sys
import time
import os
import inspect
from contextlib import nullcontext
from pathlib import Path
from datetime import datetime
from typing import Optional

import click
import dspy
import yaml
from rich.console import Console
from rich.table import Table

from evolution.core.config import EvolutionConfig
from evolution.core.dataset_builder import SyntheticDatasetBuilder, EvalDataset, GoldenDatasetLoader
from evolution.core.external_importers import build_dataset_from_external
from evolution.core.fitness import skill_fitness_metric, LLMJudge, FitnessScore
from evolution.core.integration_adapter import HermesIntegrationAdapter
from evolution.core.eval_integrity import validate_model_separation
from evolution.core.phase_gate import evaluate_phase1_gate, write_phase_gate_result
from evolution.core.reproducibility import build_reproducibility_manifest, write_manifest
from evolution.core.rollout_policy import get_rollout_policy, should_auto_rollback
from evolution.core.stop_loss import StopLossGuard
from evolution.core.constraints import ConstraintValidator
from evolution.core.benchmark_gate import run_tblite_benchmark_gate
from evolution.core.report_artifact import (
    build_github_pr_title,
    write_report_artifacts,
    write_pr_ready_artifacts,
    write_github_pr_artifacts,
)
from evolution.core.git_pr_automation import (
    build_evolution_branch_name,
    write_git_pr_automation_artifacts,
    execute_git_pr_automation,
)
from evolution.core.hermes_eval import HermesSkillEvalCase, run_skill_eval
from evolution.skills.skill_module import (
    SkillModule,
    load_skill,
    find_skill,
    reassemble_skill,
)

console = Console()

# ── Constants ──────────────────────────────────────────────────────

DEFAULT_OPTIMIZER_MODEL = "openai/gpt-4.1"
DEFAULT_EVAL_MODEL = "openai/gpt-4.1-mini"
DEFAULT_HERMES_EVAL_MAX_ITERATIONS = 12


# ── Environment helpers ─────────────────────────────────────────────

def load_env_file(path: str | Path) -> dict[str, str]:
    """Load KEY=VALUE lines from an env file without overwriting existing env vars."""
    env_path = Path(path).expanduser()
    if not env_path.exists():
        return {}

    loaded: dict[str, str] = {}
    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or key in os.environ:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        os.environ[key] = value
        loaded[key] = value

    return loaded


def load_default_env_files() -> list[Path]:
    """Load default credential env files used by local Hermes workflows."""
    loaded_paths: list[Path] = []
    for candidate in [Path.home() / ".hermes" / ".env", Path.cwd() / ".env"]:
        if load_env_file(candidate):
            loaded_paths.append(candidate)
    return loaded_paths


def load_hermes_model_config(config_path: str | Path | None = None) -> dict:
    """Read the local Hermes model config used by the runtime, if available."""
    candidate = Path(config_path).expanduser() if config_path else Path.home() / ".hermes" / "config.yaml"
    if not candidate.exists():
        return {}
    data = yaml.safe_load(candidate.read_text()) or {}
    model_cfg = data.get("model") or {}
    return model_cfg if isinstance(model_cfg, dict) else {}


def resolve_runtime_model_settings(
    *,
    optimizer_model: str,
    eval_model: str,
    config_path: str | Path | None = None,
) -> tuple[str, str, dict[str, str]]:
    """Align self-evolution model defaults with the active Hermes runtime config."""
    model_cfg = load_hermes_model_config(config_path)
    provider = str(model_cfg.get("provider") or "").strip().lower()
    default_model = str(model_cfg.get("default") or "").strip()
    base_url = str(model_cfg.get("base_url") or "").strip().rstrip("/")
    applied_env: dict[str, str] = {}

    if provider == "custom" and default_model:
        if optimizer_model == DEFAULT_OPTIMIZER_MODEL:
            optimizer_model = default_model
        if eval_model == DEFAULT_EVAL_MODEL:
            eval_model = default_model

    if provider == "custom" and base_url:
        for env_name in ("OPENAI_BASE_URL", "OPENAI_API_BASE"):
            if not os.environ.get(env_name):
                os.environ[env_name] = base_url
                applied_env[env_name] = base_url

    return optimizer_model, eval_model, applied_env


# ── Scoring / evaluation helpers ────────────────────────────────────

def score_output_against_example(
    *,
    example,
    agent_output: str,
    skill_body: str,
    eval_model: str,
    hermes_agent_path: str | Path | None = None,
) -> float:
    """Score one agent output against an eval example using the richer LLM judge."""
    judge_config = EvolutionConfig(
        eval_model=eval_model,
        judge_model=eval_model,
    )
    if hermes_agent_path is not None:
        judge_config.hermes_agent_path = Path(hermes_agent_path).expanduser()
    judge = LLMJudge(judge_config)
    score = judge.score(
        task_input=example.task_input,
        expected_behavior=example.expected_behavior,
        agent_output=agent_output,
        skill_text=skill_body,
    )
    return score.composite


def evaluate_holdout(
    *,
    dataset: EvalDataset,
    eval_backend: str,
    baseline_module,
    evolved_module,
    baseline_skill_body: str,
    evolved_skill_body: str,
    eval_model: str,
    hermes_repo: str | None = None,
    skill_name: str | None = None,
):
    """Evaluate baseline and evolved variants on the holdout split."""
    holdout_examples = dataset.to_dspy_examples("holdout")

    baseline_scores = []
    evolved_scores = []

    if eval_backend == "dspy":
        lm = dspy.LM(eval_model)
        for ex in holdout_examples:
            with dspy.context(lm=lm):
                baseline_pred = baseline_module(task_input=ex.task_input)
                baseline_scores.append(skill_fitness_metric(ex, baseline_pred))

                evolved_pred = evolved_module(task_input=ex.task_input)
                evolved_scores.append(skill_fitness_metric(ex, evolved_pred))
        return baseline_scores, evolved_scores

    if eval_backend != "hermes":
        raise ValueError(f"Unknown eval backend: {eval_backend}")
    if not skill_name:
        raise ValueError("skill_name is required when eval_backend='hermes'")

    for ex in holdout_examples:
        case = HermesSkillEvalCase(
            skill_name=skill_name,
            task_input=ex.task_input,
        )
        baseline_result = run_skill_eval(
            case,
            model=eval_model,
            hermes_repo=hermes_repo,
            skill_body_override=baseline_skill_body,
            agent_kwargs={"max_iterations": DEFAULT_HERMES_EVAL_MAX_ITERATIONS},
        )
        baseline_scores.append(
            score_output_against_example(
                example=ex,
                agent_output=baseline_result.final_response,
                skill_body=baseline_skill_body,
                eval_model=eval_model,
                hermes_agent_path=hermes_repo,
            )
        )

        evolved_result = run_skill_eval(
            case,
            model=eval_model,
            hermes_repo=hermes_repo,
            skill_body_override=evolved_skill_body,
            agent_kwargs={"max_iterations": DEFAULT_HERMES_EVAL_MAX_ITERATIONS},
        )
        evolved_scores.append(
            score_output_against_example(
                example=ex,
                agent_output=evolved_result.final_response,
                skill_body=evolved_skill_body,
                eval_model=eval_model,
                hermes_agent_path=hermes_repo,
            )
        )

    return baseline_scores, evolved_scores


# ── Benchmark gate ──────────────────────────────────────────────────

def maybe_run_tblite_gate(
    *,
    run_tblite: bool,
    skill_name: str,
    baseline_skill_body: str,
    evolved_skill_body: str,
    hermes_repo: str | None,
    tblite_regression_threshold: float = 0.02,
    tblite_task_filter: str | None = None,
    tblite_mode: str = "fast",
):
    """Run the optional TBLite regression gate when enabled."""
    if not run_tblite:
        return None

    if not hermes_repo:
        raise ValueError("hermes_repo is required when run_tblite=True")

    return run_tblite_benchmark_gate(
        skill_name=skill_name,
        baseline_skill_body=baseline_skill_body,
        evolved_skill_body=evolved_skill_body,
        hermes_repo=hermes_repo,
        regression_threshold=tblite_regression_threshold,
        task_filter=tblite_task_filter,
        mode=tblite_mode,
    )


# ── Report / PR / git automation delegates ──────────────────────────

def write_evolution_report_artifacts(
    *,
    output_dir: str | Path,
    metrics: dict,
    baseline_skill_path: str | Path,
    evolved_skill_path: str | Path,
):
    """Write structured report artifacts for one evolution run."""
    return write_report_artifacts(
        output_dir=output_dir,
        metrics=metrics,
        baseline_skill_path=baseline_skill_path,
        evolved_skill_path=evolved_skill_path,
    )


def write_evolution_pr_ready_artifacts(
    *,
    output_dir: str | Path,
    metrics: dict,
    baseline_skill_path: str | Path,
    evolved_skill_path: str | Path,
):
    """Write PR-ready artifacts for one evolution run."""
    return write_pr_ready_artifacts(
        output_dir=output_dir,
        metrics=metrics,
        baseline_skill_path=baseline_skill_path,
        evolved_skill_path=evolved_skill_path,
    )


def write_evolution_github_pr_artifacts(
    *,
    output_dir: str | Path,
    metrics: dict,
    baseline_skill_path: str | Path,
    evolved_skill_path: str | Path,
    report_path: str | Path,
    summary_path: str | Path,
    diff_summary_path: str | Path,
    review_checklist_path: str | Path,
    base_branch: str = "main",
):
    """Write GitHub-ready PR body + gh command suggestion artifacts."""
    return write_github_pr_artifacts(
        output_dir=output_dir,
        metrics=metrics,
        baseline_skill_path=baseline_skill_path,
        evolved_skill_path=evolved_skill_path,
        report_path=report_path,
        summary_path=summary_path,
        diff_summary_path=diff_summary_path,
        review_checklist_path=review_checklist_path,
        base_branch=base_branch,
    )


def write_evolution_git_pr_automation_artifacts(
    *,
    output_dir: str | Path,
    metrics: dict,
    hermes_repo: str | Path,
    skill_relpath: str,
    evolved_skill_text: str,
    github_pr_body_path: str | Path,
):
    """Write git/GitHub connected handoff artifacts for an evolved skill."""
    return write_git_pr_automation_artifacts(
        output_dir=output_dir,
        metrics=metrics,
        hermes_repo=hermes_repo,
        skill_relpath=skill_relpath,
        evolved_skill_text=evolved_skill_text,
        github_pr_body_path=github_pr_body_path,
    )


def execute_evolution_git_pr_automation(
    *,
    git_apply_plan: dict,
    gh_pr_create_command: str | None,
    execute_push: bool,
    execute_pr: bool,
):
    """Execute the real git/PR automation flow."""
    return execute_git_pr_automation(
        git_apply_plan=git_apply_plan,
        gh_pr_create_command=gh_pr_create_command,
        execute_push=execute_push,
        execute_pr=execute_pr,
    )


# ── Constraint validation ───────────────────────────────────────────

def validate_skill_constraints(
    *,
    validator: ConstraintValidator,
    full_skill_text: str,
    baseline_full_text: str | None = None,
):
    """Validate skill constraints against the complete skill text, including frontmatter."""
    return validator.validate_all(
        full_skill_text,
        "skill",
        baseline_text=baseline_full_text,
    )


# ── GEPA / optimizer adapters ───────────────────────────────────────

class NormalizedReflectionLM:
    """Adapt DSPy/LiteLLM outputs to the list[str|dict] shape GEPA reflective mutation expects."""

    def __init__(self, base_lm):
        self.base_lm = base_lm

    def __call__(self, *args, **kwargs):
        result = self.base_lm(*args, **kwargs)
        if isinstance(result, list):
            if not result:
                return [""]
            normalized = []
            for item in result:
                if item is None:
                    normalized.append("")
                elif isinstance(item, (str, dict)):
                    normalized.append(item)
                else:
                    normalized.append(str(item))
            return normalized
        if result is None:
            return [""]
        if isinstance(result, (str, dict)):
            return [result]
        return [str(result)]

    def __getattr__(self, name):
        return getattr(self.base_lm, name)


def create_gepa_optimizer(*, metric, iterations: int, optimizer_model: str | None = None):
    """Instantiate GEPA across DSPy API variants."""
    params = inspect.signature(dspy.GEPA).parameters
    kwargs = {"metric": metric}
    if "max_steps" in params:
        kwargs["max_steps"] = iterations
    else:
        if callable(metric):
            metric_arity = len(inspect.signature(metric).parameters)
            if metric_arity < 5:
                def _wrapped_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
                    return metric(gold, pred, trace)
                kwargs["metric"] = _wrapped_metric
        if "reflection_lm" in params and optimizer_model:
            kwargs["reflection_lm"] = NormalizedReflectionLM(dspy.LM(optimizer_model))
        if "max_full_evals" in params:
            kwargs["max_full_evals"] = max(1, iterations)
        elif "auto" in params:
            kwargs["auto"] = "light"
    return dspy.GEPA(**kwargs)


def optimize_skill_module(*, baseline_module, trainset, valset, iterations: int, metric, optimizer_model: str | None = None):
    """Run GEPA when available, otherwise fall back to MIPROv2."""
    try:
        optimizer = create_gepa_optimizer(metric=metric, iterations=iterations, optimizer_model=optimizer_model)
        optimized = optimizer.compile(
            baseline_module,
            trainset=trainset,
            valset=valset,
        )
        return "GEPA", optimized
    except Exception as e:
        console.print(f"[yellow]GEPA not available ({e}), falling back to MIPROv2[/yellow]")
        optimizer = dspy.MIPROv2(
            metric=metric,
            auto="light",
        )
        optimized = optimizer.compile(
            baseline_module,
            trainset=trainset,
            valset=valset,
        )
        return "MIPROv2", optimized


def score_holdout_examples(
    holdout_examples: list[dspy.Example],
    baseline_module: SkillModule,
    optimized_module: SkillModule,
    lm,
    *,
    evolved_text_changed: bool,
) -> tuple[list[float], list[float]]:
    """Score baseline/evolved modules on holdout examples.

    If optimization produced identical skill text, copy baseline scores instead of
    calling the evolved module again. This prevents stochastic LLM generations from
    inventing a fake KPI regression for a no-op optimization pass.
    """
    baseline_scores = []
    evolved_scores = []
    for ex in holdout_examples:
        with dspy.context(lm=lm) if lm is not None else nullcontext():
            baseline_pred = baseline_module(task_input=ex.task_input)
            baseline_score = skill_fitness_metric(ex, baseline_pred)
            baseline_scores.append(baseline_score)

            if evolved_text_changed:
                evolved_pred = optimized_module(task_input=ex.task_input)
                evolved_score = skill_fitness_metric(ex, evolved_pred)
            else:
                evolved_score = baseline_score
            evolved_scores.append(evolved_score)

    return baseline_scores, evolved_scores


def evolve(
    skill_name: str,
    iterations: int = 10,
    eval_source: str = "synthetic",
    dataset_path: Optional[str] = None,
    optimizer_model: str = "openai/gpt-4.1",
    eval_model: str = "openai/gpt-4.1-mini",
    judge_model: str = "openrouter/google/gemini-2.5-flash",
    hermes_repo: Optional[str] = None,
    run_tests: bool = False,
    dry_run: bool = False,
    allow_two_family_mode: bool = False,
    minimum_model_families: Optional[int] = None,
    execute_git_apply: bool = False,
    execute_push: bool = False,
    execute_pr: bool = False,
    push_remote: str = "origin",
):
    """Main evolution function — orchestrates the full optimization loop."""

    if minimum_model_families is None:
        resolved_min_families = 2 if allow_two_family_mode else 3
    else:
        resolved_min_families = max(1, int(minimum_model_families))

    config = EvolutionConfig(
        iterations=iterations,
        optimizer_model=optimizer_model,
        eval_model=eval_model,
        judge_model=judge_model,
        run_pytest=run_tests,
        minimum_model_families=resolved_min_families,
    )
    if hermes_repo:
        config.hermes_agent_path = Path(hermes_repo)

    adapter = HermesIntegrationAdapter(config.hermes_agent_path)
    if config.phase0_enforce:
        try:
            adapter.assert_compatible()
        except RuntimeError as exc:
            console.print(f"[red]✗ Phase 0 compatibility check failed: {exc}[/red]")
            sys.exit(1)

        separation_errors = validate_model_separation(
            generator_model=config.judge_model,
            judge_model=config.eval_model,
            optimizer_model=config.optimizer_model,
            minimum_distinct_families=config.minimum_model_families,
        )
        if separation_errors:
            for error in separation_errors:
                console.print(f"[red]✗ {error}[/red]")
            sys.exit(1)

    stop_loss = StopLossGuard(config)

    # ── 1. Find and load the skill ──────────────────────────────────────
    console.print(f"\n[bold cyan]🧬 Hermes Agent Self-Evolution[/bold cyan] — Evolving skill: [bold]{skill_name}[/bold]\n")

    skill_path = find_skill(skill_name, config.hermes_agent_path)
    if not skill_path:
        console.print(f"[red]✗ Skill '{skill_name}' not found in {config.hermes_agent_path / 'skills'}[/red]")
        sys.exit(1)

    skill = load_skill(skill_path)
    console.print(f"  Loaded: {skill_path.relative_to(config.hermes_agent_path)}")
    console.print(f"  Name: {skill['name']}")
    console.print(f"  Size: {len(skill['raw']):,} chars")
    console.print(f"  Description: {skill['description'][:80]}...")

    if dry_run:
        console.print("\n[bold green]DRY RUN — setup validated successfully.[/bold green]")
        console.print(f"  Would generate eval dataset (source: {eval_source})")
        console.print(f"  Would run GEPA optimization ({iterations} iterations)")
        console.print("  Would validate constraints and create PR")
        return

    # ── 2. Build or load evaluation dataset ─────────────────────────────
    console.print(f"\n[bold]Building evaluation dataset[/bold] (source: {eval_source})")

    if eval_source == "golden" and dataset_path:
        dataset = GoldenDatasetLoader.load(Path(dataset_path))
        console.print(f"  Loaded golden dataset: {len(dataset.all_examples)} examples")
    elif eval_source == "sessiondb":
        save_path = Path(dataset_path) if dataset_path else Path("datasets") / "skills" / skill_name
        dataset = build_dataset_from_external(
            skill_name=skill_name,
            skill_text=skill["raw"],
            sources=["claude-code", "copilot", "hermes"],
            output_path=save_path,
            model=eval_model,
        )
        if not dataset.all_examples:
            console.print("[red]✗ No relevant examples found from session history[/red]")
            sys.exit(1)
        console.print(f"  Mined {len(dataset.all_examples)} examples from session history")
    elif eval_source == "synthetic":
        builder = SyntheticDatasetBuilder(config)
        dataset = builder.generate(
            artifact_text=skill["raw"],
            artifact_type="skill",
        )
        # Save for reuse
        save_path = Path("datasets") / "skills" / skill_name
        dataset.save(save_path)
        console.print(f"  Generated {len(dataset.all_examples)} synthetic examples")
        console.print(f"  Saved to {save_path}/")
    elif dataset_path:
        dataset = EvalDataset.load(Path(dataset_path))
        console.print(f"  Loaded dataset: {len(dataset.all_examples)} examples")
    else:
        console.print("[red]✗ Specify --dataset-path or use --eval-source synthetic[/red]")
        sys.exit(1)

    console.print(f"  Split: {len(dataset.train)} train / {len(dataset.val)} val / {len(dataset.holdout)} holdout")

    # ── 3. Validate constraints on baseline ─────────────────────────────
    console.print("\n[bold]Validating baseline constraints[/bold]")
    validator = ConstraintValidator(config)
    baseline_constraints = validator.validate_all(skill["raw"], "skill")
    all_pass = True
    for c in baseline_constraints:
        icon = "✓" if c.passed else "✗"
        color = "green" if c.passed else "red"
        console.print(f"  [{color}]{icon} {c.constraint_name}[/{color}]: {c.message}")
        if not c.passed:
            all_pass = False

    if not all_pass:
        console.print("[yellow]⚠ Baseline skill has constraint violations — proceeding anyway[/yellow]")

    # ── 4. Set up DSPy + GEPA optimizer ─────────────────────────────────
    console.print("\n[bold]Configuring optimizer[/bold]")
    console.print(f"  Optimizer: GEPA ({iterations} iterations)")
    console.print(f"  Optimizer model: {optimizer_model}")
    console.print(f"  Eval model: {eval_model}")

    # Configure DSPy
    lm = dspy.LM(eval_model)
    dspy.configure(lm=lm)

    # Create the baseline skill module
    baseline_module = SkillModule(skill["body"])

    # Prepare DSPy examples
    trainset = dataset.to_dspy_examples("train")
    valset = dataset.to_dspy_examples("val")

    # ── 5. Run GEPA optimization ────────────────────────────────────────
    console.print(f"\n[bold cyan]Running GEPA optimization ({iterations} iterations)...[/bold cyan]\n")

    start_time = time.time()

    try:
        optimizer = dspy.GEPA(
            metric=skill_fitness_metric,
            max_steps=iterations,
        )

        optimized_module = optimizer.compile(
            baseline_module,
            trainset=trainset,
            valset=valset,
        )
    except Exception as e:
        # Fall back to MIPROv2 if GEPA isn't available in this DSPy version
        console.print(f"[yellow]GEPA not available ({e}), falling back to MIPROv2[/yellow]")
        optimizer = dspy.MIPROv2(
            metric=skill_fitness_metric,
            auto="light",
        )
        optimized_module = optimizer.compile(
            baseline_module,
            trainset=trainset,
        )

    elapsed = time.time() - start_time
    console.print(f"\n  Optimization completed in {elapsed:.1f}s")

    # ── 6. Extract evolved skill text ───────────────────────────────────
    # The optimized module's instructions contain the evolved skill text
    evolved_body = optimized_module.skill_text
    evolved_full = reassemble_skill(skill["frontmatter"], evolved_body)

    # ── 7. Validate evolved skill ───────────────────────────────────────
    console.print("\n[bold]Validating evolved skill[/bold]")
    evolved_constraints = validator.validate_all(evolved_full, "skill", baseline_text=skill["raw"])
    all_pass = True
    for c in evolved_constraints:
        icon = "✓" if c.passed else "✗"
        color = "green" if c.passed else "red"
        console.print(f"  [{color}]{icon} {c.constraint_name}[/{color}]: {c.message}")
        if not c.passed:
            all_pass = False

    if not all_pass:
        console.print("[red]✗ Evolved skill FAILED constraints — not deploying[/red]")
        # Still save for inspection
        output_path = Path("output") / skill_name / "evolved_FAILED.md"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(evolved_full, encoding="utf-8")
        console.print(f"  Saved failed variant to {output_path}")
        return

    # ── 8. Evaluate on holdout set ──────────────────────────────────────
    console.print(f"\n[bold]Evaluating on holdout set ({len(dataset.holdout)} examples)[/bold]")

    holdout_examples = dataset.to_dspy_examples("holdout")

    evolved_text_changed = evolved_body != skill["body"]
    if not evolved_text_changed:
        console.print("[yellow]No skill text changes produced; reusing baseline holdout scores[/yellow]")

    baseline_scores, evolved_scores = score_holdout_examples(
        holdout_examples,
        baseline_module,
        optimized_module,
        lm,
        evolved_text_changed=evolved_text_changed,
    )

    avg_baseline = sum(baseline_scores) / max(1, len(baseline_scores))
    avg_evolved = sum(evolved_scores) / max(1, len(evolved_scores))
    improvement = avg_evolved - avg_baseline
    stop_loss.register_attempt(
        cost_usd=0.0,
        runtime_minutes=elapsed / 60.0,
        improvement=improvement,
        stable=improvement >= config.minimum_detectable_effect,
    )

    rollout_policy = get_rollout_policy("skill_text")
    rollback = should_auto_rollback(
        kpi_delta=improvement,
        safety_incidents=0,
        policy=rollout_policy,
    )

    # ── 9. Report results ───────────────────────────────────────────────
    table = Table(title="Evolution Results")
    table.add_column("Metric", style="bold")
    table.add_column("Baseline", justify="right")
    table.add_column("Evolved", justify="right")
    table.add_column("Change", justify="right")

    change_color = "green" if improvement > 0 else "red"
    table.add_row(
        "Holdout Score",
        f"{avg_baseline:.3f}",
        f"{avg_evolved:.3f}",
        f"[{change_color}]{improvement:+.3f}[/{change_color}]",
    )
    table.add_row(
        "Skill Size",
        f"{len(skill['body']):,} chars",
        f"{len(evolved_body):,} chars",
        f"{len(evolved_body) - len(skill['body']):+,} chars",
    )
    table.add_row("Time", "", f"{elapsed:.1f}s", "")
    table.add_row("Iterations", "", str(iterations), "")

    console.print()
    console.print(table)

    if rollback:
        console.print("[yellow]⚠ Rollout policy recommends rollback due to KPI regression[/yellow]")

    stop_loss_reasons = stop_loss.termination_reasons()
    if stop_loss_reasons:
        console.print("[yellow]⚠ Stop-loss triggered:[/yellow]")
        for reason in stop_loss_reasons:
            console.print(f"  - {reason}")

    # ── 10. Save output ─────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("output") / skill_name / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = build_reproducibility_manifest(
        skill_name=skill_name,
        skill_path=skill_path,
        dataset=dataset,
        config=config,
        baseline_score=avg_baseline,
        evolved_score=avg_evolved,
        improvement=improvement,
        elapsed_seconds=elapsed,
        evolution_repo_path=Path(__file__).resolve().parents[2],
    )
    gate_result = evaluate_phase1_gate(
        baseline_score=avg_baseline,
        evolved_score=avg_evolved,
        improvement=improvement,
        config=config,
        manifest=manifest,
        benchmark_regression=None,
        safety_incidents=0,
    )

    # Save evolved skill
    (output_dir / "evolved_skill.md").write_text(evolved_full, encoding="utf-8")

    # Save baseline for comparison
    (output_dir / "baseline_skill.md").write_text(skill["raw"], encoding="utf-8")

    # Save reproducibility and gate artifacts
    write_manifest(output_dir / "reproducibility_manifest.json", manifest)
    write_phase_gate_result(output_dir / "phase1_gate.json", gate_result)
    if gate_result.passed:
        console.print("[bold green]✓ Phase 1 gate passed[/bold green]")
    else:
        console.print("[yellow]⚠ Phase 1 gate failed[/yellow]")
        for failure in gate_result.failures:
            console.print(f"  - {failure}")

    # Save metrics
    metrics = {
        "skill_name": skill_name,
        "timestamp": timestamp,
        "iterations": iterations,
        "optimizer_model": optimizer_model,
        "eval_model": eval_model,
        "baseline_score": avg_baseline,
        "evolved_score": avg_evolved,
        "improvement": improvement,
        "baseline_size": len(skill["body"]),
        "evolved_size": len(evolved_body),
        "train_examples": len(dataset.train),
        "val_examples": len(dataset.val),
        "holdout_examples": len(dataset.holdout),
        "elapsed_seconds": elapsed,
        "constraints_passed": all_pass,
        "rollout_level": rollout_policy.rollout_level,
        "rollback_recommended": rollback,
        "stop_loss_reasons": stop_loss_reasons,
        "phase1_gate_passed": gate_result.passed,
        "phase1_gate_failures": gate_result.failures,
        "reproducibility_manifest_path": str(output_dir / "reproducibility_manifest.json"),
        "phase1_gate_path": str(output_dir / "phase1_gate.json"),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    console.print(f"\n  Output saved to {output_dir}/")

    # ── 11. Git / PR automation ────────────────────────────────────
    if execute_git_apply:
        if not config.hermes_agent_path or not (config.hermes_agent_path / ".git").exists():
            console.print(
                "[red]✗ --execute-git-apply requires a valid hermes-agent git repo. "
                "Pass --hermes-repo or run from inside one.[/red]"
            )
            return

        if execute_pr and not execute_push:
            console.print("[yellow]⚠ --execute-pr requires --execute-push; enabling push automatically[/yellow]")
            execute_push = True

        console.print("\n[bold]Executing git apply plan[/bold]")

        # Determine skill_relpath from the loaded skill path
        try:
            skill_relpath = str(Path(skill_path).relative_to(config.hermes_agent_path / "skills"))
        except ValueError:
            console.print(
                f"[red]✗ Skill path {skill_path} is not inside "
                f"{config.hermes_agent_path / 'skills'}[/red]"
            )
            return

        # Write PR-ready artifacts
        pr_artifacts = write_pr_ready_artifacts(
            output_dir=output_dir,
            metrics=metrics,
            baseline_skill_path=output_dir / "baseline_skill.md",
            evolved_skill_path=output_dir / "evolved_skill.md",
        )

        # Write report artifacts
        report_artifacts = write_report_artifacts(
            output_dir=output_dir,
            metrics=metrics,
            baseline_skill_path=output_dir / "baseline_skill.md",
            evolved_skill_path=output_dir / "evolved_skill.md",
        )

        # Write GitHub PR artifacts
        github_artifacts = write_github_pr_artifacts(
            output_dir=output_dir,
            metrics=metrics,
            baseline_skill_path=output_dir / "baseline_skill.md",
            evolved_skill_path=output_dir / "evolved_skill.md",
            report_path=report_artifacts["report_md"],
            summary_path=report_artifacts["summary_json"],
            diff_summary_path=pr_artifacts["diff_summary_md"],
            review_checklist_path=pr_artifacts["review_checklist_md"],
        )

        # Write git PR automation artifacts
        github_pr_body = github_artifacts["github_pr_body_md"]
        assert github_pr_body is not None  # always written by write_github_pr_artifacts
        automation_artifacts = write_git_pr_automation_artifacts(
            output_dir=output_dir,
            metrics=metrics,
            hermes_repo=config.hermes_agent_path,
            skill_relpath=skill_relpath,
            evolved_skill_text=evolved_full,
            github_pr_body_path=github_pr_body,
            push_remote=push_remote,
        )

        # Execute the automation
        gh_command: str | None = None
        cmd_path = automation_artifacts.get("gh_pr_create_after_push_txt")
        if cmd_path is not None:
            gh_command = cmd_path.read_text().strip()

        commit_message = build_github_pr_title(metrics)
        branch_name = build_evolution_branch_name(
            skill_name=skill_name,
            timestamp=timestamp,
        )

        try:
            exec_result = execute_git_pr_automation(
                git_apply_plan={
                    "hermes_repo": str(config.hermes_agent_path),
                    "branch_name": branch_name,
                    "target_skill_path": str(automation_artifacts["target_skill_path"]),
                    "candidate_skill_file": str(automation_artifacts["candidate_skill_file"]),
                    "commit_message": commit_message,
                },
                gh_pr_create_command=gh_command,
                execute_push=execute_push,
                execute_pr=execute_pr,
            )

            console.print(
                f"\n[bold green]✓ Git apply executed — branch [bold]{branch_name}[/bold][/bold green]"
            )
            if execute_push:
                console.print(f"  Pushed to {push_remote}/{branch_name}")
            if execute_pr and exec_result.get("pr"):
                console.print(f"  PR created: {exec_result['pr'].get('output', '').strip()}")
        except Exception as exc:
            console.print(f"[red]✗ Git automation failed: {exc}[/red]")
            return

    if improvement > 0:
        console.print(f"\n[bold green]✓ Evolution improved skill by {improvement:+.3f} ({improvement/max(0.001, avg_baseline)*100:+.1f}%)[/bold green]")
        console.print(f"  Review the diff: diff {output_dir}/baseline_skill.md {output_dir}/evolved_skill.md")
    else:
        console.print(f"\n[yellow]⚠ Evolution did not improve skill (change: {improvement:+.3f})[/yellow]")
        console.print("  Try: more iterations, better eval dataset, or different optimizer model")


@click.command()
@click.option("--skill", required=True, help="Name of the skill to evolve")
@click.option("--iterations", default=10, help="Number of GEPA iterations")
@click.option("--eval-source", default="synthetic", type=click.Choice(["synthetic", "golden", "sessiondb"]),
              help="Source for evaluation dataset")
@click.option("--dataset-path", default=None, help="Path to existing eval dataset (JSONL)")
@click.option("--optimizer-model", default="openai/gpt-4.1", help="Model for GEPA reflections")
@click.option("--eval-model", default="openai/gpt-4.1-mini", help="Model for evaluations")
@click.option(
    "--judge-model",
    default="openrouter/google/gemini-2.5-flash",
    help="Model for dataset generation and relevance scoring",
)
@click.option("--hermes-repo", default=None, help="Path to hermes-agent repo")
@click.option("--run-tests", is_flag=True, help="Run full pytest suite as constraint gate")
@click.option("--dry-run", is_flag=True, help="Validate setup without running optimization")
@click.option(
    "--allow-two-family-mode",
    is_flag=True,
    help="Allow execution with two distinct model families instead of three",
)
@click.option(
    "--minimum-model-families",
    type=int,
    default=None,
    help="Minimum distinct model families required (overrides default/--allow-two-family-mode)",
)
@click.option(
    "--execute-git-apply",
    is_flag=True,
    help="Apply the evolved skill to the hermes-agent repo via git branch + commit",
)
@click.option(
    "--execute-push",
    is_flag=True,
    help="Push the evolution branch to remote (requires --execute-git-apply)",
)
@click.option(
    "--execute-pr",
    is_flag=True,
    help="Create a GitHub PR from the evolution branch (requires --execute-push)",
)
@click.option(
    "--push-remote",
    default="origin",
    help="Remote name for git push (default: origin)",
)
def main(
    skill,
    iterations,
    eval_source,
    dataset_path,
    optimizer_model,
    eval_model,
    judge_model,
    hermes_repo,
    run_tests,
    dry_run,
    allow_two_family_mode,
    minimum_model_families,
    execute_git_apply,
    execute_push,
    execute_pr,
    push_remote,
):
    """Evolve a Hermes Agent skill using DSPy + GEPA optimization."""
    evolve(
        skill_name=skill,
        iterations=iterations,
        eval_source=eval_source,
        dataset_path=dataset_path,
        optimizer_model=optimizer_model,
        eval_model=eval_model,
        judge_model=judge_model,
        hermes_repo=hermes_repo,
        run_tests=run_tests,
        dry_run=dry_run,
        allow_two_family_mode=allow_two_family_mode,
        minimum_model_families=minimum_model_families,
        execute_git_apply=execute_git_apply,
        execute_push=execute_push,
        execute_pr=execute_pr,
        push_remote=push_remote,
    )


if __name__ == "__main__":
    main()
