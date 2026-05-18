"""Evolve a Hermes Agent skill using DSPy + GEPA.

Usage:
    python -m evolution.skills.evolve_skill --skill github-code-review --iterations 10
    python -m evolution.skills.evolve_skill --skill arxiv --eval-source golden --dataset datasets/skills/arxiv/
"""

import inspect
import json
import os
import sys
import time
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
from evolution.core.fitness import LLMJudge, skill_fitness_metric
from evolution.core.constraints import ConstraintValidator
from evolution.core.hermes_eval import HermesSkillEvalCase, run_skill_eval
from evolution.core.benchmark_gate import run_tblite_benchmark_gate
from evolution.core.report_artifact import (
    build_github_pr_title,
    write_github_pr_artifacts,
    write_pr_ready_artifacts,
    write_report_artifacts,
)
from evolution.core.git_pr_automation import (
    build_evolution_branch_name,
    build_target_skill_path,
    execute_git_pr_automation,
    write_git_pr_automation_artifacts,
)
from evolution.skills.skill_module import (
    SkillModule,
    load_skill,
    find_skill,
    reassemble_skill,
)

console = Console()

DEFAULT_OPTIMIZER_MODEL = "openai/gpt-4.1"
DEFAULT_EVAL_MODEL = "openai/gpt-4.1-mini"
DEFAULT_HERMES_EVAL_MAX_ITERATIONS = 12


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
    """Align self-evolution model defaults with the active Hermes runtime config.

    On machines where Hermes is configured against a custom OpenAI-compatible endpoint,
    the repo's hardcoded OpenAI defaults tend to fail. When the caller is still using
    those defaults, prefer the Hermes runtime's configured default model and expose its
    base URL to LiteLLM/DSPy via env vars without overwriting user-provided values.
    """
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


def evolve(
    skill_name: str,
    iterations: int = 10,
    eval_source: str = "synthetic",
    dataset_path: Optional[str] = None,
    optimizer_model: str = DEFAULT_OPTIMIZER_MODEL,
    eval_model: str = DEFAULT_EVAL_MODEL,
    hermes_repo: Optional[str] = None,
    run_tests: bool = False,
    dry_run: bool = False,
    eval_backend: str = "dspy",
    run_tblite: bool = False,
    tblite_task_filter: Optional[str] = None,
    tblite_mode: str = "fast",
    execute_git_apply: bool = False,
    execute_push: bool = False,
    execute_pr: bool = False,
):
    """Main evolution function — orchestrates the full optimization loop."""

    load_default_env_files()
    optimizer_model, eval_model, runtime_env = resolve_runtime_model_settings(
        optimizer_model=optimizer_model,
        eval_model=eval_model,
    )

    config_kwargs = {
        "iterations": iterations,
        "optimizer_model": optimizer_model,
        "eval_model": eval_model,
        "judge_model": eval_model,
        "run_pytest": run_tests,
        "run_tblite": run_tblite,
    }
    if hermes_repo:
        config_kwargs["hermes_agent_path"] = Path(hermes_repo).expanduser()

    config = EvolutionConfig(**config_kwargs)

    console.print(f"\n[bold cyan]🧬 Hermes Agent Self-Evolution[/bold cyan] — Evolving skill: [bold]{skill_name}[/bold]\n")
    if runtime_env:
        env_list = ", ".join(f"{k}={v}" for k, v in runtime_env.items())
        console.print(f"  Runtime model alignment: applied {env_list}")

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
        console.print(f"\n[bold green]DRY RUN — setup validated successfully.[/bold green]")
        console.print(f"  Would generate eval dataset (source: {eval_source})")
        console.print(f"  Would run GEPA optimization ({iterations} iterations)")
        console.print(f"  Would evaluate holdout via backend: {eval_backend}")
        if run_tblite:
            console.print(
                f"  Would run TBLite {tblite_mode} regression gate "
                f"(task filter: {tblite_task_filter or 'mode default'})"
            )
        if execute_git_apply:
            console.print(
                f"  Would execute git apply flow (push={execute_push}, pr={execute_pr})"
            )
        console.print(f"  Would validate constraints and create PR")
        return

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

    console.print(f"\n[bold]Validating baseline constraints[/bold]")
    validator = ConstraintValidator(config)
    baseline_constraints = validate_skill_constraints(
        validator=validator,
        full_skill_text=skill["raw"],
    )
    all_pass = True
    for c in baseline_constraints:
        icon = "✓" if c.passed else "✗"
        color = "green" if c.passed else "red"
        console.print(f"  [{color}]{icon} {c.constraint_name}[/{color}]: {c.message}")
        if not c.passed:
            all_pass = False

    if not all_pass:
        console.print("[yellow]⚠ Baseline skill has constraint violations — proceeding anyway[/yellow]")

    console.print(f"\n[bold]Configuring optimizer[/bold]")
    console.print(f"  Optimizer: GEPA ({iterations} iterations)")
    console.print(f"  Optimizer model: {optimizer_model}")
    console.print(f"  Eval model: {eval_model}")
    console.print(f"  Holdout backend: {eval_backend}")

    lm = dspy.LM(eval_model)
    dspy.configure(lm=lm)

    baseline_module = SkillModule(skill["body"])
    trainset = dataset.to_dspy_examples("train")
    valset = dataset.to_dspy_examples("val")

    console.print(f"\n[bold cyan]Running GEPA optimization ({iterations} iterations)...[/bold cyan]\n")

    start_time = time.time()

    optimizer_name, optimized_module = optimize_skill_module(
        baseline_module=baseline_module,
        trainset=trainset,
        valset=valset,
        iterations=iterations,
        metric=skill_fitness_metric,
        optimizer_model=optimizer_model,
    )
    if optimizer_name != "GEPA":
        console.print(f"[yellow]Optimizer fallback in use: {optimizer_name}[/yellow]")

    elapsed = time.time() - start_time
    console.print(f"\n  Optimization completed in {elapsed:.1f}s")

    evolved_body = optimized_module.skill_text
    evolved_full = reassemble_skill(skill["frontmatter"], evolved_body)

    console.print(f"\n[bold]Validating evolved skill[/bold]")
    evolved_constraints = validate_skill_constraints(
        validator=validator,
        full_skill_text=evolved_full,
        baseline_full_text=skill["raw"],
    )
    all_pass = True
    for c in evolved_constraints:
        icon = "✓" if c.passed else "✗"
        color = "green" if c.passed else "red"
        console.print(f"  [{color}]{icon} {c.constraint_name}[/{color}]: {c.message}")
        if not c.passed:
            all_pass = False

    if not all_pass:
        console.print("[red]✗ Evolved skill FAILED constraints — not deploying[/red]")
        output_path = Path("output") / skill_name / "evolved_FAILED.md"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(evolved_full)
        console.print(f"  Saved failed variant to {output_path}")
        return

    console.print(f"\n[bold]Evaluating on holdout set ({len(dataset.holdout)} examples)[/bold]")

    baseline_scores, evolved_scores = evaluate_holdout(
        dataset=dataset,
        eval_backend=eval_backend,
        baseline_module=baseline_module,
        evolved_module=optimized_module,
        baseline_skill_body=skill["body"],
        evolved_skill_body=evolved_body,
        eval_model=eval_model,
        hermes_repo=str(config.hermes_agent_path),
        skill_name=skill_name,
    )

    avg_baseline = sum(baseline_scores) / max(1, len(baseline_scores))
    avg_evolved = sum(evolved_scores) / max(1, len(evolved_scores))
    improvement = avg_evolved - avg_baseline

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
    table.add_row("Eval Backend", "", eval_backend, "")

    tblite_gate_result = maybe_run_tblite_gate(
        run_tblite=run_tblite,
        skill_name=skill_name,
        baseline_skill_body=skill["body"],
        evolved_skill_body=evolved_body,
        hermes_repo=str(config.hermes_agent_path),
        tblite_regression_threshold=config.tblite_regression_threshold,
        tblite_task_filter=tblite_task_filter,
        tblite_mode=tblite_mode,
    )
    if tblite_gate_result is not None:
        gate_color = "green" if tblite_gate_result.passed else "red"
        table.add_row(
            "TBLite Gate",
            f"{tblite_gate_result.baseline_pass_rate:.3f}",
            f"{tblite_gate_result.evolved_pass_rate:.3f}",
            f"[{gate_color}]{tblite_gate_result.delta:+.3f}[/{gate_color}]",
        )

    console.print()
    console.print(table)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("output") / skill_name / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "evolved_skill.md").write_text(evolved_full)
    (output_dir / "baseline_skill.md").write_text(skill["raw"])
    baseline_skill_path = output_dir / "baseline_skill.md"
    evolved_skill_path = output_dir / "evolved_skill.md"

    metrics = {
        "skill_name": skill_name,
        "timestamp": timestamp,
        "iterations": iterations,
        "optimizer_model": optimizer_model,
        "eval_model": eval_model,
        "eval_backend": eval_backend,
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
        "tblite_gate": None if tblite_gate_result is None else {
            "passed": tblite_gate_result.passed,
            "mode": tblite_gate_result.mode,
            "task_filter": tblite_gate_result.task_filter,
            "base_config_path": str(tblite_gate_result.base_config_path),
            "baseline_pass_rate": tblite_gate_result.baseline_pass_rate,
            "evolved_pass_rate": tblite_gate_result.evolved_pass_rate,
            "delta": tblite_gate_result.delta,
            "threshold": tblite_gate_result.threshold,
            "summary": tblite_gate_result.summary,
        },
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    report_paths = write_evolution_report_artifacts(
        output_dir=output_dir,
        metrics=metrics,
        baseline_skill_path=baseline_skill_path,
        evolved_skill_path=evolved_skill_path,
    )
    pr_paths = write_evolution_pr_ready_artifacts(
        output_dir=output_dir,
        metrics=metrics,
        baseline_skill_path=baseline_skill_path,
        evolved_skill_path=evolved_skill_path,
    )
    github_pr_paths = write_evolution_github_pr_artifacts(
        output_dir=output_dir,
        metrics=metrics,
        baseline_skill_path=baseline_skill_path,
        evolved_skill_path=evolved_skill_path,
        report_path=report_paths["report_md"],
        summary_path=report_paths["summary_json"],
        diff_summary_path=pr_paths["diff_summary_md"],
        review_checklist_path=pr_paths["review_checklist_md"],
    )
    skill_relpath = str(skill_path.relative_to(config.hermes_agent_path / "skills"))
    git_pr_automation_paths = write_evolution_git_pr_automation_artifacts(
        output_dir=output_dir,
        metrics=metrics,
        hermes_repo=config.hermes_agent_path,
        skill_relpath=skill_relpath,
        evolved_skill_text=evolved_full,
        github_pr_body_path=github_pr_paths["github_pr_body_md"],
    )

    console.print(f"\n  Output saved to {output_dir}/")
    console.print(f"  Report: {report_paths['report_md']}")
    console.print(f"  Summary: {report_paths['summary_json']}")
    console.print(f"  PR Draft: {pr_paths['pr_draft_md']}")
    console.print(f"  Review Checklist: {pr_paths['review_checklist_md']}")
    console.print(f"  Diff Summary: {pr_paths['diff_summary_md']}")
    console.print(f"  GitHub PR Body: {github_pr_paths['github_pr_body_md']}")
    if github_pr_paths["gh_pr_create_command_txt"] is not None:
        console.print(f"  gh PR Command: {github_pr_paths['gh_pr_create_command_txt']}")
    else:
        console.print("  gh PR Command: skipped (decision=reject)")
    console.print(f"  Candidate Skill Patch: {git_pr_automation_paths['candidate_skill_file']}")
    console.print(f"  Git Apply Plan: {git_pr_automation_paths['git_apply_plan_sh']}")
    console.print(f"  Git Apply Guide: {git_pr_automation_paths['git_apply_plan_md']}")
    if git_pr_automation_paths["gh_pr_create_after_push_txt"] is not None:
        console.print(f"  gh PR After Push: {git_pr_automation_paths['gh_pr_create_after_push_txt']}")
    else:
        console.print("  gh PR After Push: skipped (decision=reject)")

    if tblite_gate_result is not None:
        gate_style = "bold green" if tblite_gate_result.passed else "bold red"
        gate_icon = "✓" if tblite_gate_result.passed else "✗"
        console.print(f"[{gate_style}]{gate_icon} {tblite_gate_result.summary}[/{gate_style}]")
        if not tblite_gate_result.passed:
            console.print("[red]✗ Benchmark regression gate failed — evolved skill should not be deployed[/red]")
            return

    if execute_git_apply and execute_pr and not execute_push:
        raise ValueError("execute_push must be True when execute_pr=True")

    if execute_git_apply:
        branch_name = build_evolution_branch_name(skill_name=skill_name, timestamp=timestamp)
        target_skill_path = build_target_skill_path(
            hermes_repo=config.hermes_agent_path,
            skill_relpath=skill_relpath,
        )
        git_apply_plan = {
            "hermes_repo": config.hermes_agent_path,
            "branch_name": branch_name,
            "target_skill_path": target_skill_path,
            "candidate_skill_file": git_pr_automation_paths["candidate_skill_file"],
            "commit_message": build_github_pr_title(metrics),
        }
        gh_pr_create_command = None
        if git_pr_automation_paths["gh_pr_create_after_push_txt"] is not None:
            gh_pr_create_command = Path(git_pr_automation_paths["gh_pr_create_after_push_txt"]).read_text().strip()

        execution_result = execute_evolution_git_pr_automation(
            git_apply_plan=git_apply_plan,
            gh_pr_create_command=gh_pr_create_command,
            execute_push=execute_push,
            execute_pr=execute_pr,
        )
        console.print(f"  Executed Git Apply Steps: {len(execution_result['git']['steps'])}")
        if execution_result["pr"] is not None:
            console.print("  Executed PR creation via gh")

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
@click.option("--eval-backend", default="dspy", type=click.Choice(["dspy", "hermes"]),
              help="Backend used for holdout evaluation")
@click.option("--dataset-path", default=None, help="Path to existing eval dataset (JSONL)")
@click.option("--optimizer-model", default=DEFAULT_OPTIMIZER_MODEL, help="Model for GEPA reflections")
@click.option("--eval-model", default=DEFAULT_EVAL_MODEL, help="Model for evaluations")
@click.option("--hermes-repo", default=None, help="Path to hermes-agent repo")
@click.option("--run-tests", is_flag=True, help="Run full pytest suite as constraint gate")
@click.option("--run-tblite", is_flag=True, help="Run the TBLite regression benchmark gate after holdout eval")
@click.option("--tblite-mode", default="fast", type=click.Choice(["fast", "full"]), help="TBLite gate mode: fast subset or full benchmark")
@click.option("--tblite-task-filter", default=None, help="Optional comma-separated TBLite task filter for a faster gate")
@click.option("--execute-git-apply", is_flag=True, help="Actually apply the evolved skill into the target Hermes repo branch")
@click.option("--execute-push", is_flag=True, help="When executing git apply, also push the branch to the remote")
@click.option("--execute-pr", is_flag=True, help="When executing git apply, also create the PR via gh after push")
@click.option("--dry-run", is_flag=True, help="Validate setup without running optimization")
def main(skill, iterations, eval_source, eval_backend, dataset_path, optimizer_model, eval_model, hermes_repo, run_tests, run_tblite, tblite_mode, tblite_task_filter, execute_git_apply, execute_push, execute_pr, dry_run):
    """Evolve a Hermes Agent skill using DSPy + GEPA optimization."""
    evolve(
        skill_name=skill,
        iterations=iterations,
        eval_source=eval_source,
        dataset_path=dataset_path,
        optimizer_model=optimizer_model,
        eval_model=eval_model,
        hermes_repo=hermes_repo,
        run_tests=run_tests,
        dry_run=dry_run,
        eval_backend=eval_backend,
        run_tblite=run_tblite,
        tblite_task_filter=tblite_task_filter,
        tblite_mode=tblite_mode,
        execute_git_apply=execute_git_apply,
        execute_push=execute_push,
        execute_pr=execute_pr,
    )


if __name__ == "__main__":
    main()
