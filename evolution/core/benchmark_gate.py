"""Helpers for running benchmark regression gates against a local Hermes checkout."""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import yaml

from evolution.core.hermes_eval import build_skill_system_prompt


DEFAULT_TBLITE_CONFIG = Path("environments/benchmarks/tblite/local.yaml")
DEFAULT_TBLITE_SCRIPT = Path("environments/benchmarks/tblite/tblite_env.py")
DEFAULT_FAST_TBLITE_TASK_FILTER = "broken-python,pandas-etl"


@dataclass
class TBLiteGatePlan:
    mode: str
    base_config_path: Path
    task_filter: str | None
    benchmark_script_path: Path


@dataclass
class TBLiteGateResult:
    passed: bool
    baseline_pass_rate: float
    evolved_pass_rate: float
    delta: float
    threshold: float
    summary: str
    baseline_stdout: str
    evolved_stdout: str
    mode: str
    task_filter: str | None
    base_config_path: Path


def parse_tblite_pass_rate(stdout: str) -> float:
    """Extract the overall pass rate from a TBLite/TB2 evaluation stdout string."""
    match = re.search(r"Overall Pass Rate:\s*([0-9]*\.?[0-9]+)", stdout)
    if not match:
        raise ValueError("Could not find 'Overall Pass Rate' in benchmark output")
    return float(match.group(1))


def resolve_tblite_gate_plan(
    *,
    mode: str,
    hermes_repo: str | Path,
    task_filter: str | None = None,
    base_config_path: str | Path | None = None,
    benchmark_script_path: str | Path | None = None,
) -> TBLiteGatePlan:
    """Resolve the concrete config/script/filter used for a TBLite gate run."""
    hermes_repo_path = Path(hermes_repo).expanduser().resolve()

    normalized_mode = mode.lower().strip()
    if normalized_mode not in {"fast", "full"}:
        raise ValueError(f"Unsupported TBLite gate mode: {mode}")

    default_config = (
        hermes_repo_path / DEFAULT_TBLITE_CONFIG
        if normalized_mode == "fast"
        else hermes_repo_path / Path("environments/benchmarks/tblite/default.yaml")
    )
    default_filter = DEFAULT_FAST_TBLITE_TASK_FILTER if normalized_mode == "fast" else None

    if base_config_path is None:
        resolved_config = default_config
    else:
        supplied = Path(base_config_path)
        resolved_config = supplied if supplied.is_absolute() else hermes_repo_path / supplied

    if benchmark_script_path is None:
        resolved_script = hermes_repo_path / DEFAULT_TBLITE_SCRIPT
    else:
        supplied_script = Path(benchmark_script_path)
        resolved_script = supplied_script if supplied_script.is_absolute() else hermes_repo_path / supplied_script

    return TBLiteGatePlan(
        mode=normalized_mode,
        base_config_path=resolved_config,
        task_filter=task_filter if task_filter is not None else default_filter,
        benchmark_script_path=resolved_script,
    )


def _write_benchmark_config(
    *,
    base_config_path: Path,
    system_prompt: str,
    output_dir: Path,
    label: str,
) -> Path:
    base_config = yaml.safe_load(base_config_path.read_text()) or {}
    env_cfg = dict(base_config.get("env") or {})
    env_cfg["system_prompt"] = system_prompt
    env_cfg["use_wandb"] = False
    env_cfg["data_dir_to_save_evals"] = str(output_dir / f"tblite-{label}")
    base_config["env"] = env_cfg

    config_path = output_dir / f"tblite_{label}.yaml"
    config_path.write_text(yaml.safe_dump(base_config, sort_keys=False))
    return config_path


def _run_tblite_eval(
    *,
    hermes_repo: Path,
    benchmark_script_path: Path,
    config_path: Path,
    task_filter: str | None,
) -> str:
    command = [
        shutil.which("python") or "python",
        str(benchmark_script_path),
        "evaluate",
        "--config",
        str(config_path),
    ]
    if task_filter:
        command.extend(["--env.task_filter", task_filter])

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=3600,
        cwd=str(hermes_repo),
    )
    if result.returncode != 0:
        raise RuntimeError(
            "TBLite evaluation failed with exit code "
            f"{result.returncode}: {result.stderr.strip() or result.stdout.strip()}"
        )
    return result.stdout


def run_tblite_benchmark_gate(
    *,
    skill_name: str,
    baseline_skill_body: str,
    evolved_skill_body: str,
    hermes_repo: str | Path,
    regression_threshold: float = 0.02,
    task_filter: str | None = None,
    base_config_path: str | Path | None = None,
    benchmark_script_path: str | Path | None = None,
    mode: str = "fast",
) -> TBLiteGateResult:
    """Run a TBLite regression gate against baseline and evolved skill prompts."""
    hermes_repo_path = Path(hermes_repo).expanduser().resolve()
    plan = resolve_tblite_gate_plan(
        mode=mode,
        hermes_repo=hermes_repo_path,
        task_filter=task_filter,
        base_config_path=base_config_path,
        benchmark_script_path=benchmark_script_path,
    )

    baseline_prompt, _ = build_skill_system_prompt(
        skill_name,
        hermes_repo=hermes_repo_path,
        skill_body_override=baseline_skill_body,
    )
    evolved_prompt, _ = build_skill_system_prompt(
        skill_name,
        hermes_repo=hermes_repo_path,
        skill_body_override=evolved_skill_body,
    )

    with tempfile.TemporaryDirectory(prefix="tblite-gate-") as tmpdir:
        output_dir = Path(tmpdir)
        baseline_config = _write_benchmark_config(
            base_config_path=plan.base_config_path,
            system_prompt=baseline_prompt,
            output_dir=output_dir,
            label="baseline",
        )
        evolved_config = _write_benchmark_config(
            base_config_path=plan.base_config_path,
            system_prompt=evolved_prompt,
            output_dir=output_dir,
            label="evolved",
        )

        baseline_stdout = _run_tblite_eval(
            hermes_repo=hermes_repo_path,
            benchmark_script_path=plan.benchmark_script_path,
            config_path=baseline_config,
            task_filter=plan.task_filter,
        )
        evolved_stdout = _run_tblite_eval(
            hermes_repo=hermes_repo_path,
            benchmark_script_path=plan.benchmark_script_path,
            config_path=evolved_config,
            task_filter=plan.task_filter,
        )

    baseline_pass_rate = parse_tblite_pass_rate(baseline_stdout)
    evolved_pass_rate = parse_tblite_pass_rate(evolved_stdout)
    delta = evolved_pass_rate - baseline_pass_rate
    passed = delta >= -regression_threshold

    summary = (
        f"TBLite {plan.mode} gate passed: baseline={baseline_pass_rate:.4f}, evolved={evolved_pass_rate:.4f}, "
        f"delta={delta:+.4f}, threshold=-{regression_threshold:.4f}"
        if passed
        else f"TBLite {plan.mode} regression detected: baseline={baseline_pass_rate:.4f}, evolved={evolved_pass_rate:.4f}, "
        f"delta={delta:+.4f}, threshold=-{regression_threshold:.4f}"
    )

    return TBLiteGateResult(
        passed=passed,
        baseline_pass_rate=baseline_pass_rate,
        evolved_pass_rate=evolved_pass_rate,
        delta=delta,
        threshold=regression_threshold,
        summary=summary,
        baseline_stdout=baseline_stdout,
        evolved_stdout=evolved_stdout,
        mode=plan.mode,
        task_filter=plan.task_filter,
        base_config_path=plan.base_config_path,
    )
