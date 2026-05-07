"""Tests for benchmark regression gate helpers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from evolution.core import benchmark_gate as mod


@pytest.fixture
def fake_tblite_base_config(tmp_path: Path) -> Path:
    path = tmp_path / "local.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "env": {
                    "terminal_backend": "docker",
                    "use_wandb": False,
                    "data_dir_to_save_evals": "environments/benchmarks/evals/openthoughts-tblite-local",
                },
                "openai": {
                    "model_name": "anthropic/claude-sonnet-4",
                },
            }
        )
    )
    return path


def test_parse_tblite_pass_rate_from_stdout():
    output = "hello\nOverall Pass Rate: 0.4200 (42/100)\nbye\n"

    assert mod.parse_tblite_pass_rate(output) == pytest.approx(0.42)


def test_parse_tblite_pass_rate_raises_when_missing_summary():
    with pytest.raises(ValueError, match="Overall Pass Rate"):
        mod.parse_tblite_pass_rate("no summary here")


def test_run_tblite_gate_allows_small_regression(monkeypatch, tmp_path: Path, fake_tblite_base_config: Path):
    prompts = {
        "BASE": "baseline prompt",
        "EVOLVED": "evolved prompt",
    }
    calls = []

    def _fake_build_skill_system_prompt(skill_name, hermes_repo=None, skill_body_override=None):
        return prompts[skill_body_override], [skill_name]

    def _fake_run(command, capture_output, text, timeout, cwd):
        calls.append(command)
        config_path = Path(command[command.index("--config") + 1])
        config = yaml.safe_load(config_path.read_text())
        prompt = config["env"]["system_prompt"]
        score = 0.50 if prompt == "baseline prompt" else 0.49
        return SimpleNamespace(returncode=0, stdout=f"Overall Pass Rate: {score:.4f} (1/2)", stderr="")

    monkeypatch.setattr(mod, "build_skill_system_prompt", _fake_build_skill_system_prompt)
    monkeypatch.setattr(mod.subprocess, "run", _fake_run)

    result = mod.run_tblite_benchmark_gate(
        skill_name="github-code-review",
        baseline_skill_body="BASE",
        evolved_skill_body="EVOLVED",
        hermes_repo=tmp_path,
        regression_threshold=0.02,
        task_filter="broken-python,pandas-etl",
        base_config_path=fake_tblite_base_config,
        benchmark_script_path=tmp_path / "tblite_env.py",
    )

    assert result.passed is True
    assert result.baseline_pass_rate == pytest.approx(0.50)
    assert result.evolved_pass_rate == pytest.approx(0.49)
    assert result.delta == pytest.approx(-0.01)
    assert len(calls) == 2
    assert calls[0][-2:] == ["--env.task_filter", "broken-python,pandas-etl"]


def test_run_tblite_gate_fails_large_regression(monkeypatch, tmp_path: Path, fake_tblite_base_config: Path):
    monkeypatch.setattr(
        mod,
        "build_skill_system_prompt",
        lambda skill_name, hermes_repo=None, skill_body_override=None: (f"prompt::{skill_body_override}", [skill_name]),
    )

    def _fake_run(command, capture_output, text, timeout, cwd):
        config_path = Path(command[command.index("--config") + 1])
        config = yaml.safe_load(config_path.read_text())
        prompt = config["env"]["system_prompt"]
        score = 0.50 if prompt == "prompt::BASE" else 0.40
        return SimpleNamespace(returncode=0, stdout=f"Overall Pass Rate: {score:.4f} (1/2)", stderr="")

    monkeypatch.setattr(mod.subprocess, "run", _fake_run)

    result = mod.run_tblite_benchmark_gate(
        skill_name="github-code-review",
        baseline_skill_body="BASE",
        evolved_skill_body="EVOLVED",
        hermes_repo=tmp_path,
        regression_threshold=0.02,
        base_config_path=fake_tblite_base_config,
        benchmark_script_path=tmp_path / "tblite_env.py",
    )

    assert result.passed is False
    assert result.delta == pytest.approx(-0.10)
    assert "regression" in result.summary.lower()


def test_resolve_tblite_gate_plan_fast_uses_local_config_and_default_subset(tmp_path: Path):
    plan = mod.resolve_tblite_gate_plan(mode="fast", hermes_repo=tmp_path)

    assert plan.mode == "fast"
    assert plan.base_config_path == tmp_path / "environments/benchmarks/tblite/local.yaml"
    assert plan.task_filter == mod.DEFAULT_FAST_TBLITE_TASK_FILTER



def test_resolve_tblite_gate_plan_full_uses_default_config_without_filter(tmp_path: Path):
    plan = mod.resolve_tblite_gate_plan(mode="full", hermes_repo=tmp_path)

    assert plan.mode == "full"
    assert plan.base_config_path == tmp_path / "environments/benchmarks/tblite/default.yaml"
    assert plan.task_filter is None



def test_run_tblite_gate_records_mode_metadata(monkeypatch, tmp_path: Path, fake_tblite_base_config: Path):
    monkeypatch.setattr(
        mod,
        "build_skill_system_prompt",
        lambda skill_name, hermes_repo=None, skill_body_override=None: (f"prompt::{skill_body_override}", [skill_name]),
    )
    monkeypatch.setattr(
        mod,
        "resolve_tblite_gate_plan",
        lambda **kwargs: mod.TBLiteGatePlan(
            mode="fast",
            base_config_path=fake_tblite_base_config,
            task_filter="broken-python,pandas-etl",
            benchmark_script_path=tmp_path / "tblite_env.py",
        ),
    )

    def _fake_run(command, capture_output, text, timeout, cwd):
        return SimpleNamespace(returncode=0, stdout="Overall Pass Rate: 0.5000 (1/2)", stderr="")

    monkeypatch.setattr(mod.subprocess, "run", _fake_run)

    result = mod.run_tblite_benchmark_gate(
        skill_name="github-code-review",
        baseline_skill_body="BASE",
        evolved_skill_body="EVOLVED",
        hermes_repo=tmp_path,
        regression_threshold=0.02,
        mode="fast",
    )

    assert result.mode == "fast"
    assert result.task_filter == "broken-python,pandas-etl"
    assert result.base_config_path == fake_tblite_base_config
