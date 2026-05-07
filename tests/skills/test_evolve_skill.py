"""Tests for evolve_skill orchestration helpers."""

import os
from types import SimpleNamespace

from evolution.core.dataset_builder import EvalExample, EvalDataset
from evolution.skills import evolve_skill as mod


class FakeModule:
    def __init__(self, skill_text):
        self.skill_text = skill_text

    def __call__(self, task_input: str):
        return SimpleNamespace(output=f"base::{task_input}")


class FakeOptimizedModule:
    def __init__(self, skill_text):
        self.skill_text = skill_text

    def __call__(self, task_input: str):
        return SimpleNamespace(output=f"evolved::{task_input}")


def sample_dataset():
    return EvalDataset(
        train=[EvalExample(task_input="train task", expected_behavior="train rubric")],
        val=[EvalExample(task_input="val task", expected_behavior="val rubric")],
        holdout=[
            EvalExample(task_input="task one", expected_behavior="rubric one"),
            EvalExample(task_input="task two", expected_behavior="rubric two"),
        ],
    )


def test_evaluate_holdout_with_dspy_backend(monkeypatch):
    monkeypatch.setattr(mod, "skill_fitness_metric", lambda ex, pred: 0.8 if pred.output.startswith("base::") else 0.95)

    baseline, evolved = mod.evaluate_holdout(
        dataset=sample_dataset(),
        eval_backend="dspy",
        baseline_module=FakeModule("BASE"),
        evolved_module=FakeOptimizedModule("EVOLVED"),
        baseline_skill_body="BASE",
        evolved_skill_body="EVOLVED",
        eval_model="openai/gpt-4.1-mini",
        hermes_repo="/tmp/hermes",
    )

    assert baseline == [0.8, 0.8]
    assert evolved == [0.95, 0.95]



def test_evaluate_holdout_with_hermes_backend(monkeypatch):
    calls = []

    def _fake_run_skill_eval(case, **kwargs):
        calls.append({
            "skill_name": case.skill_name,
            "task_input": case.task_input,
            "skill_body_override": kwargs.get("skill_body_override"),
            "agent_kwargs": kwargs.get("agent_kwargs"),
        })
        return SimpleNamespace(final_response=f"resp::{case.task_input}", raw_result={})

    def score_by_body(skill_body):
        return 0.4 if skill_body == "BASE" else 0.9

    monkeypatch.setattr(mod, "run_skill_eval", _fake_run_skill_eval)
    monkeypatch.setattr(mod, "score_output_against_example", lambda **kwargs: score_by_body(kwargs["skill_body"]))

    baseline, evolved = mod.evaluate_holdout(
        dataset=sample_dataset(),
        eval_backend="hermes",
        baseline_module=FakeModule("BASE"),
        evolved_module=FakeOptimizedModule("EVOLVED"),
        baseline_skill_body="BASE",
        evolved_skill_body="EVOLVED",
        eval_model="openai/gpt-4.1-mini",
        hermes_repo="/tmp/hermes",
        skill_name="github-code-review",
    )

    assert baseline == [0.4, 0.4]
    assert evolved == [0.9, 0.9]
    assert calls == [
        {
            "skill_name": "github-code-review",
            "task_input": "task one",
            "skill_body_override": "BASE",
            "agent_kwargs": {"max_iterations": mod.DEFAULT_HERMES_EVAL_MAX_ITERATIONS},
        },
        {
            "skill_name": "github-code-review",
            "task_input": "task one",
            "skill_body_override": "EVOLVED",
            "agent_kwargs": {"max_iterations": mod.DEFAULT_HERMES_EVAL_MAX_ITERATIONS},
        },
        {
            "skill_name": "github-code-review",
            "task_input": "task two",
            "skill_body_override": "BASE",
            "agent_kwargs": {"max_iterations": mod.DEFAULT_HERMES_EVAL_MAX_ITERATIONS},
        },
        {
            "skill_name": "github-code-review",
            "task_input": "task two",
            "skill_body_override": "EVOLVED",
            "agent_kwargs": {"max_iterations": mod.DEFAULT_HERMES_EVAL_MAX_ITERATIONS},
        },
    ]



def test_maybe_run_tblite_gate_skips_when_disabled():
    result = mod.maybe_run_tblite_gate(
        run_tblite=False,
        skill_name="github-code-review",
        baseline_skill_body="BASE",
        evolved_skill_body="EVOLVED",
        hermes_repo="/tmp/hermes",
    )

    assert result is None



def test_maybe_run_tblite_gate_invokes_gate_when_enabled(monkeypatch):
    calls = []

    def _fake_gate(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(passed=True, delta=0.01)

    monkeypatch.setattr(mod, "run_tblite_benchmark_gate", _fake_gate)

    result = mod.maybe_run_tblite_gate(
        run_tblite=True,
        skill_name="github-code-review",
        baseline_skill_body="BASE",
        evolved_skill_body="EVOLVED",
        hermes_repo="/tmp/hermes",
        tblite_regression_threshold=0.03,
        tblite_task_filter="broken-python",
        tblite_mode="fast",
    )

    assert result.passed is True
    assert calls == [{
        "skill_name": "github-code-review",
        "baseline_skill_body": "BASE",
        "evolved_skill_body": "EVOLVED",
        "hermes_repo": "/tmp/hermes",
        "regression_threshold": 0.03,
        "task_filter": "broken-python",
        "mode": "fast",
    }]



def test_write_report_artifacts_delegates_to_report_module(tmp_path, monkeypatch):
    metrics = {"skill_name": "github-code-review"}
    called = {}

    def _fake_write_report_artifacts(**kwargs):
        called.update(kwargs)
        return {
            "report_md": tmp_path / "report.md",
            "summary_json": tmp_path / "summary.json",
        }

    monkeypatch.setattr(mod, "write_report_artifacts", _fake_write_report_artifacts)

    result = mod.write_evolution_report_artifacts(
        output_dir=tmp_path,
        metrics=metrics,
        baseline_skill_path=tmp_path / "baseline.md",
        evolved_skill_path=tmp_path / "evolved.md",
    )

    assert result["report_md"] == tmp_path / "report.md"
    assert called["metrics"] == metrics
    assert called["baseline_skill_path"] == tmp_path / "baseline.md"



def test_write_pr_ready_artifacts_delegates_to_report_module(tmp_path, monkeypatch):
    metrics = {"skill_name": "github-code-review"}
    called = {}

    def _fake_write_pr_ready_artifacts(**kwargs):
        called.update(kwargs)
        return {"pr_draft_md": tmp_path / "pr_draft.md"}

    monkeypatch.setattr(mod, "write_pr_ready_artifacts", _fake_write_pr_ready_artifacts)

    result = mod.write_evolution_pr_ready_artifacts(
        output_dir=tmp_path,
        metrics=metrics,
        baseline_skill_path=tmp_path / "baseline.md",
        evolved_skill_path=tmp_path / "evolved.md",
    )

    assert result["pr_draft_md"] == tmp_path / "pr_draft.md"
    assert called["evolved_skill_path"] == tmp_path / "evolved.md"



def test_write_github_pr_artifacts_delegates_to_report_module(tmp_path, monkeypatch):
    metrics = {"skill_name": "github-code-review"}
    called = {}

    def _fake_write_github_pr_artifacts(**kwargs):
        called.update(kwargs)
        return {"github_pr_body_md": tmp_path / "github_pr_body.md"}

    monkeypatch.setattr(mod, "write_github_pr_artifacts", _fake_write_github_pr_artifacts)

    result = mod.write_evolution_github_pr_artifacts(
        output_dir=tmp_path,
        metrics=metrics,
        baseline_skill_path=tmp_path / "baseline.md",
        evolved_skill_path=tmp_path / "evolved.md",
        report_path=tmp_path / "report.md",
        summary_path=tmp_path / "summary.json",
        diff_summary_path=tmp_path / "diff_summary.md",
        review_checklist_path=tmp_path / "review_checklist.md",
    )

    assert result["github_pr_body_md"] == tmp_path / "github_pr_body.md"
    assert called["report_path"] == tmp_path / "report.md"
    assert called["review_checklist_path"] == tmp_path / "review_checklist.md"



def test_write_git_pr_automation_artifacts_delegates_to_automation_module(tmp_path, monkeypatch):
    metrics = {"skill_name": "github-code-review"}
    called = {}

    def _fake_write_git_pr_automation_artifacts(**kwargs):
        called.update(kwargs)
        return {"git_apply_plan_sh": tmp_path / "git_apply_plan.sh"}

    monkeypatch.setattr(mod, "write_git_pr_automation_artifacts", _fake_write_git_pr_automation_artifacts)

    result = mod.write_evolution_git_pr_automation_artifacts(
        output_dir=tmp_path,
        metrics=metrics,
        hermes_repo=tmp_path / "hermes-agent",
        skill_relpath="github/github-code-review/SKILL.md",
        evolved_skill_text="# evolved\n",
        github_pr_body_path=tmp_path / "github_pr_body.md",
    )

    assert result["git_apply_plan_sh"] == tmp_path / "git_apply_plan.sh"
    assert called["skill_relpath"] == "github/github-code-review/SKILL.md"
    assert called["github_pr_body_path"] == tmp_path / "github_pr_body.md"



def test_execute_evolution_git_pr_automation_delegates_to_execution_module(tmp_path, monkeypatch):
    called = {}

    def _fake_execute_git_pr_automation(**kwargs):
        called.update(kwargs)
        return {"git": {"steps": []}, "pr": None}

    monkeypatch.setattr(mod, "execute_git_pr_automation", _fake_execute_git_pr_automation)

    result = mod.execute_evolution_git_pr_automation(
        git_apply_plan={"hermes_repo": tmp_path / "hermes-agent"},
        gh_pr_create_command="gh pr create --draft",
        execute_push=True,
        execute_pr=False,
    )

    assert result["git"]["steps"] == []
    assert called["execute_push"] is True
    assert called["execute_pr"] is False


def test_git_apply_is_skipped_when_tblite_gate_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "load_default_env_files", lambda: [])
    monkeypatch.setattr(mod, "resolve_runtime_model_settings", lambda **kwargs: (
        kwargs["optimizer_model"],
        kwargs["eval_model"],
        {},
    ))
    monkeypatch.setattr(
        mod,
        "find_skill",
        lambda skill_name, hermes_repo: tmp_path / "hermes-agent" / "skills" / skill_name / "SKILL.md",
    )
    monkeypatch.setattr(mod, "load_skill", lambda path: {
        "raw": "---\nname: dogfood\ndescription: test\n---\n\n# Body",
        "body": "# Body",
        "frontmatter": {"name": "dogfood", "description": "test"},
        "name": "dogfood",
        "description": "test",
    })

    dataset = sample_dataset()
    monkeypatch.setattr(mod.GoldenDatasetLoader, "load", lambda path: dataset)

    class FakeValidator:
        def validate_all(self, artifact_text, artifact_type, baseline_text=None):
            return [SimpleNamespace(passed=True, constraint_name="ok", message="ok")]

    monkeypatch.setattr(mod, "ConstraintValidator", lambda config: FakeValidator())
    monkeypatch.setattr(mod.dspy, "LM", lambda model: object())
    monkeypatch.setattr(mod.dspy, "configure", lambda **kwargs: None)
    monkeypatch.setattr(mod, "SkillModule", lambda text: FakeModule(text))
    monkeypatch.setattr(mod, "optimize_skill_module", lambda **kwargs: ("GEPA", FakeOptimizedModule("# Evolved")))
    monkeypatch.setattr(mod, "evaluate_holdout", lambda **kwargs: ([0.4, 0.4], [0.9, 0.9]))
    monkeypatch.setattr(
        mod,
        "maybe_run_tblite_gate",
        lambda **kwargs: SimpleNamespace(
            passed=False,
            baseline_pass_rate=0.6,
            evolved_pass_rate=0.4,
            delta=-0.2,
            threshold=0.02,
            summary="TBLite fast regression detected",
            mode="fast",
            task_filter="broken-python",
            base_config_path=tmp_path / "local.yaml",
        ),
    )
    monkeypatch.setattr(mod, "write_evolution_report_artifacts", lambda **kwargs: {
        "report_md": tmp_path / "report.md",
        "summary_json": tmp_path / "summary.json",
    })
    monkeypatch.setattr(mod, "write_evolution_pr_ready_artifacts", lambda **kwargs: {
        "pr_draft_md": tmp_path / "pr_draft.md",
        "review_checklist_md": tmp_path / "review_checklist.md",
        "diff_summary_md": tmp_path / "diff_summary.md",
    })
    monkeypatch.setattr(mod, "write_evolution_github_pr_artifacts", lambda **kwargs: {
        "github_pr_body_md": tmp_path / "github_pr_body.md",
        "gh_pr_create_command_txt": tmp_path / "gh_pr_create_command.txt",
    })
    monkeypatch.setattr(mod, "write_evolution_git_pr_automation_artifacts", lambda **kwargs: {
        "candidate_skill_file": tmp_path / "candidate_skill_patch.md",
        "git_apply_plan_sh": tmp_path / "git_apply_plan.sh",
        "git_apply_plan_md": tmp_path / "git_apply_plan.md",
        "gh_pr_create_after_push_txt": None,
    })

    execute_calls = []
    monkeypatch.setattr(mod, "execute_evolution_git_pr_automation", lambda **kwargs: execute_calls.append(kwargs))

    mod.evolve(
        skill_name="dogfood",
        eval_source="golden",
        dataset_path=str(tmp_path / "dataset"),
        hermes_repo=str(tmp_path / "hermes-agent"),
        eval_backend="hermes",
        run_tblite=True,
        execute_git_apply=True,
        execute_push=True,
        execute_pr=False,
    )

    assert execute_calls == []


def test_load_env_file_sets_missing_values_without_overwriting_existing(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text(
        "# comment\n"
        "OPENAI_API_KEY=loaded-key\n"
        "export OPENAI_BASE_URL='https://example.test/v1'\n"
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://keep-me")

    loaded = mod.load_env_file(env_path)

    assert loaded == {"OPENAI_API_KEY": "loaded-key"}
    assert os.environ["OPENAI_API_KEY"] == "loaded-key"
    assert os.environ["OPENAI_BASE_URL"] == "https://keep-me"



def test_load_default_env_files_prefers_home_hermes_env(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    env_path = hermes_home / ".env"
    env_path.write_text("OPENAI_API_KEY=from-hermes-home\n")
    monkeypatch.setattr(mod.Path, "home", lambda: tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    loaded_paths = mod.load_default_env_files()

    assert env_path in loaded_paths
    assert os.environ["OPENAI_API_KEY"] == "from-hermes-home"



def test_resolve_runtime_model_settings_uses_custom_hermes_config_defaults(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "model:\n"
        "  provider: custom\n"
        "  default: gpt-5.4\n"
        "  base_url: https://custom.example/v1\n"
    )
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)

    optimizer_model, eval_model, applied_env = mod.resolve_runtime_model_settings(
        optimizer_model="openai/gpt-4.1",
        eval_model="openai/gpt-4.1-mini",
        config_path=config_path,
    )

    assert optimizer_model == "gpt-5.4"
    assert eval_model == "gpt-5.4"
    assert applied_env == {
        "OPENAI_BASE_URL": "https://custom.example/v1",
        "OPENAI_API_BASE": "https://custom.example/v1",
    }
    assert os.environ["OPENAI_BASE_URL"] == "https://custom.example/v1"
    assert os.environ["OPENAI_API_BASE"] == "https://custom.example/v1"



def test_resolve_runtime_model_settings_preserves_explicit_models_and_env(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "model:\n"
        "  provider: custom\n"
        "  default: gpt-5.4\n"
        "  base_url: https://custom.example/v1\n"
    )
    monkeypatch.setenv("OPENAI_BASE_URL", "https://keep.example/v1")
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)

    optimizer_model, eval_model, applied_env = mod.resolve_runtime_model_settings(
        optimizer_model="anthropic/claude-sonnet-4.6",
        eval_model="google/gemini-3-flash",
        config_path=config_path,
    )

    assert optimizer_model == "anthropic/claude-sonnet-4.6"
    assert eval_model == "google/gemini-3-flash"
    assert applied_env == {"OPENAI_API_BASE": "https://custom.example/v1"}
    assert os.environ["OPENAI_BASE_URL"] == "https://keep.example/v1"
    assert os.environ["OPENAI_API_BASE"] == "https://custom.example/v1"



def test_normalized_reflection_lm_preserves_list_shape_for_none_outputs():
    lm = mod.NormalizedReflectionLM(lambda prompt: [None])

    assert lm("prompt") == [""]



def test_normalized_reflection_lm_wraps_scalar_outputs_as_single_item_list():
    assert mod.NormalizedReflectionLM(lambda prompt: "hello")("prompt") == ["hello"]
    assert mod.NormalizedReflectionLM(lambda prompt: {"text": "hello"})("prompt") == [{"text": "hello"}]



def test_create_gepa_optimizer_uses_legacy_max_steps_when_supported(monkeypatch):
    called = {}

    class FakeLegacyGEPA:
        def __init__(self, **kwargs):
            called.update(kwargs)

    monkeypatch.setattr(mod.dspy, "GEPA", FakeLegacyGEPA)
    monkeypatch.setattr(mod.inspect, "signature", lambda obj: SimpleNamespace(parameters={"metric": object(), "max_steps": object()}))

    optimizer = mod.create_gepa_optimizer(metric="metric", iterations=7)

    assert isinstance(optimizer, FakeLegacyGEPA)
    assert called == {"metric": "metric", "max_steps": 7}



def test_create_gepa_optimizer_adapts_to_new_signature(monkeypatch):
    called = {}

    class FakeModernGEPA:
        def __init__(self, **kwargs):
            called.update(kwargs)

    class FakeLM:
        def __init__(self, model):
            self.model = model

        def __call__(self, prompt):
            return ["normalized-text"]

    monkeypatch.setattr(mod.dspy, "GEPA", FakeModernGEPA)
    monkeypatch.setattr(mod.dspy, "LM", FakeLM)
    monkeypatch.setattr(
        mod.inspect,
        "signature",
        lambda obj: SimpleNamespace(parameters={
            "metric": object(),
            "reflection_lm": object(),
            "max_full_evals": object(),
        }),
    )

    optimizer = mod.create_gepa_optimizer(metric="metric", iterations=3, optimizer_model="openai/gpt-4.1")

    assert isinstance(optimizer, FakeModernGEPA)
    assert called["metric"] == "metric"
    assert called["max_full_evals"] == 3
    assert called["reflection_lm"]("prompt") == ["normalized-text"]



def test_create_gepa_optimizer_wraps_legacy_metric_for_new_gepa(monkeypatch):
    called = {}

    class FakeModernGEPA:
        def __init__(self, **kwargs):
            called.update(kwargs)

    def _metric(example, prediction, trace=None):
        return 0.42

    class FakeLM:
        def __init__(self, model):
            self.model = model

        def __call__(self, prompt):
            return [f"lm::{self.model}"]

    monkeypatch.setattr(mod.dspy, "GEPA", FakeModernGEPA)
    monkeypatch.setattr(mod.dspy, "LM", FakeLM)

    def _fake_signature(obj):
        if obj is FakeModernGEPA:
            return SimpleNamespace(parameters={
                "metric": object(),
                "reflection_lm": object(),
                "max_full_evals": object(),
            })
        if obj is _metric:
            return SimpleNamespace(parameters={"example": object(), "prediction": object(), "trace": object()})
        raise AssertionError(f"unexpected object: {obj}")

    monkeypatch.setattr(mod.inspect, "signature", _fake_signature)

    mod.create_gepa_optimizer(metric=_metric, iterations=2, optimizer_model="openai/gpt-4.1")

    assert called["reflection_lm"]("prompt") == ["lm::openai/gpt-4.1"]
    assert called["max_full_evals"] == 2
    assert called["metric"]("gold", "pred", "trace", "pred_name", "pred_trace") == 0.42



def test_optimize_skill_module_falls_back_to_miprov2_when_gepa_fails(monkeypatch):
    calls = []

    def _fake_create_gepa_optimizer(**kwargs):
        raise TypeError("old/new API mismatch")

    class FakeMIPRO:
        def __init__(self, **kwargs):
            calls.append(("mipro_init", kwargs))

        def compile(self, baseline_module, **kwargs):
            calls.append(("mipro_compile", kwargs))
            return "optimized-module"

    monkeypatch.setattr(mod, "create_gepa_optimizer", _fake_create_gepa_optimizer)
    monkeypatch.setattr(mod.dspy, "MIPROv2", FakeMIPRO)

    optimizer_name, optimized = mod.optimize_skill_module(
        baseline_module="baseline",
        trainset=["train"],
        valset=["val"],
        iterations=2,
        metric="metric",
    )

    assert optimizer_name == "MIPROv2"
    assert optimized == "optimized-module"
    assert calls == [
        ("mipro_init", {"metric": "metric", "auto": "light"}),
        ("mipro_compile", {"trainset": ["train"], "valset": ["val"]}),
    ]


def test_validate_skill_constraints_uses_full_skill_text_for_structure(monkeypatch):
    calls = []

    class FakeValidator:
        def validate_all(self, artifact_text, artifact_type, baseline_text=None):
            calls.append((artifact_text, artifact_type, baseline_text))
            return [SimpleNamespace(passed=True, constraint_name="skill_structure", message="ok")]

    result = mod.validate_skill_constraints(
        validator=FakeValidator(),
        full_skill_text="---\nname: dogfood\ndescription: d\n---\n\n# Body",
        baseline_full_text="---\nname: base\ndescription: d\n---\n\n# Base",
    )

    assert result[0].passed is True
    assert calls == [(
        "---\nname: dogfood\ndescription: d\n---\n\n# Body",
        "skill",
        "---\nname: base\ndescription: d\n---\n\n# Base",
    )]
