"""Tests for linear baseline compatibility harness."""

from __future__ import annotations

import json

from dsgekit.diagnostics import (
    expand_baseline_sources,
    run_linear_compat,
    run_linear_regression_suite,
)


def _baseline_path(fixtures_dir):
    return fixtures_dir / "baselines" / "ar1_linear_baseline.json"


def test_ar1_baseline_passes_with_explicit_model_source(fixtures_dir, models_dir):
    report = run_linear_compat(
        baseline_source=_baseline_path(fixtures_dir),
        model_source=models_dir / "ar1.mod",
    )

    assert report.passed
    assert report.n_failed == 0
    assert report.n_checks >= 6
    assert "PASS" in report.summary()


def test_ar1_baseline_resolves_model_path_from_baseline(fixtures_dir):
    report = run_linear_compat(
        baseline_source=_baseline_path(fixtures_dir),
    )

    assert report.passed
    assert report.n_failed == 0


def test_perturbed_baseline_detected_as_failure(fixtures_dir, models_dir):
    with _baseline_path(fixtures_dir).open("r", encoding="utf-8") as f:
        baseline = json.load(f)

    # Inject a deterministic mismatch to validate failure path.
    baseline["solution"]["T"] = [[0.85]]

    report = run_linear_compat(
        baseline_source=baseline,
        model_source=models_dir / "ar1.mod",
    )

    assert not report.passed
    failed = {c.name for c in report.failed_checks}
    assert "solution.T" in failed


def test_expand_baseline_sources_supports_directory(fixtures_dir):
    sources = expand_baseline_sources([fixtures_dir / "baselines"])
    assert len(sources) >= 1
    assert any(path.name == "ar1_linear_baseline.json" for path in sources)


def test_regression_suite_mixed_pass_fail(fixtures_dir, models_dir):
    baseline_path = _baseline_path(fixtures_dir)
    with baseline_path.open("r", encoding="utf-8") as f:
        perturbed = json.load(f)

    perturbed["name"] = "AR1 perturbed"
    perturbed["solution"]["T"] = [[0.85]]

    suite = run_linear_regression_suite(
        baseline_sources=[baseline_path, perturbed],
        model_source=models_dir / "ar1.mod",
    )

    assert not suite.passed
    assert suite.n_baselines == 2
    assert suite.n_failed_baselines == 1
    assert suite.n_failed_checks >= 1
    assert suite.max_abs_error > 0.0

    dashboard = suite.dashboard_markdown()
    assert "Baseline Regression Dashboard" in dashboard
    assert "| FAIL |" in dashboard
