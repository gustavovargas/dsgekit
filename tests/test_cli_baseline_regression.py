"""CLI tests for baseline regression suite command."""

from __future__ import annotations

import json
from pathlib import Path

from dsgekit.cli.main import main


def _baseline_path(fixtures_dir: Path) -> Path:
    return fixtures_dir / "baselines" / "ar1_linear_baseline.json"


def test_cli_baseline_regression_passes_and_writes_outputs(fixtures_dir, tmp_path: Path):
    dashboard_path = tmp_path / "dashboard.md"
    json_path = tmp_path / "report.json"

    rc = main(
        [
            "baseline_regression",
            "--baselines",
            str(fixtures_dir / "baselines"),
            "--dashboard",
            str(dashboard_path),
            "--json-output",
            str(json_path),
        ]
    )

    assert rc == 0
    assert dashboard_path.exists()
    assert json_path.exists()
    assert "Baseline Regression Dashboard" in dashboard_path.read_text(encoding="utf-8")

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert payload["n_failed_baselines"] == 0


def test_cli_baseline_regression_returns_nonzero_on_failure(fixtures_dir, models_dir, tmp_path: Path):
    with _baseline_path(fixtures_dir).open("r", encoding="utf-8") as f:
        baseline = json.load(f)

    baseline["solution"]["T"] = [[0.85]]
    baseline["model_path"] = str(models_dir / "ar1.mod")

    bad_baseline = tmp_path / "bad_baseline.json"
    bad_baseline.write_text(json.dumps(baseline), encoding="utf-8")

    dashboard_path = tmp_path / "dashboard.md"
    rc = main(
        [
            "baseline_regression",
            "--baselines",
            str(bad_baseline),
            "--dashboard",
            str(dashboard_path),
        ]
    )

    assert rc == 1
    assert dashboard_path.exists()
    assert "FAIL" in dashboard_path.read_text(encoding="utf-8")
