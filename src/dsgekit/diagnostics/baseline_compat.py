"""Linear baseline compatibility harness.

Compares dsgekit linear solution and IRFs against stored reference baselines.
"""

from __future__ import annotations

import glob
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class CompatCheck:
    """Single compatibility check result."""

    name: str
    passed: bool
    message: str = ""
    max_abs_error: float | None = None
    max_rel_error: float | None = None
    atol: float | None = None
    rtol: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize check result."""
        return {
            "name": self.name,
            "passed": self.passed,
            "message": self.message,
            "max_abs_error": self.max_abs_error,
            "max_rel_error": self.max_rel_error,
            "atol": self.atol,
            "rtol": self.rtol,
        }


@dataclass
class BaselineCompatReport:
    """Compatibility report for one baseline run."""

    baseline_name: str
    model_name: str
    checks: list[CompatCheck] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def n_checks(self) -> int:
        return len(self.checks)

    @property
    def n_failed(self) -> int:
        return sum(1 for c in self.checks if not c.passed)

    @property
    def failed_checks(self) -> list[CompatCheck]:
        return [c for c in self.checks if not c.passed]

    @property
    def max_abs_error(self) -> float:
        values = [c.max_abs_error for c in self.checks if c.max_abs_error is not None]
        if not values:
            return 0.0
        return float(max(values))

    @property
    def max_rel_error(self) -> float:
        values = [c.max_rel_error for c in self.checks if c.max_rel_error is not None]
        if not values:
            return 0.0
        return float(max(values))

    def to_dict(self) -> dict[str, Any]:
        """Serialize report."""
        return {
            "baseline_name": self.baseline_name,
            "model_name": self.model_name,
            "passed": self.passed,
            "n_checks": self.n_checks,
            "n_failed": self.n_failed,
            "max_abs_error": self.max_abs_error,
            "max_rel_error": self.max_rel_error,
            "checks": [check.to_dict() for check in self.checks],
        }

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        lines = [
            "Linear Baseline Compatibility",
            "=" * 50,
            f"Baseline: {self.baseline_name}",
            f"Model:    {self.model_name}",
            f"Status:   {status}",
            f"Checks:   {self.n_checks} total, {self.n_failed} failed",
            "",
        ]
        for c in self.checks:
            mark = "OK" if c.passed else "FAIL"
            line = f"[{mark}] {c.name}"
            if c.max_abs_error is not None and c.max_rel_error is not None:
                line += (
                    f" | max_abs={c.max_abs_error:.3e}"
                    f" max_rel={c.max_rel_error:.3e}"
                )
            if c.message:
                line += f" | {c.message}"
            lines.append(line)
        return "\n".join(lines)


@dataclass
class BaselineCompatSuiteReport:
    """Compatibility report across multiple baselines."""

    reports: list[BaselineCompatReport] = field(default_factory=list)
    generated_at_utc: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"),
    )

    @property
    def passed(self) -> bool:
        return all(report.passed for report in self.reports)

    @property
    def n_baselines(self) -> int:
        return len(self.reports)

    @property
    def n_failed_baselines(self) -> int:
        return sum(1 for report in self.reports if not report.passed)

    @property
    def n_checks(self) -> int:
        return sum(report.n_checks for report in self.reports)

    @property
    def n_failed_checks(self) -> int:
        return sum(report.n_failed for report in self.reports)

    @property
    def max_abs_error(self) -> float:
        if not self.reports:
            return 0.0
        return float(max(report.max_abs_error for report in self.reports))

    @property
    def max_rel_error(self) -> float:
        if not self.reports:
            return 0.0
        return float(max(report.max_rel_error for report in self.reports))

    def to_dict(self) -> dict[str, Any]:
        """Serialize suite report."""
        return {
            "generated_at_utc": self.generated_at_utc,
            "passed": self.passed,
            "n_baselines": self.n_baselines,
            "n_failed_baselines": self.n_failed_baselines,
            "n_checks": self.n_checks,
            "n_failed_checks": self.n_failed_checks,
            "max_abs_error": self.max_abs_error,
            "max_rel_error": self.max_rel_error,
            "reports": [report.to_dict() for report in self.reports],
        }

    def summary(self) -> str:
        """Human-readable suite summary."""
        status = "PASS" if self.passed else "FAIL"
        lines = [
            "Linear Baseline Regression Suite",
            "=" * 50,
            f"Generated (UTC): {self.generated_at_utc}",
            f"Status:          {status}",
            f"Baselines:       {self.n_baselines} total, {self.n_failed_baselines} failed",
            f"Checks:          {self.n_checks} total, {self.n_failed_checks} failed",
            (
                "Worst error:     "
                f"max_abs={self.max_abs_error:.3e} max_rel={self.max_rel_error:.3e}"
            ),
            "",
        ]

        for report in self.reports:
            report_status = "PASS" if report.passed else "FAIL"
            lines.append(
                " - "
                f"{report.baseline_name} [{report_status}] "
                f"failed={report.n_failed}/{report.n_checks} "
                f"max_abs={report.max_abs_error:.3e} "
                f"max_rel={report.max_rel_error:.3e}"
            )
        return "\n".join(lines)

    def dashboard_markdown(self) -> str:
        """Build a compact Markdown dashboard of deviations."""
        lines = [
            "# Baseline Regression Dashboard",
            "",
            f"- Generated (UTC): `{self.generated_at_utc}`",
            f"- Overall status: `{'PASS' if self.passed else 'FAIL'}`",
            f"- Baselines: `{self.n_baselines}` (`{self.n_failed_baselines}` failed)",
            f"- Checks: `{self.n_checks}` (`{self.n_failed_checks}` failed)",
            (
                "- Worst deviations: "
                f"`max_abs={self.max_abs_error:.3e}`, "
                f"`max_rel={self.max_rel_error:.3e}`"
            ),
            "",
            "| Baseline | Model | Status | Failed Checks | Worst Abs | Worst Rel |",
            "|---|---|---|---:|---:|---:|",
        ]

        for report in self.reports:
            lines.append(
                "| "
                f"{report.baseline_name} | "
                f"{report.model_name} | "
                f"{'PASS' if report.passed else 'FAIL'} | "
                f"{report.n_failed}/{report.n_checks} | "
                f"{report.max_abs_error:.3e} | "
                f"{report.max_rel_error:.3e} |"
            )

        failed_reports = [report for report in self.reports if not report.passed]
        if failed_reports:
            lines.extend(["", "## Failed Checks", ""])
            for report in failed_reports:
                lines.append(f"### {report.baseline_name}")
                for check in report.failed_checks:
                    detail = f"- `{check.name}`"
                    if check.message:
                        detail += f": {check.message}"
                    elif (
                        check.max_abs_error is not None
                        and check.max_rel_error is not None
                    ):
                        detail += (
                            f" (`max_abs={check.max_abs_error:.3e}`, "
                            f"`max_rel={check.max_rel_error:.3e}`)"
                        )
                    lines.append(detail)
                lines.append("")

        return "\n".join(lines)


def _as_array(x: Any, *, name: str) -> NDArray[np.float64]:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array-like value")
    return arr


def _numeric_check(
    name: str,
    actual: NDArray[np.float64],
    expected: NDArray[np.float64],
    *,
    atol: float,
    rtol: float,
) -> CompatCheck:
    if actual.shape != expected.shape:
        return CompatCheck(
            name=name,
            passed=False,
            message=f"shape mismatch: actual {actual.shape}, expected {expected.shape}",
            atol=atol,
            rtol=rtol,
        )

    if actual.size == 0:
        return CompatCheck(name=name, passed=True, atol=atol, rtol=rtol)

    diff = np.abs(actual - expected)
    denom = np.maximum(np.abs(expected), 1e-15)
    max_abs = float(np.max(diff))
    max_rel = float(np.max(diff / denom))
    passed = bool(np.allclose(actual, expected, atol=atol, rtol=rtol))
    return CompatCheck(
        name=name,
        passed=passed,
        max_abs_error=max_abs,
        max_rel_error=max_rel,
        atol=atol,
        rtol=rtol,
    )


def _exact_check(name: str, actual: Any, expected: Any) -> CompatCheck:
    passed = actual == expected
    msg = ""
    if not passed:
        msg = f"actual={actual!r}, expected={expected!r}"
    return CompatCheck(name=name, passed=passed, message=msg)


def _load_baseline_dict(source: str | Path | dict[str, Any]) -> tuple[dict[str, Any], Path | None]:
    if isinstance(source, dict):
        return source, None

    path = Path(source)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data, path.parent


def _resolve_model_source(
    model_source: str | Path | dict[str, Any] | None,
    baseline: dict[str, Any],
    baseline_dir: Path | None,
) -> str | Path | dict[str, Any]:
    if model_source is not None:
        return model_source

    model_path = baseline.get("model_path")
    if model_path is None:
        raise ValueError("model_source is required when baseline has no 'model_path'")

    resolved = Path(model_path)
    if not resolved.is_absolute() and baseline_dir is not None:
        resolved = (baseline_dir / resolved).resolve()
    return resolved


def expand_baseline_sources(
    patterns: list[str | Path],
) -> list[Path]:
    """Expand baseline directories and glob patterns into JSON baseline paths."""
    paths: list[Path] = []
    seen: set[str] = set()

    for pattern in patterns:
        raw = str(pattern)
        p = Path(raw)

        matches: list[Path]
        if p.is_dir():
            matches = sorted(p.glob("*.json"))
        elif glob.has_magic(raw):
            matches = [Path(m) for m in sorted(glob.glob(raw))]
        else:
            matches = [p]

        for match in matches:
            key = str(match.resolve()) if match.exists() else str(match)
            if key in seen:
                continue
            seen.add(key)
            paths.append(match)

    return paths


def run_linear_compat(
    *,
    baseline_source: str | Path | dict[str, Any],
    model_source: str | Path | dict[str, Any] | None = None,
    periods: int | None = None,
) -> BaselineCompatReport:
    """Run linear compatibility checks against a baseline.

    Baseline schema:
    - ``name``: baseline label
    - ``model_path``: optional model path (used when ``model_source`` is None)
    - ``periods``: default IRF horizon
    - ``tolerances``: dict with ``matrix_atol``, ``matrix_rtol``, ``irf_atol``, ``irf_rtol``
    - ``solution``: includes ``var_names``, ``shock_names``, ``T``, ``R``, and BK counts
    - ``irfs``: mapping shock -> {"columns": [...], "data": [[...], ...]}
    """
    from dsgekit import load_model
    from dsgekit.simulate import irf
    from dsgekit.solvers import solve_linear
    from dsgekit.transforms import linearize

    baseline, baseline_dir = _load_baseline_dict(baseline_source)
    resolved_model_source = _resolve_model_source(model_source, baseline, baseline_dir)

    model, cal, ss = load_model(resolved_model_source)
    solution = solve_linear(linearize(model, ss, cal))

    baseline_name = str(baseline.get("name", "unnamed-baseline"))
    report = BaselineCompatReport(baseline_name=baseline_name, model_name=model.name)

    tol = baseline.get("tolerances", {})
    matrix_atol = float(tol.get("matrix_atol", 1e-8))
    matrix_rtol = float(tol.get("matrix_rtol", 1e-8))
    irf_atol = float(tol.get("irf_atol", 1e-8))
    irf_rtol = float(tol.get("irf_rtol", 1e-8))

    sol_ref = baseline["solution"]
    report.checks.append(_exact_check("solution.var_names", solution.var_names, sol_ref["var_names"]))
    report.checks.append(_exact_check("solution.shock_names", solution.shock_names, sol_ref["shock_names"]))
    report.checks.append(_exact_check("solution.n_stable", solution.n_stable, int(sol_ref["n_stable"])))
    report.checks.append(_exact_check("solution.n_unstable", solution.n_unstable, int(sol_ref["n_unstable"])))
    report.checks.append(
        _exact_check(
            "solution.n_predetermined",
            solution.n_predetermined,
            int(sol_ref["n_predetermined"]),
        )
    )

    T_ref = _as_array(sol_ref["T"], name="solution.T")
    R_ref = _as_array(sol_ref["R"], name="solution.R")
    report.checks.append(
        _numeric_check(
            "solution.T",
            solution.T,
            T_ref,
            atol=matrix_atol,
            rtol=matrix_rtol,
        )
    )
    report.checks.append(
        _numeric_check(
            "solution.R",
            solution.R,
            R_ref,
            atol=matrix_atol,
            rtol=matrix_rtol,
        )
    )

    irf_baseline = baseline.get("irfs", {})
    default_periods = int(baseline.get("periods", 40))
    horizon = int(periods if periods is not None else default_periods)

    for shock_name, ref_payload in irf_baseline.items():
        irf_actual = irf(solution, shock_name, periods=horizon).data.values
        irf_ref = _as_array(ref_payload["data"], name=f"irfs.{shock_name}.data")
        report.checks.append(
            _exact_check(
                f"irf.{shock_name}.columns",
                list(solution.var_names),
                list(ref_payload["columns"]),
            )
        )
        report.checks.append(
            _numeric_check(
                f"irf.{shock_name}.data",
                irf_actual,
                irf_ref,
                atol=irf_atol,
                rtol=irf_rtol,
            )
        )

    return report


def run_linear_regression_suite(
    *,
    baseline_sources: list[str | Path | dict[str, Any]],
    model_source: str | Path | dict[str, Any] | None = None,
    periods: int | None = None,
) -> BaselineCompatSuiteReport:
    """Run linear compatibility checks over a baseline set."""
    if not baseline_sources:
        raise ValueError("baseline_sources must contain at least one baseline")

    suite = BaselineCompatSuiteReport()
    for source in baseline_sources:
        try:
            report = run_linear_compat(
                baseline_source=source,
                model_source=model_source,
                periods=periods,
            )
        except Exception as err:
            if isinstance(source, dict):
                baseline_name = str(source.get("name", "unnamed-baseline"))
            else:
                baseline_name = str(source)
            report = BaselineCompatReport(
                baseline_name=baseline_name,
                model_name="unknown",
                checks=[
                    CompatCheck(
                        name="harness.execution",
                        passed=False,
                        message=str(err),
                    )
                ],
            )
        suite.reports.append(report)

    return suite
