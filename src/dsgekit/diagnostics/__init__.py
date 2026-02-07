"""Diagnostics helpers: BK diagnostics and linear baseline compatibility."""

from dsgekit.diagnostics.baseline_compat import (
    BaselineCompatReport,
    BaselineCompatSuiteReport,
    CompatCheck,
    expand_baseline_sources,
    run_linear_compat,
    run_linear_regression_suite,
)

__all__ = [
    "CompatCheck",
    "BaselineCompatReport",
    "BaselineCompatSuiteReport",
    "expand_baseline_sources",
    "run_linear_compat",
    "run_linear_regression_suite",
]
