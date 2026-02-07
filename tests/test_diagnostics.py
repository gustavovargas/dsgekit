"""Tests for Blanchard-Kahn diagnostics."""

from __future__ import annotations

import numpy as np
import pytest

from dsgekit import load_model
from dsgekit.exceptions import NoStableSolutionError
from dsgekit.solvers import (
    BKDiagnostics,
    EigenInfo,
    diagnose_bk,
    solve_linear,
)
from dsgekit.transforms import linearize

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ar1_solution(models_dir):
    """Solved AR(1) model (determinate)."""
    model, cal, ss = load_model(models_dir / "ar1.yaml")
    return solve_linear(linearize(model, ss, cal))


@pytest.fixture
def unstable_solution(models_dir):
    """AR(1) with rho=1.5 → explosive (check_bk=False to get solution)."""
    model, cal, ss = load_model(models_dir / "ar1.yaml")
    cal.set_parameter("rho", 1.5)
    lin = linearize(model, ss, cal)
    return solve_linear(lin, check_bk=False)


@pytest.fixture
def nk_solution(models_dir):
    model, cal, ss = load_model(models_dir / "nk.yaml")
    return solve_linear(linearize(model, ss, cal))


@pytest.fixture
def rbc_solution(models_dir):
    model, cal, ss = load_model(models_dir / "rbc.mod")
    return solve_linear(linearize(model, ss, cal))


# ---------------------------------------------------------------------------
# Determinate model
# ---------------------------------------------------------------------------


class TestDeterminate:
    def test_status(self, ar1_solution):
        diag = diagnose_bk(ar1_solution)
        assert diag.status == "determinate"
        assert diag.condition_met is True

    def test_counts(self, ar1_solution):
        diag = diagnose_bk(ar1_solution)
        assert diag.n_stable == 1
        assert diag.n_predetermined == 1
        assert diag.n_forward == 0

    def test_eigenvalue_classified(self, ar1_solution):
        diag = diagnose_bk(ar1_solution)
        assert len(diag.eigenvalues) >= 1
        eig = diag.eigenvalues[0]
        assert isinstance(eig, EigenInfo)
        assert eig.is_stable is True
        assert eig.is_real is True
        np.testing.assert_allclose(eig.modulus, 0.9, atol=1e-6)

    def test_summary_contains_keywords(self, ar1_solution):
        diag = diagnose_bk(ar1_solution)
        s = diag.summary()
        assert "DETERMINATE" in s
        assert "n_stable" in s
        assert "OK" in s
        assert "Trace ID" in s


# ---------------------------------------------------------------------------
# Explosive model
# ---------------------------------------------------------------------------


class TestNoStableSolution:
    def test_status(self, unstable_solution):
        diag = diagnose_bk(unstable_solution)
        assert diag.status == "no_stable_solution"
        assert diag.condition_met is False

    def test_eigenvalue_unstable(self, unstable_solution):
        diag = diagnose_bk(unstable_solution)
        # rho=1.5 → eigenvalue modulus = 1.5
        has_unstable = any(not e.is_stable for e in diag.eigenvalues)
        assert has_unstable

    def test_summary_contains_fail(self, unstable_solution):
        diag = diagnose_bk(unstable_solution)
        s = diag.summary()
        assert "NO STABLE SOLUTION" in s
        assert "FAIL" in s

    def test_message_has_recommendation(self, unstable_solution):
        diag = diagnose_bk(unstable_solution)
        assert "explosive" in diag.message.lower()


# ---------------------------------------------------------------------------
# Classification battery
# ---------------------------------------------------------------------------


class TestClassificationBattery:
    def test_nk_is_determinate(self, nk_solution):
        diag = diagnose_bk(nk_solution)
        assert nk_solution.n_stable == nk_solution.n_predetermined
        assert diag.status == "determinate"

    def test_rbc_is_determinate(self, rbc_solution):
        diag = diagnose_bk(rbc_solution)
        assert rbc_solution.n_stable == rbc_solution.n_predetermined
        assert diag.status == "determinate"

    def test_no_leads_explosive_raises_with_default_bk(self, models_dir):
        model, cal, ss = load_model(models_dir / "ar1.yaml")
        cal.set_parameter("rho", 1.5)
        lin = linearize(model, ss, cal)
        with pytest.raises(NoStableSolutionError):
            solve_linear(lin)

    def test_indeterminate_case_classified(self, models_dir):
        model, cal, ss = load_model(models_dir / "nk.yaml")
        cal.set_parameter("rho_i", 1.2)
        lin = linearize(model, ss, cal)
        solution = solve_linear(lin, check_bk=False)
        diag = diagnose_bk(solution)
        assert solution.n_stable > solution.n_predetermined
        assert diag.status == "indeterminate"


# ---------------------------------------------------------------------------
# Trace reproducibility
# ---------------------------------------------------------------------------


class TestTraceReproducibility:
    def test_trace_id_stable_across_runs(self, rbc_solution):
        diag1 = diagnose_bk(rbc_solution)
        diag2 = diagnose_bk(rbc_solution)
        assert diag1.trace_id == diag2.trace_id
        assert diag1.trace() == diag2.trace()

    def test_trace_contains_canonical_header(self, ar1_solution):
        diag = diagnose_bk(ar1_solution)
        t = diag.trace()
        assert t.startswith("BK_TRACE_ID=")
        assert "BK_TRACE_V1" in t
        assert "status=determinate" in t


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_is_dataclass(self, ar1_solution):
        diag = diagnose_bk(ar1_solution)
        assert isinstance(diag, BKDiagnostics)

    def test_eigenvalues_sorted_by_modulus(self, ar1_solution):
        diag = diagnose_bk(ar1_solution)
        moduli = [e.modulus for e in diag.eigenvalues]
        assert moduli == sorted(moduli)
