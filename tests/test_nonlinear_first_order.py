"""Tests for first-order stochastic solver on non-linear models."""

from __future__ import annotations

import numpy as np
import pytest

from dsgekit import load_model
from dsgekit.exceptions import SolverError
from dsgekit.io.formats.python_api import ModelBuilder
from dsgekit.solvers import (
    linearize_first_order,
    solve_first_order,
    solve_linear,
)
from dsgekit.transforms import linearize


@pytest.mark.parametrize("model_name", ["ar1.yaml", "nk.yaml", "rbc.mod"])
def test_first_order_linearization_matches_legacy_pipeline(models_dir, model_name):
    model, cal, ss = load_model(models_dir / model_name)

    legacy = linearize(model, ss, cal, eps=1e-6)
    approx = linearize_first_order(model, ss, cal, derivative_backend="numeric", eps=1e-6)

    np.testing.assert_allclose(approx.linear_system.A, legacy.A, atol=1e-8)
    np.testing.assert_allclose(approx.linear_system.B, legacy.B, atol=1e-8)
    np.testing.assert_allclose(approx.linear_system.C, legacy.C, atol=1e-8)
    np.testing.assert_allclose(approx.linear_system.D, legacy.D, atol=1e-8)


@pytest.mark.parametrize("model_name", ["ar1.yaml", "nk.yaml", "rbc.mod"])
def test_first_order_solution_matches_linear_solution(models_dir, model_name):
    model, cal, ss = load_model(models_dir / model_name)

    first_order = solve_first_order(model, ss, cal, derivative_backend="numeric", eps=1e-6)
    reference = solve_linear(linearize(model, ss, cal, eps=1e-6))

    np.testing.assert_allclose(first_order.T, reference.T, atol=1e-8)
    np.testing.assert_allclose(first_order.R, reference.R, atol=1e-8)
    np.testing.assert_allclose(first_order.eigenvalues, reference.eigenvalues, atol=1e-8)
    assert first_order.n_stable == reference.n_stable
    assert first_order.n_unstable == reference.n_unstable
    assert first_order.bk_meta["solver"] == "first_order_perturbation"
    assert first_order.bk_meta["derivative_backend"] == "numeric"


def test_first_order_raises_for_higher_order_timings():
    model, cal, ss = (
        ModelBuilder("lag2")
        .var("y")
        .varexo("e")
        .param("rho", 0.2)
        .equation("y = rho * y(-2) + e")
        .initval(y=0.0)
        .shock_stderr(e=0.01)
        .build()
    )
    with pytest.raises(SolverError, match="supports timings up to one lag/lead"):
        solve_first_order(model, ss, cal)
