"""Tests for second-order perturbation solver and pruning simulation."""

from __future__ import annotations

import numpy as np
import pytest

from dsgekit import load_model
from dsgekit.exceptions import SolverError
from dsgekit.io.formats.python_api import ModelBuilder
from dsgekit.simulate import (
    girf_pruned_second_order,
    irf,
    simulate,
    simulate_pruned_second_order,
    simulate_pruned_second_order_path,
)
from dsgekit.solvers import solve_second_order


def _build_quadratic_ar1_model(rho: float = 0.8, a: float = 0.2):
    return (
        ModelBuilder("quadratic_ar1")
        .var("y")
        .varexo("e")
        .param("rho", rho)
        .param("a", a)
        .equation("y = rho * y(-1) + a * y(-1)^2 + e")
        .initval(y=0.0)
        .shock_stderr(e=0.02)
        .build()
    )


def test_second_order_coefficients_match_quadratic_ar1():
    model, cal, ss = _build_quadratic_ar1_model(rho=0.8, a=0.2)
    solution = solve_second_order(model, ss, cal, derivative_backend="numeric", eps=1e-6)

    # y = rho*y(-1) + a*y(-1)^2 + e => d2y/dx2 = 2a = 0.4
    np.testing.assert_allclose(solution.T, np.array([[0.8]]), atol=1e-8)
    np.testing.assert_allclose(solution.R, np.array([[1.0]]), atol=1e-8)
    np.testing.assert_allclose(solution.quadratic_tensor[0, 0, 0], 0.4, atol=1e-5)
    np.testing.assert_allclose(solution.quadratic_tensor[0, 0, 1], 0.0, atol=1e-5)
    np.testing.assert_allclose(solution.quadratic_tensor[0, 1, 1], 0.0, atol=1e-5)


def test_pruned_deterministic_path_matches_closed_form():
    rho = 0.8
    a = 0.2
    x0 = 0.2
    periods = 10

    model, cal, ss = _build_quadratic_ar1_model(rho=rho, a=a)
    cal.set_shock_stderr("e", 0.0)
    solution = solve_second_order(model, ss, cal, derivative_backend="numeric", eps=1e-6)
    sim = simulate_pruned_second_order(
        solution,
        cal,
        n_periods=periods,
        initial_state=np.array([x0]),
        seed=42,
    )

    t = np.arange(1, periods + 1)
    x1 = (rho ** t) * x0
    x2 = a * (x0**2) * (rho ** (t - 1)) * (1.0 - rho**t) / (1.0 - rho)
    expected = x1 + x2

    np.testing.assert_allclose(sim["y"], expected, atol=1e-8)
    np.testing.assert_allclose(
        sim.first_order_component["y"].values,
        x1,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        sim.second_order_component["y"].values,
        x2,
        atol=1e-8,
    )


def test_pruned_simulation_stays_finite_for_strong_nonlinearity():
    model, cal, ss = _build_quadratic_ar1_model(rho=0.95, a=0.25)
    solution = solve_second_order(model, ss, cal, derivative_backend="numeric", eps=1e-6)

    sim = simulate_pruned_second_order(
        solution,
        cal,
        n_periods=1200,
        burn_in=200,
        seed=2026,
    )

    assert np.all(np.isfinite(sim.data.values))
    assert np.all(np.isfinite(sim.first_order_component.values))
    assert np.all(np.isfinite(sim.second_order_component.values))
    assert np.max(np.abs(sim.data.values)) < 1e4
    assert np.mean(np.abs(sim.second_order_component.values)) > 0.0


def test_second_order_reduces_to_linear_when_quadratic_term_zero(models_dir):
    model, cal, ss = load_model(models_dir / "ar1.yaml")
    solution = solve_second_order(model, ss, cal, derivative_backend="numeric", eps=1e-6)

    np.testing.assert_allclose(solution.quadratic_tensor, 0.0, atol=1e-8)

    sim_linear = simulate(solution.linear_solution, cal, n_periods=80, seed=77)
    sim_second = simulate_pruned_second_order(solution, cal, n_periods=80, seed=77)
    np.testing.assert_allclose(sim_second.data.values, sim_linear.data.values, atol=1e-10)


def test_second_order_raises_for_forward_looking_model(models_dir):
    model, cal, ss = load_model(models_dir / "nk.yaml")
    with pytest.raises(SolverError, match="backward-looking models with max_lead=0"):
        solve_second_order(model, ss, cal)


def test_simulation_path_api_matches_random_simulation_shocks():
    model, cal, ss = _build_quadratic_ar1_model(rho=0.85, a=0.15)
    solution = solve_second_order(model, ss, cal, derivative_backend="numeric", eps=1e-6)

    sim_random = simulate_pruned_second_order(solution, cal, n_periods=30, seed=123, burn_in=0)
    sim_path = simulate_pruned_second_order_path(solution, sim_random.shocks.values)

    np.testing.assert_allclose(sim_path.data.values, sim_random.data.values, atol=1e-12)
    np.testing.assert_allclose(
        sim_path.first_order_component.values,
        sim_random.first_order_component.values,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        sim_path.second_order_component.values,
        sim_random.second_order_component.values,
        atol=1e-12,
    )


def test_girf_matches_linear_irf_for_linear_model(models_dir):
    model, cal, ss = load_model(models_dir / "ar1.yaml")
    solution = solve_second_order(model, ss, cal, derivative_backend="numeric", eps=1e-6)

    periods = 10
    shock_size = 1.5
    girf_result = girf_pruned_second_order(
        solution,
        cal,
        "e",
        periods=periods,
        shock_size=shock_size,
        n_draws=5,
        seed=2026,
        initial_state=np.array([0.7]),
    )
    linear = irf(solution.linear_solution, "e", periods=periods, shock_size=shock_size)
    np.testing.assert_allclose(girf_result["y"], linear["y"], atol=1e-10)


def test_girf_is_state_dependent_in_quadratic_model():
    model, cal, ss = _build_quadratic_ar1_model(rho=0.8, a=0.2)
    solution = solve_second_order(model, ss, cal, derivative_backend="numeric", eps=1e-6)

    low_state = girf_pruned_second_order(
        solution,
        cal,
        "e",
        periods=8,
        shock_size=0.1,
        n_draws=800,
        seed=99,
        initial_state=np.array([0.0]),
    )
    high_state = girf_pruned_second_order(
        solution,
        cal,
        "e",
        periods=8,
        shock_size=0.1,
        n_draws=800,
        seed=99,
        initial_state=np.array([1.0]),
    )

    assert not np.allclose(low_state["y"], high_state["y"], atol=1e-6)
