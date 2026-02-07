"""Unit tests for the AR(1) reference pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from dsgekit import load_model
from dsgekit.simulate import (
    compute_fevd,
    irf,
    moments,
    simulate,
    simulate_many,
)
from dsgekit.solvers import solve_linear
from dsgekit.transforms import linearize, to_state_space


@pytest.fixture
def ar1_setup(models_dir):
    """Load and solve AR(1) from YAML fixture."""
    model, cal, ss = load_model(models_dir / "ar1.yaml")
    solution = solve_linear(linearize(model, ss, cal))
    return model, cal, ss, solution


class TestLoadModel:
    def test_yaml_mod_consistency(self, models_dir):
        model_yaml, cal_yaml, ss_yaml = load_model(models_dir / "ar1.yaml")
        model_mod, cal_mod, ss_mod = load_model(models_dir / "ar1.mod")

        assert model_yaml.variable_names == model_mod.variable_names
        assert model_yaml.shock_names == model_mod.shock_names
        assert model_yaml.parameter_names == model_mod.parameter_names
        assert cal_yaml.parameters == cal_mod.parameters
        assert cal_yaml.shock_stderr == cal_mod.shock_stderr
        assert ss_yaml.values == ss_mod.values

    def test_dict_input(self):
        source = {
            "name": "AR1_dict",
            "variables": ["y"],
            "shocks": ["e"],
            "parameters": {"rho": 0.9},
            "equations": [{"name": "ar1", "expr": "y = rho * y(-1) + e"}],
            "steady_state": {"y": 0.0},
            "shocks_config": {"e": {"stderr": 0.01}},
        }
        model, cal, ss = load_model(source)

        assert model.n_equations == 1
        assert model.variable_names == ["y"]
        assert model.shock_names == ["e"]
        assert cal.parameters["rho"] == pytest.approx(0.9)
        assert cal.shock_stderr["e"] == pytest.approx(0.01)
        assert ss.values["y"] == pytest.approx(0.0)

    def test_unknown_extension_raises(self, tmp_path):
        p = tmp_path / "model.txt"
        p.write_text("not a model")

        with pytest.raises(ValueError, match="Cannot determine format"):
            load_model(p)


class TestSolveAndIRF:
    def test_ar1_solution_matrices(self, ar1_setup):
        _, _, _, solution = ar1_setup
        np.testing.assert_allclose(solution.T, np.array([[0.9]]), atol=1e-12)
        np.testing.assert_allclose(solution.R, np.array([[1.0]]), atol=1e-12)
        assert solution.n_stable == 1
        assert solution.n_unstable == 0
        assert solution.n_predetermined == 1

    def test_irf_geometric_decay(self, ar1_setup):
        _, _, _, solution = ar1_setup
        periods = 8
        result = irf(solution, "e", periods=periods)
        expected = 0.9 ** np.arange(periods)
        np.testing.assert_allclose(result["y"], expected, rtol=1e-12)


class TestSimulation:
    def test_reproducible_with_seed(self, ar1_setup):
        _, cal, _, solution = ar1_setup
        sim1 = simulate(solution, cal, n_periods=50, seed=123)
        sim2 = simulate(solution, cal, n_periods=50, seed=123)

        np.testing.assert_allclose(sim1.data.values, sim2.data.values)
        np.testing.assert_allclose(sim1.shocks.values, sim2.shocks.values)

    def test_zero_shock_variance_is_deterministic(self, ar1_setup):
        _, cal, _, solution = ar1_setup
        cal.set_shock_stderr("e", 0.0)
        sim = simulate(
            solution,
            cal,
            n_periods=6,
            seed=42,
            initial_state=np.array([1.0]),
        )
        expected = 0.9 ** np.arange(1, 7)
        np.testing.assert_allclose(sim["y"], expected, rtol=1e-12)
        np.testing.assert_allclose(sim.shocks["e"].values, 0.0, atol=1e-12)

    def test_simulate_many_uses_incremental_seeds(self, ar1_setup):
        _, cal, _, solution = ar1_setup
        sims = simulate_many(
            solution, cal, n_simulations=3, n_periods=10, seed=99, burn_in=0
        )

        assert len(sims) == 3
        np.testing.assert_allclose(
            sims[0].data.values,
            simulate(solution, cal, n_periods=10, seed=99, burn_in=0).data.values,
        )
        assert not np.allclose(sims[0].data.values, sims[1].data.values)


class TestMomentsAndStateSpace:
    def test_theoretical_variance_matches_closed_form(self, ar1_setup):
        _, cal, _, solution = ar1_setup
        result = moments(solution, cal, max_lag=5)
        expected = (0.01**2) / (1 - 0.9**2)
        assert result.variance["y"] == pytest.approx(expected, rel=1e-10)

    def test_autocorrelation_matches_rho_power(self, ar1_setup):
        _, cal, _, solution = ar1_setup
        result = moments(solution, cal, max_lag=5)
        expected = 0.9 ** np.arange(6)
        np.testing.assert_allclose(result.autocorrelation["y"].values, expected)

    def test_fevd_single_shock_is_100_percent(self, ar1_setup):
        _, cal, _, solution = ar1_setup
        fevd = compute_fevd(solution, cal, horizon=5)
        np.testing.assert_allclose(fevd["e"].values, 100.0, atol=1e-12)

    def test_to_state_space_ar1_matrices(self, ar1_setup):
        _, cal, _, solution = ar1_setup
        ss = to_state_space(solution, cal, observables=["y"])

        np.testing.assert_allclose(ss.T, np.array([[0.9]]), atol=1e-12)
        np.testing.assert_allclose(ss.Z, np.array([[1.0]]), atol=1e-12)
        np.testing.assert_allclose(ss.Q, np.array([[0.0001]]), atol=1e-12)
        np.testing.assert_allclose(ss.H, np.array([[0.0]]), atol=1e-12)
