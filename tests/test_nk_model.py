"""Tests for the reference New Keynesian (NK) 3-equation model fixtures."""

from __future__ import annotations

import numpy as np
import pytest

from dsgekit import load_model
from dsgekit.simulate import irf
from dsgekit.solvers import diagnose_bk, solve_linear
from dsgekit.transforms import linearize


@pytest.fixture(params=["nk.yaml", "nk.mod"])
def nk_model_data(models_dir, request):
    """Load NK fixture in both supported formats."""
    model_path = models_dir / request.param
    model, cal, ss = load_model(model_path)
    return model, cal, ss


class TestNKFixtureStructure:
    def test_dimensions_and_symbols(self, nk_model_data):
        model, cal, ss = nk_model_data

        assert model.n_equations == 3
        assert model.n_variables == 3
        assert model.n_shocks == 3
        assert model.n_parameters == 6

        assert model.variable_names == ["x", "pi", "i"]
        assert model.shock_names == ["e_d", "e_s", "e_m"]
        assert model.parameter_names == [
            "sigma", "beta", "kappa", "phi_pi", "phi_x", "rho_i"
        ]
        assert set(ss.values.keys()) == {"x", "pi", "i"}

    def test_lead_lag_structure(self, nk_model_data):
        model, *_ = nk_model_data
        assert model.predetermined_variable_names == ["i"]
        assert model.forward_looking_variable_names == ["x", "pi"]
        assert model.n_predetermined == 1
        assert model.n_forward_looking == 2


class TestNKLinearizationAndSolve:
    def test_linearization_has_forward_and_backward_terms(self, nk_model_data):
        model, cal, ss = nk_model_data
        lin = linearize(model, ss, cal)

        # Only i appears with lag (A column 2).
        lag_cols = np.where(np.any(np.abs(lin.A) > 1e-12, axis=0))[0]
        # x and pi appear with leads (C columns 0 and 1).
        lead_cols = np.where(np.any(np.abs(lin.C) > 1e-12, axis=0))[0]

        np.testing.assert_array_equal(lag_cols, np.array([2]))
        np.testing.assert_array_equal(lead_cols, np.array([0, 1]))

    def test_default_bk_check_solves(self, nk_model_data):
        model, cal, ss = nk_model_data
        lin = linearize(model, ss, cal)
        solution = solve_linear(lin)
        assert solution.n_stable == 1
        assert solution.n_predetermined == 1

    def test_solution_and_irfs_with_bk_check_disabled(self, nk_model_data):
        model, cal, ss = nk_model_data
        lin = linearize(model, ss, cal)
        solution = solve_linear(lin, check_bk=False)
        diag = diagnose_bk(solution)

        assert solution.T.shape == (3, 3)
        assert solution.R.shape == (3, 3)
        assert solution.n_stable == 1
        assert diag.status == "determinate"

        for shock in model.shock_names:
            result = irf(solution, shock, periods=12)
            assert result.data.shape == (12, 3)
            assert list(result.data.columns) == model.variable_names
            assert np.all(np.isfinite(result.data.values))
