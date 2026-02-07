"""Tests for forecasting and historical decomposition."""

from __future__ import annotations

import numpy as np
import pytest

from dsgekit import load_model
from dsgekit.exceptions import FilterError
from dsgekit.filters import (
    ForecastResult,
    HistoricalDecompositionResult,
    forecast,
    historical_decomposition,
    kalman_filter,
    kalman_smoother,
)
from dsgekit.simulate import simulate
from dsgekit.solvers import solve_linear
from dsgekit.transforms import linearize, to_state_space


@pytest.fixture
def ar1_filter_smoother(models_dir):
    model, cal, ss = load_model(models_dir / "ar1.yaml")
    solution = solve_linear(linearize(model, ss, cal))
    state_space = to_state_space(solution, cal, observables=["y"])
    data = simulate(solution, cal, n_periods=180, seed=2033).data[["y"]]
    kf = kalman_filter(state_space, data)
    sm = kalman_smoother(state_space, kf)
    return state_space, kf, sm


class TestForecast:
    def test_shapes_and_columns(self, ar1_filter_smoother):
        ss, kf, sm = ar1_filter_smoother
        result = forecast(ss, kf, steps=12, smoother_result=sm)
        assert isinstance(result, ForecastResult)
        assert result.in_sample_observables.shape == (kf.n_periods, ss.n_obs)
        assert result.out_of_sample_states.shape == (12, ss.n_states)
        assert result.out_of_sample_observables.shape == (12, ss.n_obs)
        assert result.out_of_sample_state_cov.shape == (12, ss.n_states, ss.n_states)
        assert result.out_of_sample_obs_cov.shape == (12, ss.n_obs, ss.n_obs)
        assert list(result.out_of_sample_observables.columns) == ["y"]

    def test_in_sample_matches_predicted_mapping(self, ar1_filter_smoother):
        ss, kf, sm = ar1_filter_smoother
        result = forecast(ss, kf, steps=5, smoother_result=sm, include_steady_state=True)
        expected = (kf.predicted_states.values @ ss.Z.T).reshape(-1)
        np.testing.assert_allclose(result.in_sample_observables["y"].values, expected, atol=1e-12)

    def test_out_of_sample_ar1_recursion_from_filtered(self, ar1_filter_smoother):
        ss, kf, _ = ar1_filter_smoother
        result = forecast(ss, kf, steps=8, smoother_result=None, include_steady_state=False)
        y0 = float(kf.filtered_states["y"].iloc[-1])
        expected = y0 * (0.9 ** np.arange(1, 9))
        np.testing.assert_allclose(result.out_of_sample_observables["y"].values, expected, atol=1e-10)

    def test_intervals_have_valid_order(self, ar1_filter_smoother):
        ss, kf, sm = ar1_filter_smoother
        result = forecast(ss, kf, steps=6, smoother_result=sm)
        interval = result.intervals(alpha=0.1)
        assert ("y", "lower") in interval.columns
        assert ("y", "upper") in interval.columns
        assert np.all(interval[("y", "lower")] <= interval[("y", "upper")])

    def test_invalid_alpha_raises(self, ar1_filter_smoother):
        ss, kf, sm = ar1_filter_smoother
        result = forecast(ss, kf, steps=3, smoother_result=sm)
        with pytest.raises(FilterError, match="alpha"):
            _ = result.intervals(alpha=1.2)


class TestHistoricalDecomposition:
    def test_reconstructs_smoothed_states_and_observables(self, ar1_filter_smoother):
        ss, _, sm = ar1_filter_smoother
        result = historical_decomposition(ss, sm)
        assert isinstance(result, HistoricalDecompositionResult)
        assert result.max_abs_state_error < 1e-8
        assert result.max_abs_obs_error < 1e-8
        np.testing.assert_allclose(
            result.reconstructed_states.values,
            result.target_states.values,
            atol=1e-8,
        )
        np.testing.assert_allclose(
            result.reconstructed_observables.values,
            result.target_observables.values,
            atol=1e-8,
        )

    def test_components_include_shock_initial_residual(self, ar1_filter_smoother):
        ss, _, sm = ar1_filter_smoother
        result = historical_decomposition(ss, sm)
        assert "e" in result.components
        assert "initial" in result.components
        assert "residual" in result.components
        assert "steady_state" in result.observable_contributions.columns.get_level_values(0)

    def test_component_accessor(self, ar1_filter_smoother):
        ss, _, sm = ar1_filter_smoother
        result = historical_decomposition(ss, sm)
        e_obs = result.component("e", observables=True)
        e_state = result.component("e", observables=False)
        assert e_obs.shape[1] == ss.n_obs
        assert e_state.shape[1] == ss.n_states
        with pytest.raises(FilterError, match="not found"):
            _ = result.component("does_not_exist", observables=True)
