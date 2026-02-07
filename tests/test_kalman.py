"""Tests for Kalman filter."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dsgekit import load_model
from dsgekit.exceptions import FilterError
from dsgekit.filters import KalmanResult, kalman_filter, log_likelihood
from dsgekit.simulate import simulate
from dsgekit.solvers import solve_linear
from dsgekit.transforms import linearize, to_state_space

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ar1_pipeline(models_dir):
    """Full AR(1) pipeline: model → solution → state_space."""
    model, cal, ss = load_model(models_dir / "ar1.yaml")
    solution = solve_linear(linearize(model, ss, cal))
    state_space = to_state_space(solution, cal, observables=["y"])
    return state_space, solution, cal


@pytest.fixture
def ar1_data(ar1_pipeline):
    """Synthetic AR(1) observations (200 periods, seed=42)."""
    _, solution, cal = ar1_pipeline
    sim = simulate(solution, cal, n_periods=200, seed=42)
    return sim.data[["y"]]


# ---------------------------------------------------------------------------
# Basic execution
# ---------------------------------------------------------------------------


class TestBasicExecution:
    def test_returns_kalman_result(self, ar1_pipeline, ar1_data):
        ss, *_ = ar1_pipeline
        result = kalman_filter(ss, ar1_data)
        assert isinstance(result, KalmanResult)

    def test_shapes(self, ar1_pipeline, ar1_data):
        ss, *_ = ar1_pipeline
        result = kalman_filter(ss, ar1_data)

        n = 200
        assert result.filtered_states.shape == (n, ss.n_states)
        assert result.predicted_states.shape == (n, ss.n_states)
        assert result.filtered_cov.shape == (n, ss.n_states, ss.n_states)
        assert result.predicted_cov.shape == (n, ss.n_states, ss.n_states)
        assert result.innovations.shape == (n, ss.n_obs)
        assert result.innovation_cov.shape == (n, ss.n_obs, ss.n_obs)
        assert result.log_likelihood_contributions.shape == (n,)
        assert result.n_periods == n

    def test_log_likelihood_finite(self, ar1_pipeline, ar1_data):
        ss, *_ = ar1_pipeline
        result = kalman_filter(ss, ar1_data)
        assert np.isfinite(result.log_likelihood)
        # With σ_e=0.01, the Gaussian density is very concentrated,
        # so per-period ll ≈ -0.5*log(2π*0.0001) ≈ 3.57 → total positive.
        assert result.log_likelihood > 0

    def test_n_obs_used(self, ar1_pipeline, ar1_data):
        ss, *_ = ar1_pipeline
        result = kalman_filter(ss, ar1_data)
        # No missing obs → n_obs_used == n_periods * n_obs
        assert result.n_obs_used == 200 * ss.n_obs


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


class TestInitialisation:
    def test_lyapunov_initial_cov(self, ar1_pipeline, ar1_data):
        """P_0 should equal sigma_e^2 / (1 - rho^2) for AR(1)."""
        ss, *_ = ar1_pipeline
        result = kalman_filter(ss, ar1_data)
        # Analytical: 0.01^2 / (1 - 0.9^2) = 0.0001 / 0.19
        expected_P0 = 0.0001 / 0.19
        # predicted_cov[0] = T @ P_0 @ T' + Q, but P_0 is the Lyapunov
        # solution, so T @ P_0 @ T' + Q = P_0 (stationary).
        np.testing.assert_allclose(
            result.predicted_cov[0, 0, 0], expected_P0, rtol=1e-6
        )

    def test_custom_initial_state(self, ar1_pipeline, ar1_data):
        ss, *_ = ar1_pipeline
        result = kalman_filter(
            ss, ar1_data, initial_state=np.array([1.0])
        )
        assert isinstance(result, KalmanResult)
        assert np.isfinite(result.log_likelihood)

    def test_custom_initial_cov(self, ar1_pipeline, ar1_data):
        ss, *_ = ar1_pipeline
        result = kalman_filter(
            ss, ar1_data, initial_cov=np.eye(1) * 10.0
        )
        assert isinstance(result, KalmanResult)
        assert np.isfinite(result.log_likelihood)


# ---------------------------------------------------------------------------
# H=0: filtered states should track observed data
# ---------------------------------------------------------------------------


class TestPerfectObservation:
    def test_filtered_equals_observed_after_convergence(
        self, ar1_pipeline, ar1_data
    ):
        """With H=0, Z=I → Kalman gain → 1, filtered ≈ observed."""
        ss, *_ = ar1_pipeline
        result = kalman_filter(ss, ar1_data)
        # After a few periods the gain converges
        filtered = result.filtered_states["y"].values
        observed = ar1_data["y"].values
        # Check last 190 periods (skip first 10 for convergence)
        np.testing.assert_allclose(filtered[10:], observed[10:], atol=1e-8)


# ---------------------------------------------------------------------------
# Data input formats
# ---------------------------------------------------------------------------


class TestDataFormats:
    def test_ndarray_input(self, ar1_pipeline, ar1_data):
        ss, *_ = ar1_pipeline
        arr = ar1_data.values
        result = kalman_filter(ss, arr, demean=False)
        assert isinstance(result, KalmanResult)
        assert np.isfinite(result.log_likelihood)

    def test_ndarray_matches_dataframe(self, ar1_pipeline, ar1_data):
        ss, *_ = ar1_pipeline
        r1 = kalman_filter(ss, ar1_data, demean=False)
        r2 = kalman_filter(ss, ar1_data.values, demean=False)
        np.testing.assert_allclose(r1.log_likelihood, r2.log_likelihood)
        np.testing.assert_allclose(
            r1.filtered_states.values, r2.filtered_states.values
        )


# ---------------------------------------------------------------------------
# Demean
# ---------------------------------------------------------------------------


class TestDemean:
    def test_demean_equivalence(self, ar1_pipeline, ar1_data):
        """Adding a constant and demeaning should give same result."""
        ss, *_ = ar1_pipeline
        # AR(1) steady state is 0, so demean=True with offset data
        # should equal demean=False with original data.
        offset = 5.0
        data_levels = ar1_data.copy()
        data_levels["y"] = data_levels["y"] + offset

        # Modify steady state to match the offset
        ss_modified = type(ss)(
            T=ss.T, R=ss.R, Q=ss.Q, Z=ss.Z, H=ss.H,
            state_names=ss.state_names,
            obs_names=ss.obs_names,
            shock_names=ss.shock_names,
            steady_state={"y": offset},
        )

        r_demean = kalman_filter(ss_modified, data_levels, demean=True)
        r_no_demean = kalman_filter(ss, ar1_data, demean=False)
        np.testing.assert_allclose(
            r_demean.log_likelihood, r_no_demean.log_likelihood, rtol=1e-10
        )


# ---------------------------------------------------------------------------
# Missing observations
# ---------------------------------------------------------------------------


class TestMissingObs:
    def test_all_nan_periods(self, ar1_pipeline, ar1_data):
        ss, *_ = ar1_pipeline
        data = ar1_data.copy()
        data.iloc[10] = np.nan
        data.iloc[50] = np.nan

        result = kalman_filter(ss, data)
        assert np.isfinite(result.log_likelihood)
        # ll contribution should be 0 for missing periods
        assert result.log_likelihood_contributions[10] == 0.0
        assert result.log_likelihood_contributions[50] == 0.0
        # Innovations should be NaN at missing periods
        assert np.isnan(result.innovations.iloc[10].values).all()
        # n_obs_used should be less
        assert result.n_obs_used == (200 - 2) * ss.n_obs

    def test_missing_does_not_crash(self, ar1_pipeline, ar1_data):
        ss, *_ = ar1_pipeline
        data = ar1_data.copy()
        # Make 10% of data missing
        rng = np.random.default_rng(123)
        missing_idx = rng.choice(200, size=20, replace=False)
        data.iloc[missing_idx] = np.nan

        result = kalman_filter(ss, data)
        assert np.isfinite(result.log_likelihood)
        assert result.n_obs_used == (200 - 20) * ss.n_obs


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrors:
    def test_wrong_ndarray_columns(self, ar1_pipeline):
        ss, *_ = ar1_pipeline
        bad_data = np.zeros((50, 3))  # n_obs is 1
        with pytest.raises(FilterError, match="expected 1"):
            kalman_filter(ss, bad_data)

    def test_wrong_ndarray_ndim(self, ar1_pipeline):
        ss, *_ = ar1_pipeline
        with pytest.raises(FilterError, match="2-D"):
            kalman_filter(ss, np.zeros(50))

    def test_missing_dataframe_column(self, ar1_pipeline):
        ss, *_ = ar1_pipeline
        bad_df = pd.DataFrame({"x": np.zeros(50)})
        with pytest.raises(FilterError, match="missing columns"):
            kalman_filter(ss, bad_df)


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------


class TestLogLikelihood:
    def test_matches_full_filter(self, ar1_pipeline, ar1_data):
        ss, *_ = ar1_pipeline
        ll = log_likelihood(ss, ar1_data)
        result = kalman_filter(ss, ar1_data)
        assert ll == result.log_likelihood


# ---------------------------------------------------------------------------
# Summary / getitem
# ---------------------------------------------------------------------------


class TestMethods:
    def test_summary(self, ar1_pipeline, ar1_data):
        ss, *_ = ar1_pipeline
        result = kalman_filter(ss, ar1_data)
        s = result.summary()
        assert "Log-likelihood" in s
        assert "200" in s

    def test_getitem(self, ar1_pipeline, ar1_data):
        ss, *_ = ar1_pipeline
        result = kalman_filter(ss, ar1_data)
        y_arr = result["y"]
        assert y_arr.shape == (200,)
        np.testing.assert_array_equal(
            y_arr, result.filtered_states["y"].values
        )
