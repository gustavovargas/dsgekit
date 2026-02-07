"""Tests for Kalman smoother."""

from __future__ import annotations

import numpy as np
import pytest

from dsgekit import load_model
from dsgekit.filters import (
    SmootherResult,
    kalman_filter,
    kalman_smoother,
)
from dsgekit.simulate import simulate
from dsgekit.solvers import solve_linear
from dsgekit.transforms import linearize, to_state_space

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ar1_pipeline(models_dir):
    model, cal, ss = load_model(models_dir / "ar1.yaml")
    solution = solve_linear(linearize(model, ss, cal))
    state_space = to_state_space(solution, cal, observables=["y"])
    return state_space, solution, cal


@pytest.fixture
def ar1_filter(ar1_pipeline):
    """Run Kalman filter on 200-period AR(1) simulation."""
    ss, solution, cal = ar1_pipeline
    sim = simulate(solution, cal, n_periods=200, seed=42)
    kf = kalman_filter(ss, sim.data[["y"]])
    return ss, kf


# ---------------------------------------------------------------------------
# Basic execution
# ---------------------------------------------------------------------------


class TestBasicExecution:
    def test_returns_smoother_result(self, ar1_filter):
        ss, kf = ar1_filter
        result = kalman_smoother(ss, kf)
        assert isinstance(result, SmootherResult)

    def test_shapes(self, ar1_filter):
        ss, kf = ar1_filter
        result = kalman_smoother(ss, kf)
        n = 200
        assert result.smoothed_states.shape == (n, ss.n_states)
        assert result.smoothed_cov.shape == (n, ss.n_states, ss.n_states)
        assert result.smoother_gain.shape == (n, ss.n_states, ss.n_states)
        assert result.n_periods == n


# ---------------------------------------------------------------------------
# Smoother properties
# ---------------------------------------------------------------------------


class TestSmootherProperties:
    def test_last_period_equals_filtered(self, ar1_filter):
        """Smoothed at T-1 must equal filtered (no future info)."""
        ss, kf = ar1_filter
        result = kalman_smoother(ss, kf)
        np.testing.assert_allclose(
            result.smoothed_states.iloc[-1].values,
            kf.filtered_states.iloc[-1].values,
        )
        np.testing.assert_allclose(
            result.smoothed_cov[-1],
            kf.filtered_cov[-1],
        )

    def test_smoothed_cov_leq_filtered(self, ar1_filter):
        """Smoothed variance should be <= filtered variance (more info)."""
        ss, kf = ar1_filter
        result = kalman_smoother(ss, kf)
        for t in range(result.n_periods):
            smooth_var = np.diag(result.smoothed_cov[t])
            filt_var = np.diag(kf.filtered_cov[t])
            # Allow small numerical tolerance
            assert np.all(smooth_var <= filt_var + 1e-15), (
                f"Smoothed variance > filtered at t={t}: "
                f"{smooth_var} > {filt_var}"
            )

    def test_smoothed_cov_symmetric(self, ar1_filter):
        ss, kf = ar1_filter
        result = kalman_smoother(ss, kf)
        for t in range(result.n_periods):
            np.testing.assert_allclose(
                result.smoothed_cov[t],
                result.smoothed_cov[t].T,
                atol=1e-14,
            )


# ---------------------------------------------------------------------------
# AR(1) with H=0: smoothed ≈ observed
# ---------------------------------------------------------------------------


class TestPerfectObservation:
    def test_smoothed_close_to_observed(self, ar1_pipeline):
        """With H=0, Z=I the smoother should recover observations."""
        ss, solution, cal = ar1_pipeline
        sim = simulate(solution, cal, n_periods=200, seed=42)
        data = sim.data[["y"]]
        kf = kalman_filter(ss, data)
        sm = kalman_smoother(ss, kf)
        # Interior points (skip first/last couple for edge effects)
        np.testing.assert_allclose(
            sm.smoothed_states["y"].values[5:-5],
            data["y"].values[5:-5],
            atol=1e-6,
        )


# ---------------------------------------------------------------------------
# Missing observations
# ---------------------------------------------------------------------------


class TestMissingObs:
    def test_smoother_with_missing_data(self, ar1_pipeline):
        ss, solution, cal = ar1_pipeline
        sim = simulate(solution, cal, n_periods=200, seed=42)
        data = sim.data[["y"]].copy()
        data.iloc[50] = np.nan
        data.iloc[100] = np.nan

        kf = kalman_filter(ss, data)
        sm = kalman_smoother(ss, kf)
        assert isinstance(sm, SmootherResult)
        assert np.all(np.isfinite(sm.smoothed_states.values))


# ---------------------------------------------------------------------------
# Methods
# ---------------------------------------------------------------------------


class TestMethods:
    def test_summary(self, ar1_filter):
        ss, kf = ar1_filter
        result = kalman_smoother(ss, kf)
        s = result.summary()
        assert "Smoother" in s
        assert "200" in s

    def test_getitem(self, ar1_filter):
        ss, kf = ar1_filter
        result = kalman_smoother(ss, kf)
        y_arr = result["y"]
        assert y_arr.shape == (200,)
        np.testing.assert_array_equal(
            y_arr, result.smoothed_states["y"].values
        )
