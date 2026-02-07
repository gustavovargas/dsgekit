"""Kalman filter for DSGE state-space models.

Implements the standard multivariate Kalman filter with:
- Missing observations handling (NaN in data)
- Cholesky-based numerically stable inversion
- Log-likelihood evaluation
- Unconditional initialization via Lyapunov equation

References:
    Harvey, A.C. (1989). Forecasting, Structural Time Series Models
    and the Kalman Filter. Cambridge University Press.
    Durbin, J. & Koopman, S.J. (2012). Time Series Analysis by
    State Space Methods. Oxford University Press.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import linalg

from dsgekit.exceptions import FilterError, LikelihoodError, NonPositiveDefiniteError

if TYPE_CHECKING:
    from dsgekit.transforms.statespace import StateSpace


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class KalmanResult:
    """Results from Kalman filter.

    Attributes:
        filtered_states: Filtered state means y_{t|t}
            (rows=periods, columns=state_names).
        predicted_states: Predicted state means y_{t|t-1}.
        filtered_cov: Filtered covariances P_{t|t},
            shape (n_periods, n_states, n_states).
        predicted_cov: Predicted covariances P_{t|t-1}.
        innovations: Prediction errors v_t = z_t - Z @ y_{t|t-1}
            (NaN where observations are missing).
        innovation_cov: Innovation covariances F_t,
            shape (n_periods, n_obs, n_obs).
        log_likelihood: Total log-likelihood.
        log_likelihood_contributions: Per-period log-likelihood.
        n_periods: Number of time periods.
        n_obs_used: Number of non-missing observation-periods.
    """

    filtered_states: pd.DataFrame
    predicted_states: pd.DataFrame
    filtered_cov: NDArray[np.float64]
    predicted_cov: NDArray[np.float64]
    innovations: pd.DataFrame
    innovation_cov: NDArray[np.float64]
    log_likelihood: float
    log_likelihood_contributions: NDArray[np.float64]
    n_periods: int
    n_obs_used: int

    def __getitem__(self, var_name: str) -> NDArray[np.float64]:
        """Get filtered state series for *var_name*."""
        return self.filtered_states[var_name].values

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            "Kalman Filter Results",
            "=" * 40,
            f"  Periods:            {self.n_periods}",
            f"  States:             {len(self.filtered_states.columns)}",
            f"  Observables:        {len(self.innovations.columns)}",
            f"  Obs used (non-NaN): {self.n_obs_used}",
            "",
            f"  Log-likelihood:     {self.log_likelihood:.4f}",
            f"  Avg ll / period:    {self.log_likelihood / max(self.n_periods, 1):.4f}",
        ]
        return "\n".join(lines)

    def plot(
        self,
        variables: list[str] | None = None,
        *,
        show_data: bool = True,
        data: pd.DataFrame | None = None,
        figsize: tuple[float, float] | None = None,
        title: str = "Kalman Filter",
    ):
        """Plot filtered states.

        Args:
            variables: States to plot (default: observed variables).
            show_data: Overlay observed data points.
            data: Original data DataFrame for overlay.
            figsize: Figure size.
            title: Plot title.

        Returns:
            matplotlib Figure.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as err:
            raise ImportError(
                "matplotlib is required for plotting. "
                "Install with: pip install dsgekit[plot]"
            ) from err

        if variables is None:
            variables = list(self.innovations.columns)

        n_vars = len(variables)
        if figsize is None:
            figsize = (12, 3 * n_vars)
        fig, axes = plt.subplots(n_vars, 1, figsize=figsize, sharex=True)
        if n_vars == 1:
            axes = [axes]

        for ax, var in zip(axes, variables, strict=True):
            if var in self.filtered_states.columns:
                ax.plot(
                    self.filtered_states.index,
                    self.filtered_states[var],
                    "b-",
                    linewidth=1.2,
                    label="Filtered",
                )
            if show_data and data is not None and var in data.columns:
                ax.plot(
                    data.index,
                    data[var],
                    "ro",
                    markersize=3,
                    alpha=0.5,
                    label="Observed",
                )
            ax.set_ylabel(var)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best", fontsize="small")

        axes[-1].set_xlabel("Period")
        fig.suptitle(title)
        fig.tight_layout()
        return fig


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _initialize_filter(
    ss: StateSpace,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Unconditional initial state and covariance.

    y_0 = 0 (deviations from steady state)
    P_0 = solve_discrete_lyapunov(T, Q)
    """
    y0 = np.zeros(ss.n_states, dtype=np.float64)

    try:
        P0 = linalg.solve_discrete_lyapunov(ss.T, ss.Q)
    except linalg.LinAlgError:
        # Iterative fallback (same approach as moments.py)
        P0 = ss.Q.copy()
        for _ in range(1000):
            P0_new = ss.T @ P0 @ ss.T.T + ss.Q
            if np.max(np.abs(P0_new - P0)) < 1e-10:
                P0 = P0_new
                break
            P0 = P0_new

    P0 = 0.5 * (P0 + P0.T)  # symmetrize
    return y0, P0


def _stable_innovation_inverse(
    F: NDArray[np.float64],
    t: int,
) -> tuple[NDArray[np.float64], float]:
    """Compute F^{-1} and log|F| via Cholesky.

    Raises:
        NonPositiveDefiniteError: If F is not positive definite.
    """
    try:
        L = linalg.cholesky(F, lower=True)
        L_inv = linalg.solve_triangular(L, np.eye(F.shape[0]), lower=True)
        F_inv = L_inv.T @ L_inv
        log_det_F = 2.0 * np.sum(np.log(np.diag(L)))
        return F_inv, log_det_F
    except linalg.LinAlgError as err:
        raise NonPositiveDefiniteError("F (innovation covariance)", time_step=t) from err


def _handle_missing_obs(
    z_t: NDArray[np.float64],
    Z: NDArray[np.float64],
    H: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
    """Extract sub-system of non-missing observations."""
    valid = ~np.isnan(z_t)
    return z_t[valid], Z[valid, :], H[np.ix_(valid, valid)], valid


# ---------------------------------------------------------------------------
# Main filter
# ---------------------------------------------------------------------------


def kalman_filter(
    state_space: StateSpace,
    data: pd.DataFrame | NDArray[np.float64],
    *,
    demean: bool = True,
    initial_state: NDArray[np.float64] | None = None,
    initial_cov: NDArray[np.float64] | None = None,
) -> KalmanResult:
    """Run Kalman filter on observed data.

    State equation:       y_t = T @ y_{t-1} + eta_t,  eta_t ~ N(0, Q)
    Observation equation: z_t = Z @ y_t + v_t,         v_t  ~ N(0, H)

    Args:
        state_space: StateSpace from ``to_state_space()``.
        data: Observed data — DataFrame with columns matching
            ``state_space.obs_names``, or 2-D array (n_periods × n_obs).
        demean: Subtract steady-state values from data before filtering.
            Set False if data is already in deviations.
        initial_state: Initial state mean (default: zeros).
        initial_cov: Initial covariance (default: Lyapunov solution).

    Returns:
        KalmanResult

    Raises:
        FilterError: Data dimension mismatch.
        NonPositiveDefiniteError: Innovation covariance not PD.
        LikelihoodError: Non-finite log-likelihood.
    """
    ss = state_space
    T, Q, Z, H = ss.T, ss.Q, ss.Z, ss.H
    n_states = ss.n_states
    n_obs = ss.n_obs

    # ---- convert data to array ----
    if isinstance(data, pd.DataFrame):
        missing_cols = set(ss.obs_names) - set(data.columns)
        if missing_cols:
            raise FilterError(
                f"Data missing columns for observables: {sorted(missing_cols)}. "
                f"Expected: {ss.obs_names}"
            )
        obs_array = data[ss.obs_names].values.astype(np.float64)
        data_index = data.index
    else:
        obs_array = np.asarray(data, dtype=np.float64)
        if obs_array.ndim != 2:
            raise FilterError(f"Data must be 2-D, got {obs_array.ndim}-D")
        if obs_array.shape[1] != n_obs:
            raise FilterError(
                f"Data has {obs_array.shape[1]} columns, expected {n_obs}"
            )
        data_index = range(obs_array.shape[0])

    n_periods = obs_array.shape[0]

    # ---- demean ----
    if demean:
        ss_vals = np.array(
            [ss.steady_state.get(name, 0.0) for name in ss.obs_names],
            dtype=np.float64,
        )
        obs_array = obs_array - ss_vals[np.newaxis, :]

    # ---- initialise ----
    if initial_state is not None:
        y = initial_state.copy()
    else:
        y, _ = _initialize_filter(ss)

    if initial_cov is not None:
        P = initial_cov.copy()
    else:
        _, P = _initialize_filter(ss)

    # ---- storage ----
    filt_s = np.zeros((n_periods, n_states))
    pred_s = np.zeros((n_periods, n_states))
    filt_P = np.zeros((n_periods, n_states, n_states))
    pred_P = np.zeros((n_periods, n_states, n_states))
    innov = np.full((n_periods, n_obs), np.nan)
    innov_F = np.zeros((n_periods, n_obs, n_obs))
    ll_t = np.zeros(n_periods)

    n_obs_used = 0
    I_n = np.eye(n_states)

    # ---- main loop ----
    for t in range(n_periods):
        # --- prediction ---
        y_pred = T @ y
        P_pred = T @ P @ T.T + Q
        P_pred = 0.5 * (P_pred + P_pred.T)

        pred_s[t] = y_pred
        pred_P[t] = P_pred

        # --- observation handling ---
        z_t = obs_array[t]
        all_missing = np.all(np.isnan(z_t))

        if all_missing:
            y, P = y_pred, P_pred
            filt_s[t] = y
            filt_P[t] = P
            ll_t[t] = 0.0
            continue

        any_missing = np.any(np.isnan(z_t))
        if any_missing:
            z_v, Z_t, H_t, mask = _handle_missing_obs(z_t, Z, H)
            n_obs_t = int(np.sum(mask))
        else:
            z_v, Z_t, H_t = z_t, Z, H
            n_obs_t = n_obs
            mask = None

        n_obs_used += n_obs_t

        # --- update ---
        v = z_v - Z_t @ y_pred
        F = Z_t @ P_pred @ Z_t.T + H_t
        F = 0.5 * (F + F.T)

        F_inv, log_det_F = _stable_innovation_inverse(F, t)

        K = P_pred @ Z_t.T @ F_inv
        y = y_pred + K @ v
        P = (I_n - K @ Z_t) @ P_pred
        P = 0.5 * (P + P.T)

        # store innovations
        if mask is not None:
            innov[t, mask] = v
            innov_F[t][np.ix_(mask, mask)] = F
        else:
            innov[t] = v
            innov_F[t] = F

        filt_s[t] = y
        filt_P[t] = P

        # log-likelihood contribution
        ll_t[t] = -0.5 * (
            n_obs_t * np.log(2.0 * np.pi) + log_det_F + v @ F_inv @ v
        )

    # ---- aggregate ----
    total_ll = float(np.sum(ll_t))
    if not np.isfinite(total_ll):
        raise LikelihoodError(
            total_ll,
            "Log-likelihood is not finite. Check model and data.",
        )

    # ---- build DataFrames ----
    filtered_df = pd.DataFrame(filt_s, index=data_index, columns=ss.state_names)
    filtered_df.index.name = "period"

    predicted_df = pd.DataFrame(pred_s, index=data_index, columns=ss.state_names)
    predicted_df.index.name = "period"

    innovations_df = pd.DataFrame(innov, index=data_index, columns=ss.obs_names)
    innovations_df.index.name = "period"

    return KalmanResult(
        filtered_states=filtered_df,
        predicted_states=predicted_df,
        filtered_cov=filt_P,
        predicted_cov=pred_P,
        innovations=innovations_df,
        innovation_cov=innov_F,
        log_likelihood=total_ll,
        log_likelihood_contributions=ll_t,
        n_periods=n_periods,
        n_obs_used=n_obs_used,
    )


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------


def log_likelihood(
    state_space: StateSpace,
    data: pd.DataFrame | NDArray[np.float64],
    *,
    demean: bool = True,
) -> float:
    """Compute log-likelihood of data given a state-space model.

    Thin wrapper around :func:`kalman_filter` returning only the scalar
    log-likelihood. Useful for optimisation routines.
    """
    return kalman_filter(state_space, data, demean=demean).log_likelihood
