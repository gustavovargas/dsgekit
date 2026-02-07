"""Rauch-Tung-Striebel (RTS) Kalman smoother.

Backward pass over Kalman filter results to obtain smoothed state
estimates y_{t|T} that use the full sample of observations.

References:
    Rauch, H.E., Tung, F. & Striebel, C.T. (1965). Maximum likelihood
    estimates of linear dynamic systems. AIAA Journal, 3(8), 1445-1450.
    Durbin, J. & Koopman, S.J. (2012). Time Series Analysis by
    State Space Methods. Oxford University Press, Ch. 4.4.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from dsgekit.exceptions import FilterError

if TYPE_CHECKING:
    from dsgekit.filters.kalman import KalmanResult
    from dsgekit.transforms.statespace import StateSpace


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class SmootherResult:
    """Results from Kalman smoother.

    Attributes:
        smoothed_states: Smoothed state means y_{t|T}
            (rows=periods, columns=state_names).
        smoothed_cov: Smoothed covariances P_{t|T},
            shape (n_periods, n_states, n_states).
        smoother_gain: Smoother gains J_t,
            shape (n_periods, n_states, n_states).
            J_{T-1} is zero (no backward update at last period).
        n_periods: Number of time periods.
    """

    smoothed_states: pd.DataFrame
    smoothed_cov: NDArray[np.float64]
    smoother_gain: NDArray[np.float64]
    n_periods: int

    def __getitem__(self, var_name: str) -> NDArray[np.float64]:
        """Get smoothed state series for *var_name*."""
        return self.smoothed_states[var_name].values

    def summary(self) -> str:
        """Human-readable summary."""
        n_states = len(self.smoothed_states.columns)
        state_names = ", ".join(self.smoothed_states.columns)
        lines = [
            "Kalman Smoother Results",
            "=" * 40,
            f"  Periods: {self.n_periods}",
            f"  States:  {n_states} ({state_names})",
        ]
        return "\n".join(lines)

    def plot(
        self,
        variables: list[str] | None = None,
        *,
        filtered: pd.DataFrame | None = None,
        data: pd.DataFrame | None = None,
        figsize: tuple[float, float] | None = None,
        title: str = "Kalman Smoother",
    ):
        """Plot smoothed states.

        Args:
            variables: States to plot (default: all).
            filtered: Filtered states DataFrame for comparison overlay.
            data: Observed data DataFrame for overlay.
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
            variables = list(self.smoothed_states.columns)

        n_vars = len(variables)
        if figsize is None:
            figsize = (12, 3 * n_vars)
        fig, axes = plt.subplots(n_vars, 1, figsize=figsize, sharex=True)
        if n_vars == 1:
            axes = [axes]

        for ax, var in zip(axes, variables, strict=True):
            ax.plot(
                self.smoothed_states.index,
                self.smoothed_states[var],
                "b-",
                linewidth=1.2,
                label="Smoothed",
            )
            if filtered is not None and var in filtered.columns:
                ax.plot(
                    filtered.index,
                    filtered[var],
                    "g--",
                    linewidth=0.8,
                    alpha=0.7,
                    label="Filtered",
                )
            if data is not None and var in data.columns:
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
# Main smoother
# ---------------------------------------------------------------------------


def kalman_smoother(
    state_space: StateSpace,
    kalman_result: KalmanResult,
) -> SmootherResult:
    """Run RTS smoother (backward pass) on Kalman filter output.

    Computes smoothed state estimates y_{t|T} and covariances P_{t|T}
    using the full sample of observations.

    Equations (backward recursion for t = T-2, ..., 0):
        J_t     = P_{t|t} @ T' @ P_{t+1|t}^{-1}
        y_{t|T} = y_{t|t} + J_t @ (y_{t+1|T} - y_{t+1|t})
        P_{t|T} = P_{t|t} + J_t @ (P_{t+1|T} - P_{t+1|t}) @ J_t'

    Args:
        state_space: StateSpace used for filtering.
        kalman_result: Output from :func:`kalman_filter`.

    Returns:
        SmootherResult

    Raises:
        FilterError: If inputs are inconsistent.
    """
    kf = kalman_result
    T_mat = state_space.T
    n_periods = kf.n_periods
    n_states = T_mat.shape[0]

    if kf.filtered_cov.shape[1] != n_states:
        raise FilterError(
            f"State dimension mismatch: state_space has {n_states} states, "
            f"kalman_result has {kf.filtered_cov.shape[1]}"
        )

    # Extract arrays from KalmanResult
    filt_s = kf.filtered_states.values   # (n_periods, n_states)
    pred_s = kf.predicted_states.values  # (n_periods, n_states)
    filt_P = kf.filtered_cov             # (n_periods, n_states, n_states)
    pred_P = kf.predicted_cov            # (n_periods, n_states, n_states)

    # Storage
    smooth_s = np.zeros_like(filt_s)
    smooth_P = np.zeros_like(filt_P)
    J = np.zeros_like(filt_P)

    # Initialise: last period smoothed = filtered
    smooth_s[-1] = filt_s[-1]
    smooth_P[-1] = filt_P[-1]

    # Backward pass
    T_T = T_mat.T
    for t in range(n_periods - 2, -1, -1):
        # P_{t+1|t} is the predicted covariance at t+1
        P_pred_next = pred_P[t + 1]

        # Smoother gain: J_t = P_{t|t} @ T' @ P_{t+1|t}^{-1}
        # Compute via solve: J_t @ P_{t+1|t} = P_{t|t} @ T'
        # → J_t = (P_{t+1|t}^{-1} @ (P_{t|t} @ T')^T)^T  ... cleaner with solve
        # Or: J_t = P_{t|t} @ T' @ inv(P_{t+1|t})
        # Using solve for stability: solve(P_pred_next.T, (filt_P[t] @ T_T).T).T
        J_t = np.linalg.solve(P_pred_next.T, (filt_P[t] @ T_T).T).T

        # Smoothed state
        smooth_s[t] = filt_s[t] + J_t @ (smooth_s[t + 1] - pred_s[t + 1])

        # Smoothed covariance
        smooth_P[t] = filt_P[t] + J_t @ (smooth_P[t + 1] - P_pred_next) @ J_t.T
        smooth_P[t] = 0.5 * (smooth_P[t] + smooth_P[t].T)  # symmetrize

        J[t] = J_t

    # Build DataFrame
    smoothed_df = pd.DataFrame(
        smooth_s,
        index=kf.filtered_states.index,
        columns=kf.filtered_states.columns,
    )
    smoothed_df.index.name = "period"

    return SmootherResult(
        smoothed_states=smoothed_df,
        smoothed_cov=smooth_P,
        smoother_gain=J,
        n_periods=n_periods,
    )
