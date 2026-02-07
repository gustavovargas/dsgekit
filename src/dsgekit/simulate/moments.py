"""Theoretical and simulated moments for DSGE models.

Computes:
- Unconditional variances/covariances (solving Lyapunov equation)
- Autocorrelations
- Forecast error variance decomposition (FEVD)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import linalg

if TYPE_CHECKING:
    from dsgekit.model.calibration import Calibration
    from dsgekit.solvers.linear import LinearSolution


@dataclass
class MomentsResult:
    """Results from moments computation.

    Attributes:
        variance: Unconditional variances (dict var_name -> variance)
        covariance: Full covariance matrix
        std_dev: Standard deviations
        correlation: Correlation matrix
        autocorrelation: Autocorrelations at various lags
    """

    variance: dict[str, float]
    covariance: pd.DataFrame
    std_dev: dict[str, float]
    correlation: pd.DataFrame
    autocorrelation: pd.DataFrame | None = None

    def summary(self) -> str:
        """Return summary of moments."""
        lines = ["Unconditional Moments:", "=" * 40]
        lines.append(f"{'Variable':<12} {'Std.Dev.':<12} {'Variance':<12}")
        lines.append("-" * 36)

        for var in self.variance:
            std = self.std_dev[var]
            var_val = self.variance[var]
            lines.append(f"{var:<12} {std:<12.4f} {var_val:<12.6f}")

        return "\n".join(lines)


def compute_variance(
    solution: LinearSolution,
    calibration: Calibration,
) -> NDArray[np.float64]:
    """Compute unconditional covariance matrix by solving Lyapunov equation.

    For the system y_t = T * y_{t-1} + R * u_t with E[u_t u_t'] = Σ,
    the unconditional covariance Var(y) = V satisfies:
        V = T * V * T' + R * Σ * R'

    This is a discrete Lyapunov equation.

    Args:
        solution: LinearSolution with T and R matrices
        calibration: Calibration with shock covariances

    Returns:
        Covariance matrix V (n_vars x n_vars)
    """
    T = solution.T
    R = solution.R

    # Shock covariance
    Sigma = calibration.shock_cov_matrix(solution.shock_names)

    # R * Sigma * R'
    Q = R @ Sigma @ R.T

    # Solve discrete Lyapunov: V = T @ V @ T.T + Q
    # scipy.linalg.solve_discrete_lyapunov solves: A @ X @ A.H + Q = X
    try:
        V = linalg.solve_discrete_lyapunov(T, Q)
    except linalg.LinAlgError:
        # Fallback: iterate
        V = _solve_lyapunov_iterative(T, Q)

    return V


def _solve_lyapunov_iterative(
    T: NDArray[np.float64],
    Q: NDArray[np.float64],
    max_iter: int = 1000,
    tol: float = 1e-10,
) -> NDArray[np.float64]:
    """Solve Lyapunov equation by iteration."""
    V = Q.copy()
    for _ in range(max_iter):
        V_new = T @ V @ T.T + Q
        if np.max(np.abs(V_new - V)) < tol:
            return V_new
        V = V_new
    return V


def compute_autocorrelation(
    solution: LinearSolution,
    calibration: Calibration,
    max_lag: int = 10,
) -> pd.DataFrame:
    """Compute autocorrelation function.

    Args:
        solution: LinearSolution
        calibration: Calibration
        max_lag: Maximum lag to compute

    Returns:
        DataFrame with autocorrelations (rows=lags, columns=variables)
    """
    T = solution.T
    V = compute_variance(solution, calibration)

    # Standard deviations
    std_devs = np.sqrt(np.diag(V))

    # Autocorrelations
    # Cov(y_t, y_{t-k}) = T^k @ V
    # Corr(y_t, y_{t-k}) = diag(T^k @ V) / var(y)

    acf_data = np.zeros((max_lag + 1, len(solution.var_names)))

    T_power = np.eye(T.shape[0])
    for k in range(max_lag + 1):
        # Cov(y_t, y_{t-k}) for each variable (diagonal elements)
        cov_k = np.diag(T_power @ V)
        # Normalize by variance
        acf_data[k, :] = cov_k / (std_devs**2)
        T_power = T_power @ T

    return pd.DataFrame(
        acf_data,
        index=range(max_lag + 1),
        columns=solution.var_names,
    )


def compute_fevd(
    solution: LinearSolution,
    calibration: Calibration,
    horizon: int = 40,
) -> pd.DataFrame:
    """Compute Forecast Error Variance Decomposition.

    Shows the contribution of each shock to the forecast error variance
    of each variable at different horizons.

    Args:
        solution: LinearSolution
        calibration: Calibration
        horizon: Maximum horizon

    Returns:
        DataFrame with MultiIndex (horizon, variable) and columns=shocks
    """
    T = solution.T
    R = solution.R
    n_vars = T.shape[0]
    n_shocks = R.shape[1]

    Sigma = calibration.shock_cov_matrix(solution.shock_names)

    # Compute MSE contribution from each shock at each horizon
    # MSE_h = sum_{j=0}^{h-1} T^j @ R @ Sigma @ R' @ (T^j)'

    # For FEVD, we need contribution of each shock separately
    # Contribution of shock k at horizon h:
    # sum_{j=0}^{h-1} (T^j @ R @ e_k)^2 * sigma_k^2

    fevd_data = []

    for h in range(1, horizon + 1):
        # Total variance at horizon h
        total_var = np.zeros(n_vars)
        shock_contrib = np.zeros((n_vars, n_shocks))

        T_power = np.eye(n_vars)
        for _j in range(h):
            # Impact of shocks at lag j
            impact = T_power @ R  # n_vars x n_shocks

            for k in range(n_shocks):
                # Contribution of shock k
                shock_var = Sigma[k, k]
                contrib = (impact[:, k] ** 2) * shock_var
                shock_contrib[:, k] += contrib
                total_var += contrib

            T_power = T_power @ T

        # Normalize to percentages
        for i in range(n_vars):
            if total_var[i] > 0:
                shock_contrib[i, :] /= total_var[i]

        # Store results
        for i, var_name in enumerate(solution.var_names):
            row = {"horizon": h, "variable": var_name}
            for k, shock_name in enumerate(solution.shock_names):
                row[shock_name] = shock_contrib[i, k] * 100
            fevd_data.append(row)

    df = pd.DataFrame(fevd_data)
    df = df.set_index(["horizon", "variable"])

    return df


def moments(
    solution: LinearSolution,
    calibration: Calibration,
    max_lag: int = 10,
) -> MomentsResult:
    """Compute all theoretical moments.

    Args:
        solution: LinearSolution
        calibration: Calibration
        max_lag: Maximum lag for autocorrelations

    Returns:
        MomentsResult with all computed moments
    """
    V = compute_variance(solution, calibration)

    var_names = solution.var_names

    # Variances
    variances = dict(zip(var_names, np.diag(V), strict=True))

    # Standard deviations
    std_devs = dict(zip(var_names, np.sqrt(np.diag(V)), strict=True))

    # Covariance DataFrame
    cov_df = pd.DataFrame(V, index=var_names, columns=var_names)

    # Correlation matrix
    D_inv = np.diag(1.0 / np.sqrt(np.diag(V)))
    corr_matrix = D_inv @ V @ D_inv
    corr_df = pd.DataFrame(corr_matrix, index=var_names, columns=var_names)

    # Autocorrelations
    acf_df = compute_autocorrelation(solution, calibration, max_lag)

    return MomentsResult(
        variance=variances,
        covariance=cov_df,
        std_dev=std_devs,
        correlation=corr_df,
        autocorrelation=acf_df,
    )
