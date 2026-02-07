"""State-space representation for Kalman filtering.

Converts LinearSolution to state-space form suitable for estimation.

State equation:     y_t = T @ y_{t-1} + R @ u_t
Observation equation: z_t = Z @ y_t + v_t

where:
- y_t: state vector (deviations from steady state)
- z_t: observed variables
- u_t ~ N(0, Σ): structural shocks
- v_t ~ N(0, H): measurement errors
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from dsgekit.exceptions import InvalidObservableError, StateSpaceError

if TYPE_CHECKING:
    from dsgekit.model.calibration import Calibration
    from dsgekit.solvers.linear import LinearSolution


@dataclass
class StateSpace:
    """State-space representation for Kalman filtering.

    State:       y_t = T @ y_{t-1} + R @ u_t
    Observation: z_t = Z @ y_t + v_t

    Attributes:
        T: Transition matrix (n_states x n_states)
        R: Shock impact matrix (n_states x n_shocks)
        Q: Process covariance = R @ Σ @ R' (n_states x n_states)
        Z: Observation matrix (n_obs x n_states)
        H: Measurement covariance (n_obs x n_obs)
        state_names: Names of state variables
        obs_names: Names of observed variables
        shock_names: Names of shocks
        steady_state: Steady state values for converting deviations ↔ levels
    """

    T: NDArray[np.float64]
    R: NDArray[np.float64]
    Q: NDArray[np.float64]
    Z: NDArray[np.float64]
    H: NDArray[np.float64]
    state_names: list[str]
    obs_names: list[str]
    shock_names: list[str]
    steady_state: dict[str, float]

    @property
    def n_states(self) -> int:
        """Number of state variables."""
        return self.T.shape[0]

    @property
    def n_obs(self) -> int:
        """Number of observed variables."""
        return self.Z.shape[0]

    @property
    def n_shocks(self) -> int:
        """Number of shocks."""
        return self.R.shape[1]

    def summary(self) -> str:
        """Return a summary of the state-space representation."""
        lines = [
            "State-Space Representation:",
            f"  States: {self.n_states} ({', '.join(self.state_names[:5])}{'...' if self.n_states > 5 else ''})",
            f"  Observables: {self.n_obs} ({', '.join(self.obs_names)})",
            f"  Shocks: {self.n_shocks} ({', '.join(self.shock_names[:5])}{'...' if self.n_shocks > 5 else ''})",
            "",
            "Matrix dimensions:",
            f"  T: {self.T.shape}",
            f"  R: {self.R.shape}",
            f"  Q: {self.Q.shape}",
            f"  Z: {self.Z.shape}",
            f"  H: {self.H.shape}",
        ]
        return "\n".join(lines)


def _build_selection_matrix(
    state_names: list[str], obs_names: list[str]
) -> NDArray[np.float64]:
    """Build selection matrix Z that picks observed variables from state vector.

    Args:
        state_names: Names of all state variables
        obs_names: Names of variables to observe

    Returns:
        Z matrix of shape (n_obs, n_states) with 1s at positions of observed vars

    Raises:
        InvalidObservableError: If an observable is not in state_names
    """
    n_obs = len(obs_names)
    n_states = len(state_names)
    Z = np.zeros((n_obs, n_states), dtype=np.float64)

    state_name_to_idx = {name: i for i, name in enumerate(state_names)}

    for i, obs in enumerate(obs_names):
        if obs not in state_name_to_idx:
            raise InvalidObservableError(
                obs, f"not found in state variables: {state_names}"
            )
        Z[i, state_name_to_idx[obs]] = 1.0

    return Z


def _build_measurement_cov(
    spec: float | NDArray[np.float64] | None, n_obs: int
) -> NDArray[np.float64]:
    """Build measurement error covariance matrix H.

    Args:
        spec: None (H=0), scalar variance (H=σ²I), or full matrix
        n_obs: Number of observables

    Returns:
        H matrix of shape (n_obs, n_obs)

    Raises:
        StateSpaceError: If spec has wrong dimensions
    """
    if spec is None:
        return np.zeros((n_obs, n_obs), dtype=np.float64)

    if np.isscalar(spec):
        return float(spec) * np.eye(n_obs, dtype=np.float64)

    # Full matrix provided
    H = np.asarray(spec, dtype=np.float64)
    if H.shape != (n_obs, n_obs):
        raise StateSpaceError(
            f"Measurement covariance matrix has shape {H.shape}, "
            f"expected ({n_obs}, {n_obs})"
        )
    return H


def validate_state_space(ss: StateSpace) -> None:
    """Validate dimensions and properties of a StateSpace.

    Args:
        ss: StateSpace to validate

    Raises:
        StateSpaceError: If validation fails
    """
    n_states = ss.n_states
    n_obs = ss.n_obs
    n_shocks = ss.n_shocks

    # Check T dimensions
    if ss.T.shape != (n_states, n_states):
        raise StateSpaceError(
            f"T matrix has shape {ss.T.shape}, expected ({n_states}, {n_states})"
        )

    # Check R dimensions
    if ss.R.shape != (n_states, n_shocks):
        raise StateSpaceError(
            f"R matrix has shape {ss.R.shape}, expected ({n_states}, {n_shocks})"
        )

    # Check Q dimensions
    if ss.Q.shape != (n_states, n_states):
        raise StateSpaceError(
            f"Q matrix has shape {ss.Q.shape}, expected ({n_states}, {n_states})"
        )

    # Check Z dimensions
    if ss.Z.shape != (n_obs, n_states):
        raise StateSpaceError(
            f"Z matrix has shape {ss.Z.shape}, expected ({n_obs}, {n_states})"
        )

    # Check H dimensions
    if ss.H.shape != (n_obs, n_obs):
        raise StateSpaceError(
            f"H matrix has shape {ss.H.shape}, expected ({n_obs}, {n_obs})"
        )

    # Check name list lengths
    if len(ss.state_names) != n_states:
        raise StateSpaceError(
            f"state_names has {len(ss.state_names)} elements, expected {n_states}"
        )

    if len(ss.obs_names) != n_obs:
        raise StateSpaceError(
            f"obs_names has {len(ss.obs_names)} elements, expected {n_obs}"
        )

    if len(ss.shock_names) != n_shocks:
        raise StateSpaceError(
            f"shock_names has {len(ss.shock_names)} elements, expected {n_shocks}"
        )

    # Check Q is symmetric
    if not np.allclose(ss.Q, ss.Q.T):
        raise StateSpaceError("Q matrix is not symmetric")

    # Check H is symmetric
    if not np.allclose(ss.H, ss.H.T):
        raise StateSpaceError("H matrix is not symmetric")

    # Check Q is positive semi-definite
    eigvals_Q = np.linalg.eigvalsh(ss.Q)
    if np.any(eigvals_Q < -1e-10):
        raise StateSpaceError(
            f"Q matrix is not positive semi-definite (min eigenvalue: {eigvals_Q.min():.2e})"
        )

    # Check H is positive semi-definite
    eigvals_H = np.linalg.eigvalsh(ss.H)
    if np.any(eigvals_H < -1e-10):
        raise StateSpaceError(
            f"H matrix is not positive semi-definite (min eigenvalue: {eigvals_H.min():.2e})"
        )


def to_state_space(
    solution: LinearSolution,
    calibration: Calibration,
    observables: list[str] | NDArray[np.float64],
    measurement_error: float | NDArray[np.float64] | None = None,
) -> StateSpace:
    """Convert LinearSolution to StateSpace for Kalman filtering.

    Args:
        solution: LinearSolution from gensys solver (y_t = T @ y_{t-1} + R @ u_t)
        calibration: Calibration object with shock covariances
        observables: Either:
            - list[str]: Variable names to observe (builds selection matrix)
            - NDArray: Custom Z matrix (n_obs x n_states)
        measurement_error: Measurement error specification:
            - None: No measurement error (H = 0)
            - float: Scalar variance (H = σ²I)
            - NDArray: Full covariance matrix (n_obs x n_obs)

    Returns:
        StateSpace ready for Kalman filtering

    Raises:
        InvalidObservableError: If observable name not found in state variables
        StateSpaceError: If dimensions don't match

    Example:
        >>> solution = solve_linear(linearize(model, ss, cal))
        >>> # Observe 'y' and 'pi' directly
        >>> state_space = to_state_space(solution, cal, observables=['y', 'pi'])
        >>> # With measurement error variance
        >>> state_space = to_state_space(
        ...     solution, cal,
        ...     observables=['y', 'pi'],
        ...     measurement_error=0.001**2
        ... )
    """
    # Extract state-space matrices from solution
    T = solution.T
    R = solution.R
    state_names = solution.var_names
    shock_names = solution.shock_names
    steady_state = solution.steady_state

    n_states = T.shape[0]

    # Build observation matrix Z
    if isinstance(observables, list):
        # Build selection matrix from variable names
        obs_names = observables
        Z = _build_selection_matrix(state_names, obs_names)
    else:
        # User provided custom Z matrix
        Z = np.asarray(observables, dtype=np.float64)
        if Z.ndim != 2:
            raise StateSpaceError(f"Z matrix must be 2D, got {Z.ndim}D")
        if Z.shape[1] != n_states:
            raise StateSpaceError(
                f"Z matrix has {Z.shape[1]} columns, expected {n_states} (n_states)"
            )
        # Generate generic observation names for custom Z
        obs_names = [f"obs_{i+1}" for i in range(Z.shape[0])]

    n_obs = Z.shape[0]

    # Build process covariance Q = R @ Σ @ R'
    Sigma = calibration.shock_cov_matrix(shock_names)
    Q = R @ Sigma @ R.T

    # Build measurement covariance H
    H = _build_measurement_cov(measurement_error, n_obs)

    # Create StateSpace
    ss = StateSpace(
        T=T,
        R=R,
        Q=Q,
        Z=Z,
        H=H,
        state_names=list(state_names),
        obs_names=list(obs_names),
        shock_names=list(shock_names),
        steady_state=dict(steady_state),
    )

    # Validate before returning
    validate_state_space(ss)

    return ss
