"""Gensys-style linear rational expectations solver.

Solves the linearized system:
    A * y_{t-1} + B * y_t + C * y_{t+1} + D * u_t = 0

Using QZ (generalized Schur) decomposition and Blanchard-Kahn conditions.

The solution is in the form:
    y_t = T * y_{t-1} + R * u_t

where T is the transition matrix and R is the shock impact matrix.

References:
- Sims, C. (2002). Solving Linear Rational Expectations Models.
- Klein, P. (2000). Using the generalized Schur form to solve a
  multivariate linear rational expectations model.
- Blanchard, O. & Kahn, C. (1980). The Solution of Linear Difference
  Models under Rational Expectations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from scipy import linalg

if TYPE_CHECKING:
    from dsgekit.transforms.linearize import LinearizedSystem


@dataclass
class LinearSolution:
    """Solution to a linear rational expectations model.

    The solution is: y_t = T * y_{t-1} + R * u_t

    Attributes:
        T: Transition matrix (n_vars x n_vars)
        R: Shock impact matrix (n_vars x n_shocks)
        var_names: Variable names in order
        shock_names: Shock names in order
        steady_state: Steady state values
        eigenvalues: Generalized eigenvalues from QZ decomposition
        n_stable: Number of stable eigenvalues (|λ| < 1)
        n_unstable: Number of unstable eigenvalues (|λ| >= 1)
        n_predetermined: Number of predetermined (state) variables
        bk_meta: Metadata used for BK diagnostics and reproducible traces
    """

    T: NDArray[np.float64]
    R: NDArray[np.float64]
    var_names: list[str]
    shock_names: list[str]
    steady_state: dict[str, float]
    eigenvalues: NDArray[np.complex128] = field(default_factory=lambda: np.array([]))
    n_stable: int = 0
    n_unstable: int = 0
    n_predetermined: int = 0
    bk_meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_variables(self) -> int:
        return len(self.var_names)

    @property
    def n_shocks(self) -> int:
        return len(self.shock_names)

    def is_determinate(self) -> bool:
        """Check if solution is unique (determinate)."""
        return self.n_stable == self.n_predetermined

    def summary(self) -> str:
        """Return summary of the solution."""
        status = "DETERMINATE" if self.is_determinate() else "INDETERMINATE/NO SOLUTION"
        lines = [
            f"Linear Solution ({status}):",
            f"  Variables: {self.n_variables}",
            f"  Shocks: {self.n_shocks}",
            f"  Stable eigenvalues: {self.n_stable}",
            f"  Unstable eigenvalues: {self.n_unstable}",
            f"  Predetermined variables: {self.n_predetermined}",
        ]
        if len(self.eigenvalues) > 0:
            finite_eigs = self.eigenvalues[np.isfinite(self.eigenvalues)]
            if len(finite_eigs) > 0:
                lines.append(f"  Max |eigenvalue|: {np.max(np.abs(finite_eigs)):.4f}")
        return "\n".join(lines)


def _solve_no_leads(
    A: NDArray[np.float64],
    B: NDArray[np.float64],
    D: NDArray[np.float64],
    tol: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.complex128]]:
    """Solve system with no forward-looking variables (C = 0).

    System: A*y_{t-1} + B*y_t + D*u_t = 0
    Solution: y_t = -B^{-1}*A*y_{t-1} - B^{-1}*D*u_t
              y_t = T*y_{t-1} + R*u_t
    where T = -B^{-1}*A and R = -B^{-1}*D
    """
    from dsgekit.exceptions import SolverError

    try:
        B_inv = np.linalg.inv(B)
    except np.linalg.LinAlgError as err:
        raise SolverError("Matrix B is singular, cannot solve system") from err

    T = -B_inv @ A
    R = -B_inv @ D

    # Eigenvalues of T determine stability
    eigenvalues = np.linalg.eigvals(T)

    return T, R, eigenvalues


def _effective_stability_counts(
    eigenvalues: NDArray[np.complex128],
    n_structural_zero_expected: int,
    tol: float,
) -> tuple[int, int, dict[str, int | float]]:
    """Count stable/unstable roots, discounting structural zero roots.

    The companion representation used for lead-lag systems can introduce
    artificial zero roots when the lead matrix is rank-deficient.
    For BK classification we remove those structural zeros from the stable count.
    """
    finite_mask = np.isfinite(eigenvalues)
    mod = np.abs(eigenvalues)

    n_finite = int(np.sum(finite_mask))
    n_stable_raw = int(np.sum(finite_mask & (mod < 1.0)))
    n_unstable_raw = int(len(eigenvalues) - n_stable_raw)

    zero_tol = float(max(1e-12, 100.0 * tol))
    n_zero_candidates = int(np.sum(finite_mask & (mod <= zero_tol)))
    n_zero_removed = int(min(max(n_structural_zero_expected, 0), n_zero_candidates))

    n_stable = int(max(0, n_stable_raw - n_zero_removed))
    n_dynamic_finite = int(max(0, n_finite - n_zero_removed))
    n_unstable = int(max(0, n_dynamic_finite - n_stable))

    meta: dict[str, int | float] = {
        "n_finite_roots": n_finite,
        "n_stable_raw": n_stable_raw,
        "n_unstable_raw": n_unstable_raw,
        "n_structural_zero_expected": int(max(n_structural_zero_expected, 0)),
        "n_zero_candidates": n_zero_candidates,
        "n_zero_removed": n_zero_removed,
        "zero_tol": zero_tol,
        "n_stable_effective": n_stable,
        "n_unstable_effective": n_unstable,
        "n_dynamic_finite_effective": n_dynamic_finite,
    }
    return n_stable, n_unstable, meta


def _solve_with_leads(
    A: NDArray[np.float64],
    B: NDArray[np.float64],
    C: NDArray[np.float64],
    D: NDArray[np.float64],
    n_predetermined: int,
    tol: float,
    check_bk: bool,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.complex128],
    int,
    int,
    dict[str, Any],
]:
    """Solve system with forward-looking variables using QZ decomposition.

    System: A*y_{t-1} + B*y_t + C*y_{t+1} + D*u_t = 0

    Uses Klein (2000) method with QZ decomposition.
    """
    from dsgekit.exceptions import (
        IndeterminacyError,
        NoStableSolutionError,
        SolverError,
    )

    n_vars = B.shape[0]
    n_shocks = D.shape[1]

    # Build companion form matrices
    # Rewrite as: Γ₀ * z_{t+1} = Γ₁ * z_t + Ψ * u_t
    # where z_t = [y_t]
    #
    # From: A*y_{t-1} + B*y_t + C*y_{t+1} + D*u_t = 0
    # We get: C*y_{t+1} = -B*y_t - A*y_{t-1} - D*u_t
    #
    # State vector: s_t = [y_{t-1}; y_t]
    # Then: [0 C; I 0] * [y_t; y_{t+1}] = [-A -B; 0 I] * [y_{t-1}; y_t] + [-D; 0] * u_t
    #
    # Or equivalently in the form where we can use ordqz properly:
    # Γ₀ * s_{t+1} = Γ₁ * s_t + Ψ * u_t

    # Use the standard form for QZ:
    # [I  0 ] [y_t  ]     [0   I] [y_{t-1}]   [0]
    # [-B -C] [y_{t+1}] = [-A  0] [y_t    ] + [D] u_t

    Gamma0 = np.block([
        [np.eye(n_vars), np.zeros((n_vars, n_vars))],
        [-B, -C]
    ])

    Gamma1 = np.block([
        [np.zeros((n_vars, n_vars)), np.eye(n_vars)],
        [-A, np.zeros((n_vars, n_vars))]
    ])

    # QZ decomposition with stable roots first (inside unit circle)
    try:
        _, _, alpha, beta, _, Z = linalg.ordqz(
            Gamma0, Gamma1,
            sort='iuc',  # |eigenvalue| < 1 first
            output='complex'
        )
    except linalg.LinAlgError as err:
        raise SolverError(f"QZ decomposition failed: {err}") from err

    # Generalized eigenvalues for (Gamma0, Gamma1): lambda = alpha / beta.
    # When beta is ~0 the root is treated as infinite (outside unit circle).
    eigenvalues = np.full(alpha.shape, np.inf + 0j, dtype=np.complex128)
    finite_mask = np.abs(beta) > tol
    eigenvalues[finite_mask] = alpha[finite_mask] / beta[finite_mask]

    rank_a = int(np.linalg.matrix_rank(A, tol=tol))
    rank_c = int(np.linalg.matrix_rank(C, tol=tol))
    n_structural_zero_expected = n_vars - rank_c
    n_structural_inf_expected = n_vars - rank_a

    n_stable, n_unstable, count_meta = _effective_stability_counts(
        eigenvalues=eigenvalues,
        n_structural_zero_expected=n_structural_zero_expected,
        tol=tol,
    )
    bk_meta: dict[str, Any] = {
        "method": "qz_companion",
        "tol": float(tol),
        "n_variables": int(n_vars),
        "n_shocks": int(n_shocks),
        "rank_a": rank_a,
        "rank_c": rank_c,
        "n_structural_inf_expected": int(max(n_structural_inf_expected, 0)),
        **count_meta,
    }

    # Blanchard-Kahn conditions
    if check_bk:
        if n_stable < n_predetermined:
            raise NoStableSolutionError(
                n_stable=n_stable,
                n_predetermined=n_predetermined,
                eigenvalues=eigenvalues,
            )
        elif n_stable > n_predetermined:
            raise IndeterminacyError(
                n_stable=n_stable,
                n_predetermined=n_predetermined,
                eigenvalues=eigenvalues,
            )

    # Extract solution using the stable block
    # Z partitioned as [Z11 Z12; Z21 Z22] where subscript 1 = stable
    Z11 = Z[:n_vars, :n_stable]
    Z21 = Z[n_vars:, :n_stable]

    # Transition matrix: involves mapping from stable subspace
    # T = Z21 @ inv(Z11) for the reduced form
    try:
        if n_stable > 0:
            T = np.real(Z21 @ np.linalg.inv(Z11))
        else:
            T = np.zeros((n_vars, n_vars))
    except np.linalg.LinAlgError:
        T = np.real(Z21 @ np.linalg.pinv(Z11))

    # Impact matrix R from the contemporaneous response to shocks
    # From B*y_t + C*E[y_{t+1}] = -A*y_{t-1} - D*u_t
    # Using y_{t+1} = T*y_t + R*u_{t+1} and taking expectations:
    # B*y_t + C*T*y_t = -A*y_{t-1} - D*u_t
    # (B + C*T)*y_t = -A*y_{t-1} - D*u_t
    # So response to shock: (B + C*T)*R = -D
    BCT = B + C @ T
    try:
        R = -np.linalg.solve(BCT, D)
    except np.linalg.LinAlgError:
        R = -np.linalg.pinv(BCT) @ D

    return T, R, eigenvalues, n_stable, n_unstable, bk_meta


def solve_linear(
    linear_sys: LinearizedSystem,
    n_predetermined: int | None = None,
    tol: float = 1e-10,
    check_bk: bool = True,
) -> LinearSolution:
    """Solve linearized DSGE using QZ decomposition.

    Args:
        linear_sys: Linearized system matrices (A, B, C, D)
        n_predetermined: Number of predetermined variables (if None, inferred from A)
        tol: Tolerance for numerical comparisons
        check_bk: Whether to check Blanchard-Kahn conditions

    Returns:
        LinearSolution with transition and shock matrices

    Raises:
        BlanchardKahnError: If BK conditions not satisfied
        SolverError: If numerical issues occur
    """
    from dsgekit.exceptions import (
        IndeterminacyError,
        NoStableSolutionError,
    )

    A = linear_sys.A
    B = linear_sys.B
    C = linear_sys.C
    D = linear_sys.D

    n_vars = B.shape[0]

    # Infer number of predetermined variables
    if n_predetermined is None:
        n_predetermined = int(np.sum(np.any(np.abs(A) > tol, axis=0)))

    # Check if there are forward-looking variables
    has_leads = np.any(np.abs(C) > tol)

    if not has_leads:
        # Simple case: no forward-looking variables
        T, R, eigenvalues = _solve_no_leads(A, B, D, tol)
        rank_a = int(np.linalg.matrix_rank(A, tol=tol))
        n_stable, n_unstable, count_meta = _effective_stability_counts(
            eigenvalues=eigenvalues,
            n_structural_zero_expected=n_vars - rank_a,
            tol=tol,
        )
        bk_meta: dict[str, Any] = {
            "method": "no_leads",
            "tol": float(tol),
            "n_variables": int(n_vars),
            "n_shocks": int(D.shape[1]),
            "rank_a": rank_a,
            **count_meta,
        }

        if check_bk:
            if n_stable < n_predetermined:
                raise NoStableSolutionError(
                    n_stable=n_stable,
                    n_predetermined=n_predetermined,
                    eigenvalues=eigenvalues,
                )
            if n_stable > n_predetermined:
                raise IndeterminacyError(
                    n_stable=n_stable,
                    n_predetermined=n_predetermined,
                    eigenvalues=eigenvalues,
                )
    else:
        # General case with forward-looking variables
        T, R, eigenvalues, n_stable, n_unstable, bk_meta = _solve_with_leads(
            A, B, C, D, n_predetermined, tol, check_bk
        )

    return LinearSolution(
        T=T,
        R=R,
        var_names=linear_sys.var_names,
        shock_names=linear_sys.shock_names,
        steady_state=linear_sys.steady_state,
        eigenvalues=eigenvalues,
        n_stable=n_stable,
        n_unstable=n_unstable,
        n_predetermined=n_predetermined,
        bk_meta=bk_meta,
    )


def eigenvalue_analysis(
    eigenvalues: NDArray[np.complex128],
    var_names: list[str] | None = None,
) -> str:
    """Generate a report on eigenvalues.

    Args:
        eigenvalues: Array of generalized eigenvalues
        var_names: Optional variable names for context

    Returns:
        Human-readable eigenvalue analysis
    """
    lines = ["Eigenvalue Analysis:", "=" * 40]

    # Sort by modulus
    finite_mask = np.isfinite(eigenvalues)
    finite_eigs = eigenvalues[finite_mask]
    inf_count = np.sum(~finite_mask)

    sorted_idx = np.argsort(np.abs(finite_eigs))
    sorted_eigs = finite_eigs[sorted_idx]

    lines.append(f"{'#':<4} {'Modulus':<12} {'Real':<12} {'Imag':<12} {'Type':<10}")
    lines.append("-" * 50)

    for i, eig in enumerate(sorted_eigs):
        mod = np.abs(eig)
        re = np.real(eig)
        im = np.imag(eig)
        typ = "stable" if mod < 1.0 else "unstable"
        lines.append(f"{i+1:<4} {mod:<12.4f} {re:<12.4f} {im:<12.4f} {typ:<10}")

    if inf_count > 0:
        lines.append(f"... + {inf_count} infinite eigenvalue(s)")

    n_stable = np.sum(np.abs(eigenvalues[finite_mask]) < 1.0)
    n_unstable = len(eigenvalues) - n_stable

    lines.append("-" * 50)
    lines.append(f"Stable: {n_stable}, Unstable: {n_unstable}")

    return "\n".join(lines)
