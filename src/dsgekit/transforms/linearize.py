"""Linearization of DSGE models around steady state.

Linearizes the model:
    E_t[f(y_{t-1}, y_t, y_{t+1}, u_t, θ)] = 0

to obtain:
    A * ŷ_{t-1} + B * ŷ_t + C * ŷ_{t+1} + D * u_t = 0

where ŷ = y - y_ss (deviations from steady state).

The matrices A, B, C, D are the Jacobians of f with respect to
y_{t-1}, y_t, y_{t+1}, and u_t, evaluated at steady state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from dsgekit.model.calibration import Calibration
    from dsgekit.model.ir import ModelIR
    from dsgekit.model.steady_state import SteadyState


@dataclass
class LinearizedSystem:
    """Linearized DSGE system matrices.

    The system is: A * y_{t-1} + B * y_t + C * y_{t+1} + D * u_t = 0

    All matrices are n_eq x n_vars (or n_eq x n_shocks for D).

    Attributes:
        A: Jacobian w.r.t. y_{t-1} (lagged variables)
        B: Jacobian w.r.t. y_t (current variables)
        C: Jacobian w.r.t. y_{t+1} (lead variables)
        D: Jacobian w.r.t. u_t (shocks)
        var_names: Ordered variable names
        shock_names: Ordered shock names
        steady_state: Steady state values used for linearization
    """

    A: NDArray[np.float64]  # n_eq x n_vars, coeffs on y_{t-1}
    B: NDArray[np.float64]  # n_eq x n_vars, coeffs on y_t
    C: NDArray[np.float64]  # n_eq x n_vars, coeffs on y_{t+1}
    D: NDArray[np.float64]  # n_eq x n_shocks, coeffs on u_t
    var_names: list[str]
    shock_names: list[str]
    steady_state: dict[str, float]

    @property
    def n_equations(self) -> int:
        return self.B.shape[0]

    @property
    def n_variables(self) -> int:
        return self.B.shape[1]

    @property
    def n_shocks(self) -> int:
        return self.D.shape[1]

    def summary(self) -> str:
        """Return summary of linearized system."""
        lines = [
            "Linearized System:",
            f"  Equations: {self.n_equations}",
            f"  Variables: {self.n_variables}",
            f"  Shocks: {self.n_shocks}",
            f"  A (lag) non-zeros: {np.count_nonzero(self.A)}",
            f"  B (current) non-zeros: {np.count_nonzero(self.B)}",
            f"  C (lead) non-zeros: {np.count_nonzero(self.C)}",
            f"  D (shocks) non-zeros: {np.count_nonzero(self.D)}",
        ]
        return "\n".join(lines)


def linearize(
    model: ModelIR,
    steady_state: SteadyState,
    calibration: Calibration,
    eps: float = 1e-6,
) -> LinearizedSystem:
    """Linearize model around steady state using numerical derivatives.

    Uses central finite differences for numerical Jacobian computation.

    Args:
        model: Validated ModelIR
        steady_state: Steady state values
        calibration: Parameter calibration
        eps: Step size for finite differences

    Returns:
        LinearizedSystem with Jacobian matrices
    """
    from dsgekit.model.equations import EvalContext

    var_names = model.variable_names
    shock_names = model.shock_names
    n_vars = len(var_names)
    n_shocks = len(shock_names)
    n_eq = model.n_equations

    # Initialize matrices
    A = np.zeros((n_eq, n_vars))  # y_{t-1}
    B = np.zeros((n_eq, n_vars))  # y_t
    C = np.zeros((n_eq, n_vars))  # y_{t+1}
    D = np.zeros((n_eq, n_shocks))  # u_t

    # Base evaluation context at steady state
    ss_vals = steady_state.values
    ctx = EvalContext(parameters=calibration.parameters.copy())
    for name in var_names:
        ss_val = ss_vals[name]
        # Fixed timings used by current linear model: t-1, t, t+1.
        ctx.variables[name] = {-1: ss_val, 0: ss_val, 1: ss_val}
    for name in shock_names:
        ctx.shocks[name] = 0.0

    predetermined = set(model.predetermined_variable_names)
    forward_looking = set(model.forward_looking_variable_names)

    # Compute Jacobians using central differences

    # A: derivatives w.r.t. y_{t-1}
    for j, var_name in enumerate(var_names):
        if var_name not in predetermined:
            continue  # Column stays zero

        base = ctx.variables[var_name][-1]
        ctx.variables[var_name][-1] = base + eps
        res_plus = model.residuals(ctx)
        ctx.variables[var_name][-1] = base - eps
        res_minus = model.residuals(ctx)
        ctx.variables[var_name][-1] = base
        A[:, j] = (res_plus - res_minus) / (2 * eps)

    # B: derivatives w.r.t. y_t
    for j, var_name in enumerate(var_names):
        base = ctx.variables[var_name][0]
        ctx.variables[var_name][0] = base + eps
        res_plus = model.residuals(ctx)
        ctx.variables[var_name][0] = base - eps
        res_minus = model.residuals(ctx)
        ctx.variables[var_name][0] = base
        B[:, j] = (res_plus - res_minus) / (2 * eps)

    # C: derivatives w.r.t. y_{t+1}
    for j, var_name in enumerate(var_names):
        if var_name not in forward_looking:
            continue  # Column stays zero

        base = ctx.variables[var_name][1]
        ctx.variables[var_name][1] = base + eps
        res_plus = model.residuals(ctx)
        ctx.variables[var_name][1] = base - eps
        res_minus = model.residuals(ctx)
        ctx.variables[var_name][1] = base
        C[:, j] = (res_plus - res_minus) / (2 * eps)

    # D: derivatives w.r.t. u_t
    for j, shock_name in enumerate(shock_names):
        base = ctx.shocks[shock_name]
        ctx.shocks[shock_name] = base + eps
        res_plus = model.residuals(ctx)
        ctx.shocks[shock_name] = base - eps
        res_minus = model.residuals(ctx)
        ctx.shocks[shock_name] = base
        D[:, j] = (res_plus - res_minus) / (2 * eps)

    return LinearizedSystem(
        A=A,
        B=B,
        C=C,
        D=D,
        var_names=var_names,
        shock_names=shock_names,
        steady_state=ss_vals.copy(),
    )


def check_linearization(
    linear_sys: LinearizedSystem,
    tol: float = 1e-8,
) -> dict[str, bool]:
    """Check properties of the linearized system.

    Returns dict with diagnostic flags.
    """

    results = {}

    # Check B is invertible (needed for some solution methods)
    try:
        B_cond = np.linalg.cond(linear_sys.B)
        results["B_invertible"] = B_cond < 1 / tol
        results["B_condition_number"] = B_cond
    except np.linalg.LinAlgError:
        results["B_invertible"] = False
        results["B_condition_number"] = np.inf

    # Check for zero rows (redundant or missing equations)
    zero_rows_B = np.all(np.abs(linear_sys.B) < tol, axis=1)
    results["has_zero_rows"] = np.any(zero_rows_B)

    # Check for zero columns (variables that don't appear)
    all_matrices = np.abs(linear_sys.A) + np.abs(linear_sys.B) + np.abs(linear_sys.C)
    zero_cols = np.all(all_matrices < tol, axis=0)
    results["has_unused_variables"] = np.any(zero_cols)

    return results
