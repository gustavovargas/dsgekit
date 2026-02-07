"""First-order stochastic solution for non-linear DSGE models.

Builds a first-order perturbation (policy linear in state deviations) around
the provided steady state:

    A * y_{t-1} + B * y_t + C * E_t[y_{t+1}] + D * u_t = 0

and solves it via the linear QZ/BK stack.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from dsgekit.derivatives import (
    DerivativeCoordinate,
    DerivativeStack,
    linearization_coordinates,
)

if TYPE_CHECKING:
    from dsgekit.model.calibration import Calibration
    from dsgekit.model.ir import ModelIR
    from dsgekit.model.steady_state import SteadyState
    from dsgekit.solvers.linear import LinearSolution
    from dsgekit.transforms.linearize import LinearizedSystem


@dataclass(slots=True)
class FirstOrderApproximation:
    """First-order perturbation output before solution."""

    linear_system: LinearizedSystem
    coordinates: list[DerivativeCoordinate]
    backend_name: str
    jacobian: NDArray[np.float64]


def _build_eval_context(
    model: ModelIR,
    steady_state: SteadyState,
    calibration: Calibration,
):
    from dsgekit.model.equations import EvalContext

    timings = list(range(-model.lead_lag.max_lag, model.lead_lag.max_lead + 1))
    shocks = dict.fromkeys(model.shock_names, 0.0)
    return EvalContext.from_steady_state(
        steady_state=steady_state.values,
        parameters=calibration.parameters,
        shocks=shocks,
        timings=timings,
    )


def linearize_first_order(
    model: ModelIR,
    steady_state: SteadyState,
    calibration: Calibration,
    *,
    derivative_backend: str = "numeric",
    eps: float = 1e-6,
) -> FirstOrderApproximation:
    """Compute first-order perturbation matrices for a non-linear model."""
    from dsgekit.exceptions import SolverError
    from dsgekit.transforms.linearize import LinearizedSystem

    model.validate()
    if model.lead_lag.max_lag > 1 or model.lead_lag.max_lead > 1:
        raise SolverError(
            "First-order perturbation solver supports timings up to one lag/lead "
            f"(max_lag={model.lead_lag.max_lag}, max_lead={model.lead_lag.max_lead})."
        )

    coordinates = linearization_coordinates(model)
    stack = DerivativeStack(derivative_backend, eps=eps)
    context = _build_eval_context(model, steady_state, calibration)
    J = stack.jacobian(model, context, coordinates)

    n_eq = model.n_equations
    n_vars = model.n_variables
    n_shocks = model.n_shocks

    A = np.zeros((n_eq, n_vars), dtype=np.float64)
    B = np.zeros((n_eq, n_vars), dtype=np.float64)
    C = np.zeros((n_eq, n_vars), dtype=np.float64)
    D = np.zeros((n_eq, n_shocks), dtype=np.float64)

    var_to_col = {name: i for i, name in enumerate(model.variable_names)}
    shock_to_col = {name: i for i, name in enumerate(model.shock_names)}

    for j, coord in enumerate(coordinates):
        col_values = J[:, j]
        if coord.kind == "var":
            col = var_to_col[coord.name]
            if coord.timing == -1:
                A[:, col] = col_values
            elif coord.timing == 0:
                B[:, col] = col_values
            elif coord.timing == 1:
                C[:, col] = col_values
            else:
                raise SolverError(
                    "Unsupported variable timing in first-order perturbation: "
                    f"{coord.name}({coord.timing})"
                )
        elif coord.kind == "shock":
            D[:, shock_to_col[coord.name]] = col_values
        else:
            raise SolverError(
                "Parameter coordinates are not supported in first-order policy "
                f"linearization: {coord.name}"
            )

    linear_system = LinearizedSystem(
        A=A,
        B=B,
        C=C,
        D=D,
        var_names=model.variable_names,
        shock_names=model.shock_names,
        steady_state=steady_state.values.copy(),
    )
    return FirstOrderApproximation(
        linear_system=linear_system,
        coordinates=coordinates,
        backend_name=stack.backend_name,
        jacobian=J,
    )


def solve_first_order(
    model: ModelIR,
    steady_state: SteadyState,
    calibration: Calibration,
    *,
    derivative_backend: str = "numeric",
    eps: float = 1e-6,
    n_predetermined: int | None = None,
    tol: float = 1e-10,
    check_bk: bool = True,
) -> LinearSolution:
    """Solve first-order stochastic policy around steady state."""
    from dsgekit.solvers.linear import solve_linear

    approximation = linearize_first_order(
        model,
        steady_state,
        calibration,
        derivative_backend=derivative_backend,
        eps=eps,
    )
    if n_predetermined is None:
        n_predetermined = model.n_predetermined

    solution = solve_linear(
        approximation.linear_system,
        n_predetermined=n_predetermined,
        tol=tol,
        check_bk=check_bk,
    )
    solution.bk_meta = {
        **solution.bk_meta,
        "solver": "first_order_perturbation",
        "derivative_backend": approximation.backend_name,
        "n_derivative_coordinates": len(approximation.coordinates),
    }
    return solution
