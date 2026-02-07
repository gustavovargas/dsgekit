"""Exception hierarchy for dsgekit.

All exceptions inherit from DSGEKitError for easy catching.
Each exception type provides actionable information to help users
diagnose and fix problems.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


class DSGEKitError(Exception):
    """Base exception for all dsgekit errors."""

    pass


# =============================================================================
# Model Specification Errors
# =============================================================================


class ModelSpecError(DSGEKitError):
    """Error in model specification (variables, parameters, equations)."""

    pass


class UndeclaredSymbolError(ModelSpecError):
    """Reference to a symbol that was not declared."""

    def __init__(self, symbol: str, context: str | None = None):
        self.symbol = symbol
        self.context = context
        msg = f"Undeclared symbol: '{symbol}'"
        if context:
            msg += f" in {context}"
        super().__init__(msg)


class DuplicateSymbolError(ModelSpecError):
    """Symbol declared more than once."""

    def __init__(self, symbol: str, first_type: str, second_type: str):
        self.symbol = symbol
        self.first_type = first_type
        self.second_type = second_type
        msg = (
            f"Symbol '{symbol}' declared as both {first_type} and {second_type}. "
            "Each symbol must have a unique declaration."
        )
        super().__init__(msg)


class InvalidTimingError(ModelSpecError):
    """Invalid timing specification for a variable (e.g., x(-2) when only x(-1) allowed)."""

    def __init__(self, variable: str, timing: int, allowed: Sequence[int]):
        self.variable = variable
        self.timing = timing
        self.allowed = allowed
        allowed_str = ", ".join(str(t) for t in allowed)
        msg = (
            f"Invalid timing {timing} for variable '{variable}'. "
            f"Allowed timings: {allowed_str}"
        )
        super().__init__(msg)


class EquationCountError(ModelSpecError):
    """Number of equations does not match number of endogenous variables."""

    def __init__(self, n_equations: int, n_endogenous: int):
        self.n_equations = n_equations
        self.n_endogenous = n_endogenous
        msg = (
            f"Model has {n_equations} equations but {n_endogenous} endogenous variables. "
            "These must be equal for a well-defined system."
        )
        super().__init__(msg)


# =============================================================================
# Steady State Errors
# =============================================================================


class SteadyStateError(DSGEKitError):
    """Error computing or validating steady state."""

    pass


class SteadyStateNotFoundError(SteadyStateError):
    """Numerical solver failed to find steady state."""

    def __init__(
        self,
        message: str = "Steady state solver did not converge",
        residuals: dict[str, float] | None = None,
        iterations: int | None = None,
    ):
        self.residuals = residuals
        self.iterations = iterations
        msg = message
        if iterations is not None:
            msg += f" after {iterations} iterations"
        if residuals:
            # Show equations with largest residuals
            sorted_res = sorted(residuals.items(), key=lambda x: abs(x[1]), reverse=True)
            top_3 = sorted_res[:3]
            msg += ". Largest residuals: "
            msg += ", ".join(f"{eq}={res:.2e}" for eq, res in top_3)
        super().__init__(msg)


class SteadyStateValidationError(SteadyStateError):
    """Provided steady state values do not satisfy model equations."""

    def __init__(self, residuals: dict[str, float], tolerance: float = 1e-6):
        self.residuals = residuals
        self.tolerance = tolerance
        violations = {eq: res for eq, res in residuals.items() if abs(res) > tolerance}
        msg = f"Steady state validation failed. {len(violations)} equation(s) violated:\n"
        for eq, res in sorted(violations.items(), key=lambda x: abs(x[1]), reverse=True):
            msg += f"  {eq}: residual = {res:.2e}\n"
        super().__init__(msg.rstrip())


# =============================================================================
# Linearization Errors
# =============================================================================


class LinearizationError(DSGEKitError):
    """Error during model linearization."""

    pass


class SingularJacobianError(LinearizationError):
    """Jacobian matrix is singular or nearly singular."""

    def __init__(self, matrix_name: str, condition_number: float | None = None):
        self.matrix_name = matrix_name
        self.condition_number = condition_number
        msg = f"Jacobian matrix '{matrix_name}' is singular or ill-conditioned"
        if condition_number is not None:
            msg += f" (condition number: {condition_number:.2e})"
        msg += ". Check model specification and steady state values."
        super().__init__(msg)


# =============================================================================
# Blanchard-Kahn / Solution Errors
# =============================================================================


class BlanchardKahnError(DSGEKitError):
    """Blanchard-Kahn conditions not satisfied."""

    pass


class NoStableSolutionError(BlanchardKahnError):
    """No stable solution exists (too few stable eigenvalues)."""

    def __init__(
        self,
        n_stable: int,
        n_predetermined: int,
        eigenvalues: Sequence[complex] | None = None,
    ):
        self.n_stable = n_stable
        self.n_predetermined = n_predetermined
        self.eigenvalues = eigenvalues
        msg = (
            f"No stable solution exists. "
            f"Found {n_stable} stable eigenvalue(s) but need {n_predetermined} "
            f"(number of predetermined variables). "
            "The model is explosive."
        )
        super().__init__(msg)


class IndeterminacyError(BlanchardKahnError):
    """Multiple stable solutions exist (too many stable eigenvalues)."""

    def __init__(
        self,
        n_stable: int,
        n_predetermined: int,
        eigenvalues: Sequence[complex] | None = None,
    ):
        self.n_stable = n_stable
        self.n_predetermined = n_predetermined
        self.eigenvalues = eigenvalues
        msg = (
            f"Indeterminacy: multiple stable solutions exist. "
            f"Found {n_stable} stable eigenvalue(s) but need exactly {n_predetermined} "
            f"(number of predetermined variables). "
            "The equilibrium is not uniquely determined."
        )
        super().__init__(msg)


class SolverError(DSGEKitError):
    """General solver error."""

    pass


# =============================================================================
# State-Space Errors
# =============================================================================


class StateSpaceError(DSGEKitError):
    """Error in state-space representation."""

    pass


class InvalidObservableError(StateSpaceError):
    """Observable specification is invalid."""

    def __init__(self, obs_name: str, reason: str):
        self.obs_name = obs_name
        self.reason = reason
        super().__init__(f"Invalid observable '{obs_name}': {reason}")


# =============================================================================
# I/O and Format Errors
# =============================================================================


class FormatError(DSGEKitError):
    """Error parsing or processing a model file."""

    pass


class ParseError(FormatError):
    """Syntax error in model file."""

    def __init__(self, message: str, line: int | None = None, column: int | None = None):
        self.line = line
        self.column = column
        loc = ""
        if line is not None:
            loc = f" at line {line}"
            if column is not None:
                loc += f", column {column}"
        super().__init__(f"Parse error{loc}: {message}")


class UnsupportedFormatFeatureError(FormatError):
    """Model file uses a feature not supported by dsgekit."""

    def __init__(self, feature: str, format_name: str = "mod"):
        self.feature = feature
        self.format_name = format_name
        msg = (
            f"Unsupported {format_name} feature: '{feature}'. "
            "See documentation for supported features."
        )
        super().__init__(msg)


# =============================================================================
# Estimation Errors
# =============================================================================


class EstimationError(DSGEKitError):
    """Error during estimation."""

    pass


class FilterError(EstimationError):
    """Error in Kalman filter or smoother."""

    pass


class NonPositiveDefiniteError(FilterError):
    """Covariance matrix is not positive definite."""

    def __init__(self, matrix_name: str, time_step: int | None = None):
        self.matrix_name = matrix_name
        self.time_step = time_step
        msg = f"Matrix '{matrix_name}' is not positive definite"
        if time_step is not None:
            msg += f" at time step {time_step}"
        super().__init__(msg)


class LikelihoodError(EstimationError):
    """Error computing likelihood (e.g., NaN, -inf)."""

    def __init__(self, value: float, reason: str = ""):
        self.value = value
        self.reason = reason
        msg = f"Invalid likelihood value: {value}"
        if reason:
            msg += f". {reason}"
        super().__init__(msg)
