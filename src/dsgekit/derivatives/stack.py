"""Unified derivative stack for DSGE equation systems.

Provides a common API to compute Jacobians, Hessians, and third-order
derivative tensors of model residual equations, with swappable backends.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from dsgekit.model.equations import EvalContext
    from dsgekit.model.ir import ModelIR

CoordinateKind = Literal["var", "shock", "param"]


@dataclass(frozen=True, slots=True)
class DerivativeCoordinate:
    """Coordinate in equation state-space for derivative computations."""

    kind: CoordinateKind
    name: str
    timing: int = 0

    def __post_init__(self) -> None:
        if self.kind not in {"var", "shock", "param"}:
            raise ValueError(f"Unknown coordinate kind: {self.kind}")
        if self.kind != "var" and self.timing != 0:
            raise ValueError("Only variable coordinates can carry non-zero timing")


def var_coord(name: str, timing: int = 0) -> DerivativeCoordinate:
    return DerivativeCoordinate(kind="var", name=name, timing=timing)


def shock_coord(name: str) -> DerivativeCoordinate:
    return DerivativeCoordinate(kind="shock", name=name, timing=0)


def param_coord(name: str) -> DerivativeCoordinate:
    return DerivativeCoordinate(kind="param", name=name, timing=0)


def linearization_coordinates(model: ModelIR) -> list[DerivativeCoordinate]:
    """Coordinates aligned with the first-order linearization structure."""
    coords: list[DerivativeCoordinate] = []
    for var_name in model.variable_names:
        if model.lead_lag.is_predetermined(var_name):
            coords.append(var_coord(var_name, -1))
    for var_name in model.variable_names:
        coords.append(var_coord(var_name, 0))
    for var_name in model.variable_names:
        if model.lead_lag.is_forward_looking(var_name):
            coords.append(var_coord(var_name, 1))
    for shock_name in model.shock_names:
        coords.append(shock_coord(shock_name))
    return coords


def _get_coord_value(context: EvalContext, coord: DerivativeCoordinate) -> float:
    if coord.kind == "var":
        if coord.name not in context.variables:
            raise KeyError(f"Variable '{coord.name}' not in context")
        if coord.timing not in context.variables[coord.name]:
            raise KeyError(
                f"Variable '{coord.name}' timing {coord.timing} not in context"
            )
        return float(context.variables[coord.name][coord.timing])
    if coord.kind == "shock":
        if coord.name not in context.shocks:
            raise KeyError(f"Shock '{coord.name}' not in context")
        return float(context.shocks[coord.name])
    if coord.name not in context.parameters:
        raise KeyError(f"Parameter '{coord.name}' not in context")
    return float(context.parameters[coord.name])


def _set_coord_value(context: EvalContext, coord: DerivativeCoordinate, value: float) -> None:
    v = float(value)
    if coord.kind == "var":
        context.set_variable(coord.name, coord.timing, v)
    elif coord.kind == "shock":
        context.set_shock(coord.name, v)
    else:
        context.set_parameter(coord.name, v)


def _apply_values(
    context: EvalContext,
    coordinates: list[DerivativeCoordinate],
    values: NDArray[np.float64],
) -> None:
    for coord, val in zip(coordinates, values, strict=True):
        _set_coord_value(context, coord, float(val))


def _extract_values(
    context: EvalContext,
    coordinates: list[DerivativeCoordinate],
) -> NDArray[np.float64]:
    return np.array([_get_coord_value(context, c) for c in coordinates], dtype=np.float64)


class DerivativeBackend(ABC):
    """Backend interface for derivative computations."""

    name: str

    @abstractmethod
    def jacobian(
        self,
        model: ModelIR,
        context: EvalContext,
        coordinates: list[DerivativeCoordinate],
    ) -> NDArray[np.float64]:
        pass

    @abstractmethod
    def hessian(
        self,
        model: ModelIR,
        context: EvalContext,
        coordinates: list[DerivativeCoordinate],
    ) -> NDArray[np.float64]:
        pass

    @abstractmethod
    def third_order(
        self,
        model: ModelIR,
        context: EvalContext,
        coordinates: list[DerivativeCoordinate],
    ) -> NDArray[np.float64]:
        pass


class FiniteDifferenceBackend(DerivativeBackend):
    """Finite-difference backend for equation derivatives."""

    name = "numeric"

    def __init__(self, eps: float = 1e-6):
        self.eps = float(eps)

    def _evaluate_point(
        self,
        model: ModelIR,
        context: EvalContext,
        coordinates: list[DerivativeCoordinate],
        x: NDArray[np.float64],
        cache: dict[bytes, NDArray[np.float64]],
    ) -> NDArray[np.float64]:
        x_arr = np.asarray(x, dtype=np.float64).reshape(-1)
        key = x_arr.tobytes()
        if key not in cache:
            _apply_values(context, coordinates, x_arr)
            cache[key] = model.residuals(context)
        return cache[key]

    def _hessian_from_eval(
        self,
        eval_point,
        x0: NDArray[np.float64],
        n_eq: int,
        n_coords: int,
    ) -> NDArray[np.float64]:
        eps = self.eps
        H = np.zeros((n_eq, n_coords, n_coords), dtype=np.float64)
        f0 = eval_point(x0)

        for i in range(n_coords):
            ei = np.zeros(n_coords, dtype=np.float64)
            ei[i] = eps
            fpp = eval_point(x0 + ei + ei)
            fmm = eval_point(x0 - ei - ei)
            H[:, i, i] = (fpp - 2.0 * f0 + fmm) / (4.0 * eps * eps)

            for j in range(i + 1, n_coords):
                ej = np.zeros(n_coords, dtype=np.float64)
                ej[j] = eps
                fpp = eval_point(x0 + ei + ej)
                fpm = eval_point(x0 + ei - ej)
                fmp = eval_point(x0 - ei + ej)
                fmm = eval_point(x0 - ei - ej)
                hij = (fpp - fpm - fmp + fmm) / (4.0 * eps * eps)
                H[:, i, j] = hij
                H[:, j, i] = hij
        return H

    def jacobian(
        self,
        model: ModelIR,
        context: EvalContext,
        coordinates: list[DerivativeCoordinate],
    ) -> NDArray[np.float64]:
        n_eq = model.n_equations
        n_coords = len(coordinates)
        J = np.zeros((n_eq, n_coords), dtype=np.float64)
        eps = self.eps

        x0 = _extract_values(context, coordinates)
        cache: dict[bytes, NDArray[np.float64]] = {}

        def eval_point(x: NDArray[np.float64]) -> NDArray[np.float64]:
            return self._evaluate_point(model, context, coordinates, x, cache)

        try:
            for i in range(n_coords):
                ei = np.zeros(n_coords, dtype=np.float64)
                ei[i] = eps
                f_plus = eval_point(x0 + ei)
                f_minus = eval_point(x0 - ei)
                J[:, i] = (f_plus - f_minus) / (2.0 * eps)
        finally:
            _apply_values(context, coordinates, x0)

        return J

    def hessian(
        self,
        model: ModelIR,
        context: EvalContext,
        coordinates: list[DerivativeCoordinate],
    ) -> NDArray[np.float64]:
        x0 = _extract_values(context, coordinates)
        cache: dict[bytes, NDArray[np.float64]] = {}
        n_eq = model.n_equations
        n_coords = len(coordinates)

        def eval_point(x: NDArray[np.float64]) -> NDArray[np.float64]:
            return self._evaluate_point(model, context, coordinates, x, cache)

        try:
            return self._hessian_from_eval(eval_point, x0, n_eq, n_coords)
        finally:
            _apply_values(context, coordinates, x0)

    def third_order(
        self,
        model: ModelIR,
        context: EvalContext,
        coordinates: list[DerivativeCoordinate],
    ) -> NDArray[np.float64]:
        x0 = _extract_values(context, coordinates)
        cache: dict[bytes, NDArray[np.float64]] = {}
        n_eq = model.n_equations
        n_coords = len(coordinates)
        eps = self.eps

        def eval_point(x: NDArray[np.float64]) -> NDArray[np.float64]:
            return self._evaluate_point(model, context, coordinates, x, cache)

        third = np.zeros((n_eq, n_coords, n_coords, n_coords), dtype=np.float64)

        try:
            for k in range(n_coords):
                ek = np.zeros(n_coords, dtype=np.float64)
                ek[k] = eps
                h_plus = self._hessian_from_eval(eval_point, x0 + ek, n_eq, n_coords)
                h_minus = self._hessian_from_eval(eval_point, x0 - ek, n_eq, n_coords)
                third[:, :, :, k] = (h_plus - h_minus) / (2.0 * eps)
        finally:
            _apply_values(context, coordinates, x0)

        return third


class SympyBackend(DerivativeBackend):
    """Symbolic backend powered by SymPy."""

    name = "sympy"

    def __init__(self):
        try:
            import sympy as sp
        except ImportError as exc:  # pragma: no cover - tested via branch checks
            raise ImportError(
                "SymPy backend requires sympy. Install with: pip install dsgekit[sym]"
            ) from exc
        self._sp = sp

    @staticmethod
    def _coord_key(coord: DerivativeCoordinate) -> tuple[str, str, int]:
        return (coord.kind, coord.name, coord.timing)

    def _symbol_name(self, key: tuple[str, str, int]) -> str:
        kind, name, timing = key
        if kind == "var":
            t = f"m{abs(timing)}" if timing < 0 else f"p{timing}"
            return f"var__{name}__{t}"
        if kind == "shock":
            return f"shock__{name}"
        return f"param__{name}"

    def _build_symbol_map(
        self,
        model: ModelIR,
        context: EvalContext,
    ) -> tuple[dict[tuple[str, str, int], object], dict[object, float]]:
        symbol_by_key: dict[tuple[str, str, int], object] = {}

        def sym_for_key(key: tuple[str, str, int]):
            if key not in symbol_by_key:
                symbol_by_key[key] = self._sp.Symbol(self._symbol_name(key), real=True)
            return symbol_by_key[key]

        for eq in model.equations:
            for tv in eq.get_variables():
                sym_for_key(("var", tv.name, tv.timing))
            for sh in eq.get_shocks():
                sym_for_key(("shock", sh.name, 0))
            for pa in eq.get_parameters():
                sym_for_key(("param", pa.name, 0))

        subs: dict[object, float] = {}
        for key, symbol in symbol_by_key.items():
            kind, name, timing = key
            if kind == "var":
                value = context.variables[name][timing]
            elif kind == "shock":
                value = context.shocks[name]
            else:
                value = context.parameters[name]
            subs[symbol] = float(value)

        return symbol_by_key, subs

    def _expr_to_sympy(self, expr, symbol_by_key):
        from dsgekit.model.equations import (
            BinaryOp,
            Constant,
            FunctionCall,
            ParameterRef,
            ShockRef,
            UnaryOp,
            VariableRef,
        )

        if isinstance(expr, Constant):
            return self._sp.Float(expr.value)
        if isinstance(expr, VariableRef):
            key = ("var", expr.timed_var.name, expr.timed_var.timing)
            return symbol_by_key[key]
        if isinstance(expr, ShockRef):
            key = ("shock", expr.shock.name, 0)
            return symbol_by_key[key]
        if isinstance(expr, ParameterRef):
            key = ("param", expr.param.name, 0)
            return symbol_by_key[key]
        if isinstance(expr, UnaryOp):
            operand = self._expr_to_sympy(expr.operand, symbol_by_key)
            if expr.op == "-":
                return -operand
            if expr.op == "+":
                return +operand
            raise ValueError(f"Unsupported unary operator: {expr.op}")
        if isinstance(expr, BinaryOp):
            left = self._expr_to_sympy(expr.left, symbol_by_key)
            right = self._expr_to_sympy(expr.right, symbol_by_key)
            if expr.op == "+":
                return left + right
            if expr.op == "-":
                return left - right
            if expr.op == "*":
                return left * right
            if expr.op == "/":
                return left / right
            if expr.op == "^":
                return left**right
            raise ValueError(f"Unsupported binary operator: {expr.op}")
        if isinstance(expr, FunctionCall):
            args = [self._expr_to_sympy(a, symbol_by_key) for a in expr.args]
            name = expr.name.lower()
            if name in {"log", "ln"}:
                return self._sp.log(args[0])
            if name == "exp":
                return self._sp.exp(args[0])
            if name == "sqrt":
                return self._sp.sqrt(args[0])
            if name == "abs":
                return self._sp.Abs(args[0])
            if name == "sin":
                return self._sp.sin(args[0])
            if name == "cos":
                return self._sp.cos(args[0])
            if name == "tan":
                return self._sp.tan(args[0])
            if name == "pow":
                return args[0] ** args[1]
            if name == "min":
                return self._sp.Min(*args)
            if name == "max":
                return self._sp.Max(*args)
            raise ValueError(f"Unsupported function for symbolic backend: {expr.name}")
        raise TypeError(f"Unsupported expression node: {type(expr)!r}")

    def _prepare(
        self,
        model: ModelIR,
        context: EvalContext,
        coordinates: list[DerivativeCoordinate],
    ) -> tuple[list[object], list[object], dict[object, float]]:
        symbol_by_key, subs = self._build_symbol_map(model, context)

        coord_symbols: list[object] = []
        for coord in coordinates:
            key = self._coord_key(coord)
            if key not in symbol_by_key:
                symbol_by_key[key] = self._sp.Symbol(self._symbol_name(key), real=True)
                subs[symbol_by_key[key]] = _get_coord_value(context, coord)
            coord_symbols.append(symbol_by_key[key])

        eq_exprs = [self._expr_to_sympy(eq.expression, symbol_by_key) for eq in model.equations]
        return eq_exprs, coord_symbols, subs

    def jacobian(
        self,
        model: ModelIR,
        context: EvalContext,
        coordinates: list[DerivativeCoordinate],
    ) -> NDArray[np.float64]:
        eq_exprs, coord_symbols, subs = self._prepare(model, context, coordinates)
        n_eq = len(eq_exprs)
        n_coords = len(coord_symbols)
        J = np.zeros((n_eq, n_coords), dtype=np.float64)

        for e_idx, expr in enumerate(eq_exprs):
            for c_idx, sym in enumerate(coord_symbols):
                deriv = self._sp.diff(expr, sym)
                J[e_idx, c_idx] = float(deriv.evalf(subs=subs))
        return J

    def hessian(
        self,
        model: ModelIR,
        context: EvalContext,
        coordinates: list[DerivativeCoordinate],
    ) -> NDArray[np.float64]:
        eq_exprs, coord_symbols, subs = self._prepare(model, context, coordinates)
        n_eq = len(eq_exprs)
        n_coords = len(coord_symbols)
        H = np.zeros((n_eq, n_coords, n_coords), dtype=np.float64)

        for e_idx, expr in enumerate(eq_exprs):
            for i, sym_i in enumerate(coord_symbols):
                for j, sym_j in enumerate(coord_symbols):
                    deriv = self._sp.diff(expr, sym_i, sym_j)
                    H[e_idx, i, j] = float(deriv.evalf(subs=subs))
        return H

    def third_order(
        self,
        model: ModelIR,
        context: EvalContext,
        coordinates: list[DerivativeCoordinate],
    ) -> NDArray[np.float64]:
        eq_exprs, coord_symbols, subs = self._prepare(model, context, coordinates)
        n_eq = len(eq_exprs)
        n_coords = len(coord_symbols)
        T = np.zeros((n_eq, n_coords, n_coords, n_coords), dtype=np.float64)

        for e_idx, expr in enumerate(eq_exprs):
            for i, sym_i in enumerate(coord_symbols):
                for j, sym_j in enumerate(coord_symbols):
                    for k, sym_k in enumerate(coord_symbols):
                        deriv = self._sp.diff(expr, sym_i, sym_j, sym_k)
                        T[e_idx, i, j, k] = float(deriv.evalf(subs=subs))
        return T


class DerivativeStack:
    """Unified API for Jacobian/Hessian/third-order derivatives."""

    def __init__(
        self,
        backend: str | DerivativeBackend = "numeric",
        *,
        eps: float = 1e-6,
    ):
        if isinstance(backend, str):
            key = backend.lower()
            if key in {"numeric", "finite_diff", "finite_difference", "fd"}:
                self.backend = FiniteDifferenceBackend(eps=eps)
            elif key in {"sympy", "symbolic"}:
                self.backend = SympyBackend()
            else:
                raise ValueError(
                    f"Unknown derivative backend '{backend}'. "
                    "Supported: numeric, sympy"
                )
        else:
            self.backend = backend

    @property
    def backend_name(self) -> str:
        return self.backend.name

    def _normalize_coords(
        self,
        coordinates: list[DerivativeCoordinate],
    ) -> list[DerivativeCoordinate]:
        if not coordinates:
            raise ValueError("At least one derivative coordinate is required")
        if len(set(coordinates)) != len(coordinates):
            raise ValueError("Derivative coordinates must be unique")
        return list(coordinates)

    def jacobian(
        self,
        model: ModelIR,
        context: EvalContext,
        coordinates: list[DerivativeCoordinate],
    ) -> NDArray[np.float64]:
        coords = self._normalize_coords(coordinates)
        return self.backend.jacobian(model, context, coords)

    def hessian(
        self,
        model: ModelIR,
        context: EvalContext,
        coordinates: list[DerivativeCoordinate],
    ) -> NDArray[np.float64]:
        coords = self._normalize_coords(coordinates)
        return self.backend.hessian(model, context, coords)

    def third_order(
        self,
        model: ModelIR,
        context: EvalContext,
        coordinates: list[DerivativeCoordinate],
    ) -> NDArray[np.float64]:
        coords = self._normalize_coords(coordinates)
        return self.backend.third_order(model, context, coords)
