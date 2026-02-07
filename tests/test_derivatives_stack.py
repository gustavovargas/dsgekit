"""Tests for the unified derivative stack backends."""

from __future__ import annotations

import numpy as np
import pytest

from dsgekit import load_model
from dsgekit.derivatives import (
    DerivativeStack,
    linearization_coordinates,
    shock_coord,
    var_coord,
)
from dsgekit.io.formats.python_api import ModelBuilder
from dsgekit.model.equations import EvalContext


def _context_at_steady_state(model, cal, ss):
    ctx = EvalContext(parameters=cal.parameters.copy())
    for v in model.variable_names:
        s = float(ss.values[v])
        for t in (-1, 0, 1):
            ctx.set_variable(v, t, s)
    for shock in model.shock_names:
        ctx.set_shock(shock, 0.0)
    return ctx


def _build_cubic_model():
    return (
        ModelBuilder("cubic")
        .var("y")
        .varexo("e")
        .param("a", 0.5)
        .equation("y = a * y(-1)^3 + e")
        .initval(y=0.0)
        .shock_stderr(e=0.01)
        .build()
    )


class TestFiniteDifferenceBackend:
    def test_ar1_derivatives_match_known_linear_coeffs(self, models_dir):
        model, cal, ss = load_model(models_dir / "ar1.yaml")
        ctx = _context_at_steady_state(model, cal, ss)
        coords = [var_coord("y", -1), var_coord("y", 0), shock_coord("e")]

        stack = DerivativeStack("numeric", eps=1e-6)
        J = stack.jacobian(model, ctx, coords)
        H = stack.hessian(model, ctx, coords)
        T = stack.third_order(model, ctx, coords)

        np.testing.assert_allclose(J, np.array([[-0.9, 1.0, -1.0]]), atol=1e-6)
        np.testing.assert_allclose(H, 0.0, atol=1e-9)
        np.testing.assert_allclose(T, 0.0, atol=1e-4)

    def test_third_order_nonzero_for_cubic_equation(self):
        model, cal, ss = _build_cubic_model()
        ctx = _context_at_steady_state(model, cal, ss)
        coords = [var_coord("y", -1), var_coord("y", 0), shock_coord("e")]

        stack = DerivativeStack("numeric", eps=1e-5)
        J = stack.jacobian(model, ctx, coords)
        H = stack.hessian(model, ctx, coords)
        T = stack.third_order(model, ctx, coords)

        # At steady state y=0:
        # f = y - a*y(-1)^3 - e
        # d f / d y(-1) = 0, d^2 f / d y(-1)^2 = 0, d^3 f / d y(-1)^3 = -6a = -3
        np.testing.assert_allclose(J[0, 0], 0.0, atol=1e-8)
        np.testing.assert_allclose(H[0, 0, 0], 0.0, atol=1e-8)
        np.testing.assert_allclose(T[0, 0, 0, 0], -3.0, atol=1e-6)


class TestBackendSelection:
    def test_invalid_backend_name_raises(self):
        with pytest.raises(ValueError, match="Unknown derivative backend"):
            DerivativeStack("unknown-backend")

    def test_linearization_coordinates_for_nk(self, models_dir):
        model, *_ = load_model(models_dir / "nk.yaml")
        coords = linearization_coordinates(model)

        assert len(coords) == 9
        assert coords[0] == var_coord("i", -1)
        assert coords[1:4] == [
            var_coord("x", 0),
            var_coord("pi", 0),
            var_coord("i", 0),
        ]
        assert coords[4:6] == [var_coord("x", 1), var_coord("pi", 1)]


class TestSympyBackend:
    def test_sympy_backend_switch(self):
        try:
            import sympy  # noqa: F401
        except ImportError:
            with pytest.raises(ImportError, match="SymPy backend requires sympy"):
                DerivativeStack("sympy")
            return

        model, cal, ss = _build_cubic_model()
        ctx = _context_at_steady_state(model, cal, ss)
        coords = [var_coord("y", -1), var_coord("y", 0), shock_coord("e")]

        # Third-order finite differences are roundoff-sensitive with very small eps.
        num = DerivativeStack("numeric", eps=1e-5)
        sym = DerivativeStack("sympy")

        np.testing.assert_allclose(
            num.jacobian(model, ctx, coords),
            sym.jacobian(model, ctx, coords),
            atol=1e-8,
        )
        np.testing.assert_allclose(
            num.hessian(model, ctx, coords),
            sym.hessian(model, ctx, coords),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            num.third_order(model, ctx, coords),
            sym.third_order(model, ctx, coords),
            atol=1e-4,
        )
