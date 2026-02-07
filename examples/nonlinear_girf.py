"""State-dependent GIRF example using second-order pruning.

Run from repository root:
    python examples/nonlinear_girf.py
"""

from __future__ import annotations

import numpy as np

from dsgekit.io.formats.python_api import ModelBuilder
from dsgekit.simulate import girf_pruned_second_order, simulate_pruned_second_order_path
from dsgekit.solvers import solve_second_order


def build_quadratic_ar1():
    return (
        ModelBuilder("quadratic_ar1")
        .var("y")
        .varexo("e")
        .param("rho", 0.8)
        .param("a", 0.2)
        .equation("y = rho * y(-1) + a * y(-1)^2 + e")
        .initval(y=0.0)
        .shock_stderr(e=0.02)
        .build()
    )


def main() -> None:
    model, cal, ss = build_quadratic_ar1()
    solution = solve_second_order(model, ss, cal, derivative_backend="numeric", eps=1e-6)

    girf_low = girf_pruned_second_order(
        solution,
        cal,
        "e",
        periods=10,
        shock_size=0.1,
        n_draws=1000,
        seed=42,
        initial_state=np.array([0.0]),
    )
    girf_high = girf_pruned_second_order(
        solution,
        cal,
        "e",
        periods=10,
        shock_size=0.1,
        n_draws=1000,
        seed=42,
        initial_state=np.array([1.0]),
    )

    print("== GIRF at low state (y(-1)=0.0) ==")
    print(girf_low.data.to_string())
    print()
    print("== GIRF at high state (y(-1)=1.0) ==")
    print(girf_high.data.to_string())
    print()

    shock_path = np.zeros((12, 1))
    shock_path[0, 0] = 0.1
    shock_path[5, 0] = -0.05
    path = simulate_pruned_second_order_path(
        solution,
        shock_path,
        initial_state=np.array([0.3]),
    )

    print("== State-dependent path for custom shocks ==")
    print(path.data.to_string())


if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True)
    main()
