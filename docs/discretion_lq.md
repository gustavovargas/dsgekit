# Discretionary LQ Policy

`dsgekit` provides a linear-quadratic (LQ) discretionary policy solver using a receding-horizon setup.

Core API:

- `solve_discretion_lq(...)`
- `solve_discretion_from_linear_solution(...)`
- `simulate_discretion_path(...)`

## Problem

Given linear dynamics

- `x_{t+1} = A x_t + B u_t`

and period loss

- `x_t' Q x_t + u_t' R u_t`

the discretionary solver computes a finite-horizon plan each period and applies only the first control:

- `u_t = -K x_t`

The horizon length is configurable with `horizon`.

## Generic LQ Example

```python
import numpy as np
from dsgekit.solvers import solve_discretion_lq, simulate_discretion_path

A = np.array([[1.0]])
B = np.array([[1.0]])
Q = np.array([[1.0]])
R = np.array([[0.5]])

discretion = solve_discretion_lq(A, B, Q, R, beta=0.99, horizon=8)
path = simulate_discretion_path(discretion, initial_state=[2.0], n_periods=40)

print(discretion.summary())
print(path.summary())
```

## From `LinearSolution`

```python
from dsgekit import load_model
from dsgekit.solvers import (
    solve_discretion_from_linear_solution,
    solve_linear,
)
from dsgekit.transforms import linearize

model, cal, ss = load_model("tests/fixtures/models/ar1.yaml")
solution = solve_linear(linearize(model, ss, cal))

discretion = solve_discretion_from_linear_solution(
    solution,
    control_shocks=["e"],
    state_weights={"y": 1.0},
    control_weights={"e": 0.5},
    beta=0.99,
    horizon=8,
)

print(discretion.K)
```

## Notes

- `Q`, `R`, and `terminal_weight` must be symmetric.
- `R` should be positive definite for a well-posed control problem.
- Larger `horizon` values move the feedback rule closer to the infinite-horizon commitment solution.
