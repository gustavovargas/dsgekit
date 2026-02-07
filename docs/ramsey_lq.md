# Ramsey LQ Policy

`dsgekit` provides a linear-quadratic (LQ) commitment solver for Ramsey-style policy design.

Core API:

- `solve_ramsey_lq(...)`
- `solve_ramsey_from_linear_solution(...)`
- `simulate_ramsey_path(...)`

## Problem

Given linear dynamics

- `x_{t+1} = A x_t + B u_t`

and discounted quadratic objective

- `min E sum_t beta^t [x_t' Q x_t + u_t' R u_t]`

the solver returns the optimal commitment feedback:

- `u_t = -K x_t`

## Generic LQ Example

```python
import numpy as np
from dsgekit.solvers import solve_ramsey_lq, simulate_ramsey_path

A = np.array([[1.0]])
B = np.array([[1.0]])
Q = np.array([[1.0]])
R = np.array([[0.5]])

ramsey = solve_ramsey_lq(A, B, Q, R, beta=0.99)
path = simulate_ramsey_path(ramsey, initial_state=[2.0], n_periods=40)

print(ramsey.summary())
print(path.summary())
```

## From `LinearSolution`

```python
from dsgekit import load_model
from dsgekit.solvers import (
    solve_linear,
    solve_ramsey_from_linear_solution,
)
from dsgekit.transforms import linearize

model, cal, ss = load_model("tests/fixtures/models/ar1.yaml")
solution = solve_linear(linearize(model, ss, cal))

ramsey = solve_ramsey_from_linear_solution(
    solution,
    control_shocks=["e"],
    state_weights={"y": 1.0},
    control_weights={"e": 0.5},
    beta=0.99,
)

print(ramsey.K)
```

## Notes

- `Q` and `R` must be symmetric; `R` should be positive definite for a well-posed control problem.
- In `solve_ramsey_from_linear_solution(...)`, selected shock innovations are treated as control channels.
- This solver targets linear-quadratic commitment core workflows; extensions to discretionary policy remain separate.
